"""
wer.py — Weighted Word Error Rate (WER) computation.

Global Score = (sum over samples of Edits_i * L_i**weight_factor)
               / (sum over samples of L_i**weight_factor)

Where:
    Edits_i = Word-level edit distance between the reference and
              hypothesis for sample i (Substitutions + Deletions +
              Insertions), computed by jiwer (rapidfuzz-backed).
    L_i     = number of words in the (jiwer-tokenized) reference for
              sample i.

Missing predictions (no matching row in the prediction CSV, or an
empty/NaN prediction) are scored as if the model had produced an
empty string — they are never skipped.

Conversational role markers such as "user", "assistant",
"<|user|>", "<|assistant|>" are stripped from both reference and
hypothesis text before scoring.
"""

import re
import pandas as pd
import jiwer

# -----------------------------
# Config: candidate column names
# -----------------------------
_ID_COL_CANDIDATES = ["ID", "id", "Id", "file_id", "filename", "audio_id"]
_TEXT_COL_CANDIDATES = [
    "transcription",
    "transcript",
    "text",
    "sentence",
    "reference",
    "hypothesis",
    "prediction",
    "target",
    "Target"
]

# Strip role/format markers that commonly leak out of LLM-style outputs,
# e.g. "<|user|>", "<|assistant|>", or bare lines like "user:" / "assistant:".
_TAG_PATTERN = re.compile(
    r"<\|?\s*(user|assistant|system)\s*\|?>|\b(user|assistant|system)\s*:",
    flags=re.IGNORECASE,
)


def _strip_tags(text: str) -> str:
    """Remove conversational role markers/tags from a string."""
    if text is None:
        return ""
    text = str(text)
    text = _TAG_PATTERN.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _detect_column(df: pd.DataFrame, candidates, role: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not find a {role} column in columns={list(df.columns)}. "
        f"Expected one of {candidates}."
    )


def _word_edits_and_length(ref_text: str, hyp_text: str):
    """
    Run jiwer on a single (reference, hypothesis) pair and return
    (edits, L_i) where edits = S + D + I and L_i is the number of
    words in jiwer's tokenized reference.
    """
    ref_text = _strip_tags(ref_text)
    hyp_text = _strip_tags(hyp_text)

    if not ref_text:
        # Empty reference: L_i = 0, edits = number of hypothesis words (insertions).
        hyp_len = len(hyp_text.split()) if hyp_text else 0
        return hyp_len, 0

    if not hyp_text:
        # jiwer requires a non-empty hypothesis string; score as all-deletions.
        L_i = len(ref_text.split())
        return L_i, L_i

    out = jiwer.process_words(ref_text, hyp_text)
    edits = out.substitutions + out.deletions + out.insertions
    L_i = len(out.references[0])
    return edits, L_i


def _load_pairs(gt_csv_path: str, pred_csv_path: str, id_col=None, text_col=None):
    """
    Returns:
        pairs: list of (sample_id, ref_text, hyp_text)
        matched: number of reference samples that had a matching prediction row
        total_reference: total number of reference samples
    """
    gt_df = pd.read_csv(gt_csv_path)
    pred_df = pd.read_csv(pred_csv_path)

    gt_id_col = id_col or _detect_column(gt_df, _ID_COL_CANDIDATES, "id (ground truth)")
    pred_id_col = id_col or _detect_column(pred_df, _ID_COL_CANDIDATES, "id (prediction)")

    gt_text_col = text_col or _detect_column(gt_df, _TEXT_COL_CANDIDATES, "text (ground truth)")
    pred_text_col = text_col or _detect_column(pred_df, _TEXT_COL_CANDIDATES, "text (prediction)")

    gt_df = gt_df[[gt_id_col, gt_text_col]].rename(columns={gt_id_col: "_id", gt_text_col: "_ref"})
    pred_df = pred_df[[pred_id_col, pred_text_col]].rename(columns={pred_id_col: "_id", pred_text_col: "_hyp"})

    # Drop exact-duplicate prediction IDs, keeping the first occurrence.
    pred_df = pred_df.drop_duplicates(subset="_id", keep="first")

    merged = gt_df.merge(pred_df, on="_id", how="left")

    total_reference = len(merged)
    matched = int(merged["_hyp"].notna().sum())

    # Missing predictions -> treated as empty string, never skipped.
    merged["_hyp"] = merged["_hyp"].fillna("")

    pairs = list(zip(merged["_id"], merged["_ref"].astype(str), merged["_hyp"].astype(str)))
    return pairs, matched, total_reference


def compute_weighted_wer(
    gt_csv_path: str,
    pred_csv_path: str,
    weight_factor: float = 0.5,
    id_col: str = None,
    text_col: str = None,
):
    """
    Compute the weighted global Word Error Rate.

    Global Score = sum(Edits_i * L_i**weight_factor) / sum(L_i**weight_factor)

    Returns a dict with keys: score, matched, total_reference, total_edits,
    total_weight.
    """
    pairs, matched, total_reference = _load_pairs(gt_csv_path, pred_csv_path, id_col, text_col)

    total_weighted_edits = 0.0
    total_weight = 0.0
    total_edits = 0

    for _id, ref_text, hyp_text in pairs:
        edits, L_i = _word_edits_and_length(ref_text, hyp_text)

        w_i = (L_i ** weight_factor) if L_i > 0 else 0.0

        total_weighted_edits += edits * w_i
        total_weight += w_i
        total_edits += edits

    score = (total_weighted_edits / total_weight) if total_weight > 0 else 1.0

    return {
        "score": score,
        "matched": matched,
        "total_reference": total_reference,
        "total_edits": total_edits,
        "total_weight": total_weight,
    }