def prf(n_tp, n_fp, n_fn):
    """Precision, recall and F1 from counts. Undefined values are NaN."""
    nan = float("nan")
    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) else nan
    recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) else nan
    if precision != precision or recall != recall or (precision + recall) == 0:
        f1 = nan
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"tp": n_tp, "fp": n_fp, "fn": n_fn, "precision": precision, "recall": recall, "f1": f1}
