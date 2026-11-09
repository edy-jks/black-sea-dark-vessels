import numpy as np


def to_db(linear, floor=1e-6):
    """Linear backscatter -> dB. Non-finite and non-positive values become NaN."""
    x = np.asarray(linear, dtype="float32")
    out = np.full(x.shape, np.nan, dtype="float32")
    ok = np.isfinite(x) & (x > 0)
    out[ok] = 10.0 * np.log10(np.maximum(x[ok], floor))
    return out
