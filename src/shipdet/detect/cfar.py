import numpy as np
from scipy import ndimage


def _window_sums(arr, size):
    """Sum of `arr` over a size x size window (zero padded)."""
    return ndimage.uniform_filter(arr, size=size, mode="constant", cval=0.0) * (size * size)


def two_parameter_cfar(db, valid, guard_px=41, background_px=81, k=5.0, min_db=None, min_valid_fraction=0.6):
    """Two-parameter CFAR on dB values.

    A pixel is flagged when (x - mean_bg) / std_bg > k, where the background is the ring between
    the guard window and the outer window. Returns (flag_mask, background_mean, background_std).
    """
    if guard_px % 2 == 0 or background_px % 2 == 0 or guard_px >= background_px:
        raise ValueError("guard_px and background_px must be odd, with guard_px < background_px")

    x = np.where(valid, db, 0.0).astype("float64")
    v = valid.astype("float64")

    n = _window_sums(v, background_px) - _window_sums(v, guard_px)
    s = _window_sums(x, background_px) - _window_sums(x, guard_px)
    q = _window_sums(x * x, background_px) - _window_sums(x * x, guard_px)

    with np.errstate(divide="ignore", invalid="ignore"):
        mean = s / n
        var = q / n - mean ** 2
        std = np.sqrt(np.maximum(var, 0.01))      # floor avoids division by ~0 in flat areas
        z = (db - mean) / std

    ring_area = background_px ** 2 - guard_px ** 2
    flagged = valid & (n >= min_valid_fraction * ring_area) & np.isfinite(z) & (z > k)
    if min_db is not None:
        flagged &= db > min_db
    return flagged, mean, std
