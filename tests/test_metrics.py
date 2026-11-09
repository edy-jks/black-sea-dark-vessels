import math

from shipdet.validate.metrics import prf


def test_prf_basic():
    m = prf(n_tp=8, n_fp=2, n_fn=2)
    assert math.isclose(m["precision"], 0.8)
    assert math.isclose(m["recall"], 0.8)
    assert math.isclose(m["f1"], 0.8)


def test_prf_zero_division_gives_nan():
    m = prf(0, 0, 0)
    assert math.isnan(m["precision"]) and math.isnan(m["recall"]) and math.isnan(m["f1"])
