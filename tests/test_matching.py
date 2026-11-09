import numpy as np

from shipdet.validate.matching import match_one_to_one


def test_matches_within_distance_only():
    ours = np.array([[0.0, 0.0], [100.0, 0.0], [5000.0, 5000.0]])
    ref = np.array([[10.0, 0.0], [105.0, 0.0]])
    pairs = match_one_to_one(ours, ref, max_distance_m=50)
    assert sorted((i, j) for i, j, _ in pairs) == [(0, 0), (1, 1)]


def test_reference_point_is_never_used_twice():
    ours = np.array([[0.0, 0.0], [20.0, 0.0]])
    ref = np.array([[10.0, 0.0]])
    assert len(match_one_to_one(ours, ref, max_distance_m=50)) == 1
