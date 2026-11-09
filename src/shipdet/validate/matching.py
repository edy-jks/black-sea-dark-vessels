import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def match_one_to_one(ours_xy, ref_xy, max_distance_m):
    """Greedy nearest-neighbour one-to-one matching. Inputs: (n, 2) arrays in metres.

    Returns a list of (ours_index, ref_index, distance_m). Each point is used at most once.
    """
    if len(ours_xy) == 0 or len(ref_xy) == 0:
        return []
    tree = cKDTree(ref_xy)
    candidates = []
    for i, js in enumerate(tree.query_ball_point(ours_xy, r=max_distance_m)):
        for j in js:
            candidates.append((float(np.hypot(*(ours_xy[i] - ref_xy[j]))), i, j))
    candidates.sort()
    used_i, used_j, out = set(), set(), []
    for d, i, j in candidates:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        out.append((i, j, d))
    return out


def match_detections(ours, ref, max_distance_m):
    """GeoDataFrames in a metric CRS with a reset index. Returns pairs, ours_only and ref_only (positional)."""
    def xy(gdf):
        if len(gdf) == 0:
            return np.empty((0, 2))
        return np.column_stack([gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()])

    pairs = pd.DataFrame(match_one_to_one(xy(ours), xy(ref), max_distance_m),
                         columns=["ours_idx", "ref_idx", "distance_m"])
    return {
        "pairs": pairs,
        "ours_only": sorted(set(range(len(ours))) - set(pairs["ours_idx"])),
        "ref_only": sorted(set(range(len(ref))) - set(pairs["ref_idx"])),
    }