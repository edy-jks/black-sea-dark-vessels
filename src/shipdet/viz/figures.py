import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_overview(db, transform, ours, ref, res, out_png, title):
    """Backscatter image with our detections (matched/unmatched) and GFW-only detections."""
    h, w = db.shape
    left, top = transform.c, transform.f
    right, bottom = left + transform.a * w, top + transform.e * h
    lo, hi = np.nanpercentile(db, [2, 99.5])

    ours_tp = ours.iloc[res["pairs"]["ours_idx"].to_numpy()]
    ours_fp = ours.iloc[res["ours_only"]]
    ref_fn = ref.iloc[res["ref_only"]]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(db, cmap="gray", vmin=lo, vmax=hi, extent=(left, right, bottom, top), origin="upper")
    ax.scatter(ours_tp.geometry.x, ours_tp.geometry.y, s=80, facecolors="none", edgecolors="lime",
               label=f"ours, matched to GFW ({len(ours_tp)})")
    ax.scatter(ours_fp.geometry.x, ours_fp.geometry.y, s=80, facecolors="none", edgecolors="red",
               label=f"ours only ({len(ours_fp)})")
    ax.scatter(ref_fn.geometry.x, ref_fn.geometry.y, s=90, marker="s", facecolors="none", edgecolors="cyan",
               label=f"GFW only ({len(ref_fn)})")
    ax.legend(loc="upper right")
    ax.set_title(title)
    ax.set_xlabel("Easting (m, UTM 35N)")
    ax.set_ylabel("Northing (m, UTM 35N)")
    fig.text(0.01, 0.01, "Contains modified Copernicus Sentinel data", fontsize=8)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
