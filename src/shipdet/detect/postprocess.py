import geopandas as gpd
import numpy as np
from scipy import ndimage
from shapely.geometry import Point

COLUMNS = ["n_pixels", "peak_db", "contrast_db", "approx_length_m", "geometry"]


def mask_to_detections(mask, db, transform, crs, bg_mean=None, merge_dilate_px=2, min_pixels=3, max_pixels=2000):
    """Group flagged pixels into single detections. Returns a GeoDataFrame in the raster CRS."""
    grown = ndimage.binary_dilation(mask, iterations=merge_dilate_px) if merge_dilate_px > 0 else mask
    labels, _ = ndimage.label(grown, structure=np.ones((3, 3), dtype=int))
    res_m = abs(transform.a)

    rows = []
    for lab, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        sub = (labels[sl] == lab) & mask[sl]
        n_px = int(sub.sum())
        if n_px < min_pixels or n_px > max_pixels:
            continue
        rr, cc = np.nonzero(sub)
        vals = db[sl][sub]
        w = 10.0 ** (vals / 10.0)                              # weight by linear intensity
        row = sl[0].start + float((rr * w).sum() / w.sum())
        col = sl[1].start + float((cc * w).sum() / w.sum())
        x, y = transform * (col + 0.5, row + 0.5)              # pixel centre
        peak = float(vals.max())
        r0, c0 = int(round(row)), int(round(col))
        contrast = np.nan
        if bg_mean is not None and np.isfinite(bg_mean[r0, c0]):
            contrast = peak - float(bg_mean[r0, c0])
        extent_px = max(int(np.ptp(rr)), int(np.ptp(cc))) + 1
        rows.append({"n_pixels": n_px, "peak_db": peak, "contrast_db": contrast,
                     "approx_length_m": extent_px * res_m, "geometry": Point(x, y)})

    gdf = gpd.GeoDataFrame(rows, columns=COLUMNS, geometry="geometry", crs=crs)
    gdf.insert(0, "detect_id", range(len(gdf)))
    return gdf
