import numpy as np
from rasterio import features


def polygon_mask(geoms, shape, transform):
    """True inside the given shapely geometries (which must be in the raster CRS)."""
    geoms = list(geoms)
    if not geoms:
        return np.zeros(shape, dtype=bool)
    burned = features.rasterize(((g, 1) for g in geoms), out_shape=shape, transform=transform,
                                fill=0, dtype="uint8")
    return burned.astype(bool)


def valid_mask(db, exclude=None):
    """Pixels the detector may use: finite values, outside the optional `exclude` mask."""
    ok = np.isfinite(db)
    if exclude is not None:
        ok &= ~exclude
    return ok
