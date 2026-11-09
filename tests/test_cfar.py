import numpy as np
from rasterio.transform import from_origin

from shipdet.detect.cfar import two_parameter_cfar
from shipdet.detect.postprocess import mask_to_detections


def make_sea(seed=0, shape=(300, 300)):
    rng = np.random.default_rng(seed)
    return rng.normal(-20.0, 1.5, shape).astype("float32")


def test_cfar_finds_one_bright_ship():
    db = make_sea()
    db[150:153, 150:153] = -2.0
    flagged, _, _ = two_parameter_cfar(db, np.isfinite(db), guard_px=21, background_px=41, k=5.0, min_db=-15.0)
    assert flagged[150:153, 150:153].all()
    assert flagged.sum() < 30


def test_cfar_flags_almost_nothing_on_pure_sea():
    db = make_sea(seed=1)
    flagged, _, _ = two_parameter_cfar(db, np.isfinite(db), guard_px=21, background_px=41, k=5.0, min_db=-15.0)
    assert flagged.sum() <= 2


def test_mask_to_detections_centroid_position():
    db = make_sea()
    db[100:103, 200:203] = -2.0
    mask = np.zeros(db.shape, dtype=bool)
    mask[100:103, 200:203] = True
    transform = from_origin(500000.0, 4900000.0, 10.0, 10.0)
    gdf = mask_to_detections(mask, db, transform, "EPSG:32635", merge_dilate_px=1, min_pixels=3)
    assert len(gdf) == 1
    assert abs(gdf.geometry.iloc[0].x - (500000.0 + 201.5 * 10)) < 1.0
    assert abs(gdf.geometry.iloc[0].y - (4900000.0 - 101.5 * 10)) < 1.0
