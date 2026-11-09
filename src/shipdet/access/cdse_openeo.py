"""Run SAR calibration on the CDSE back-end and download only a small clipped GeoTIFF."""
import datetime as dt

import numpy as np
import openeo
import rasterio


def fetch_chip(bounds, scene_start, scene_end, out_path, backend="openeo.dataspace.copernicus.eu",
               resolution_m=10, utm_epsg=32635, pad_s=30):
    """Backscatter (sigma0, linear, VV and VH) for one acquisition, clipped to `bounds`."""
    west, south, east, north = bounds
    t0 = dt.datetime.fromisoformat(scene_start.replace("Z", "+00:00")) - dt.timedelta(seconds=pad_s)
    t1 = dt.datetime.fromisoformat(scene_end.replace("Z", "+00:00")) + dt.timedelta(seconds=pad_s)

    conn = openeo.connect(backend).authenticate_oidc()   # browser login; the token is stored outside the repo
    cube = conn.load_collection(
        "SENTINEL1_GRD",
        spatial_extent={"west": west, "south": south, "east": east, "north": north, "crs": "EPSG:4326"},
        temporal_extent=[t0.isoformat(), t1.isoformat()],
        bands=["VV", "VH"],
        properties=[openeo.collection_property("polarisation") == "VV&VH"],
    )
    cube = cube.sar_backscatter(coefficient="sigma0-ellipsoid", elevation_model="COPERNICUS_30", noise_removal=True)
    cube = cube.resample_spatial(resolution=resolution_m, projection=utm_epsg)
    cube = cube.reduce_dimension(dimension="t", reducer="first")
    cube.execute_batch(outputfile=str(out_path), out_format="GTiff", title="shipdet pilot chip")
    return out_path


def load_chip(path):
    """Return (vv_linear, vh_linear, transform, crs). Band 1 = VV, band 2 = VH."""
    with rasterio.open(path) as src:
        vv = src.read(1).astype("float32")
        vh = src.read(2).astype("float32") if src.count >= 2 else None
        nodata, transform, crs = src.nodata, src.transform, src.crs
    for arr in (vv, vh):
        if arr is not None and nodata is not None:
            arr[arr == nodata] = np.nan
    return vv, vh, transform, crs