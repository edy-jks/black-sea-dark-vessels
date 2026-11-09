"""Read Global Fishing Watch SAR vessel detections and footprints from Data Download Portal files."""
import geopandas as gpd
import pandas as pd
from shapely import wkt


def _find_col(columns, candidates):
    lowered = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    return None


def _utc(value):
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def load_gfw_detections(csv_path, bounds, t_start=None, t_end=None, tolerance_s=0):
    """Return GFW detections as a GeoDataFrame (EPSG:4326) inside `bounds` (and the time window, if given).

    Adds a nullable boolean column `matched` (True = matched to an AIS identity).
    """
    df = pd.read_csv(csv_path)
    lat = _find_col(df.columns, ["lat", "latitude", "detect_lat"])
    lon = _find_col(df.columns, ["lon", "longitude", "detect_lon"])
    ts = _find_col(df.columns, ["timestamp", "detect_timestamp", "datetime", "time"])
    if None in (lat, lon, ts):
        raise ValueError(f"Cannot find lat/lon/timestamp in columns: {list(df.columns)}")
    df = df.rename(columns={lat: "lat", lon: "lon", ts: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    west, south, east, north = bounds
    df = df[df["lon"].between(west, east) & df["lat"].between(south, north)]
    if t_start is not None and t_end is not None:
        pad = pd.Timedelta(seconds=tolerance_s)
        df = df[(df["timestamp"] >= _utc(t_start) - pad) & (df["timestamp"] <= _utc(t_end) + pad)]
    df = df.reset_index(drop=True)

    id_col = _find_col(df.columns, ["ssvid", "mmsi", "vessel_id"])
    df["matched"] = df[id_col].notna() if id_col else pd.NA
    df["matched"] = df["matched"].astype("boolean")

    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326")


def load_footprints(csv_path):
    """Return the footprint polygons (WKT column) as a GeoSeries in EPSG:4326."""
    df = pd.read_csv(csv_path)
    starts = ("POLYGON", "MULTIPOLYGON")
    col = next((c for c in df.columns if df[c].astype(str).str.upper().str.startswith(starts).any()), None)
    if col is None:
        raise ValueError(f"No WKT polygon column found in: {list(df.columns)}")
    geoms = [wkt.loads(s) for s in df[col].dropna().astype(str) if s.upper().startswith(starts)]
    return gpd.GeoSeries(geoms, crs="EPSG:4326")
