"""One-off helper: cut a large GFW monthly download down to the pilot area.

Run from the repo root:
    python scripts/extract_gfw_aoi.py data/cache/gfw_raw/<your_file>.csv
"""
import sys

import pandas as pd

from shipdet.access.gfw import _find_col
from shipdet.config import load_aoi, load_config, repo_path


def main(src):
    cfg = load_config()
    west, south, east, north = load_aoi(cfg).bounds
    out = repo_path(cfg["gfw"]["detections_csv"])
    first, kept, total = True, 0, 0
    for chunk in pd.read_csv(src, chunksize=500_000, low_memory=False):
        lat = _find_col(chunk.columns, ["lat", "latitude", "detect_lat"])
        lon = _find_col(chunk.columns, ["lon", "longitude", "detect_lon"])
        if lat is None or lon is None:
            raise SystemExit(f"Cannot find lat/lon columns. Columns are: {list(chunk.columns)}")
        total += len(chunk)
        sub = chunk[chunk[lon].between(west, east) & chunk[lat].between(south, north)]
        kept += len(sub)
        sub.to_csv(out, mode="w" if first else "a", header=first, index=False)
        first = False
    print(f"Read {total} rows, kept {kept} inside the area -> {out}")
    if kept == 0:
        print("Nothing inside the area. Try another month or widen config/aoi.geojson.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/extract_gfw_aoi.py <path to the GFW csv>")
    main(sys.argv[1])