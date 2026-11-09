"""Pilot entry point. Each step is re-runnable on its own:

    python scripts/run_pipeline.py gfw-summary
    python scripts/run_pipeline.py search
    python scripts/run_pipeline.py fetch
    python scripts/run_pipeline.py detect
    python scripts/run_pipeline.py validate
"""
import argparse

import geopandas as gpd
import pandas as pd

from shipdet.config import load_aoi, load_config, repo_dir, repo_path


def cmd_gfw_summary(cfg):
    from shipdet.access.gfw import load_gfw_detections

    aoi = load_aoi(cfg)
    gfw = load_gfw_detections(repo_path(cfg["gfw"]["detections_csv"]), aoi.bounds)
    print(f"{len(gfw)} GFW detections inside the AOI bounding box")
    print("Columns:", list(gfw.columns))
    counts = (gfw.groupby([gfw["timestamp"].dt.date.rename("date"),
                           gfw["timestamp"].dt.hour.rename("hour_utc")])
                 .size().sort_values(ascending=False).head(15))
    print(counts.to_string())


def cmd_search(cfg):
    from shipdet.access.cdse_stac import search_s1_scenes

    s = cfg["search"]
    rows = search_s1_scenes(s["stac_url"], s["collection"], load_aoi(cfg), s["start"], s["end"])
    df = pd.DataFrame(rows)
    if df.empty:
        print("No scenes found. Check the dates, the AOI and the STAC URL.")
        return
    ok = df[(df["mode"] == "IW") & df["polarizations"].str.contains("VV") &
            df["polarizations"].str.contains("VH") & df["covers_aoi"]]
    if ok.empty:
        print("No candidate passed the filters. First rows of the raw result:")
        print(df.head().to_string(index=False))
        return
    ok.to_csv(repo_path("results/tables/scene_candidates.csv"), index=False)
    print(ok.to_string(index=False))
    print("\nCopy scene_id, start and end of ONE row into config/study.yaml under pilot:")


def cmd_fetch(cfg):
    from shipdet.access.cdse_openeo import fetch_chip

    p, f = cfg["pilot"], cfg["fetch"]
    if not p["scene_start"] or not p["scene_end"]:
        raise SystemExit("Fill pilot.scene_start and pilot.scene_end in config/study.yaml first.")
    out = repo_path(f["chip_path"])
    if out.exists():
        print(f"{out} already exists, skipping (delete it to fetch again).")
        return
    fetch_chip(load_aoi(cfg).bounds, p["scene_start"], p["scene_end"], out,
               backend=f["openeo_backend"], resolution_m=f["resolution_m"], utm_epsg=f["utm_epsg"])
    print(f"Saved {out} ({out.stat().st_size / 1e6:.1f} MB)")


def cmd_detect(cfg):
    from shipdet.access.cdse_openeo import load_chip
    from shipdet.detect.cfar import two_parameter_cfar
    from shipdet.detect.postprocess import mask_to_detections
    from shipdet.preprocess.masks import polygon_mask, valid_mask
    from shipdet.preprocess.sar_calibration import to_db

    c = cfg["cfar"]
    vv, vh, transform, crs = load_chip(repo_path(cfg["fetch"]["chip_path"]))

    exclude = None
    excl_file = repo_path("data/external/exclusion.geojson")
    if excl_file.exists():
        exclude = polygon_mask(list(gpd.read_file(excl_file).to_crs(crs).geometry), vv.shape, transform)

    out_dir = repo_dir(cfg["outputs"]["detections_dir"])
    for pol, lin in {"VV": vv, "VH": vh}.items():
        if lin is None:
            continue
        db = to_db(lin)
        valid = valid_mask(db, exclude=exclude)
        flagged, bg_mean, _ = two_parameter_cfar(
            db, valid, guard_px=c["guard_px"], background_px=c["background_px"], k=c["k"],
            min_db=c["min_db"][pol], min_valid_fraction=c["min_valid_fraction"])
        gdf = mask_to_detections(flagged, db, transform, crs, bg_mean=bg_mean,
                                 merge_dilate_px=c["merge_dilate_px"], min_pixels=c["min_pixels"],
                                 max_pixels=c["max_pixels"])
        print(f"{pol}: {int(flagged.sum())} flagged pixels -> {len(gdf)} detections")
        if gdf.empty:
            continue
        path = out_dir / f"pilot_detections_{pol}.geojson"
        gdf.to_crs(4326).to_file(path, driver="GeoJSON")
        print(f"  saved {path}")


def cmd_validate(cfg):
    from shapely import union_all

    from shipdet.access.cdse_openeo import load_chip
    from shipdet.access.gfw import load_footprints, load_gfw_detections
    from shipdet.preprocess.sar_calibration import to_db
    from shipdet.validate.matching import match_detections
    from shipdet.validate.metrics import prf
    from shipdet.viz.figures import plot_overview

    p, g, v = cfg["pilot"], cfg["gfw"], cfg["validate"]
    aoi = load_aoi(cfg)
    vv, vh, transform, crs = load_chip(repo_path(cfg["fetch"]["chip_path"]))
    aoi_utm = gpd.GeoSeries([aoi], crs=4326).to_crs(crs).iloc[0]

    ref = load_gfw_detections(repo_path(g["detections_csv"]), aoi.bounds,
                              p["scene_start"], p["scene_end"], g["time_tolerance_s"]).to_crs(crs)
    ref = ref[ref.within(aoi_utm)].reset_index(drop=True)
    print(f"{len(ref)} GFW detections inside the AOI for this scene")

    footprint = None
    fp_file = repo_path(g["footprints_csv"])
    if fp_file.exists():
        footprint = union_all(list(load_footprints(fp_file).to_crs(crs)))
    else:
        print("WARNING: no footprint file. Detections outside the GFW processed area "
              "cannot be separated from false alarms.")

    tables, figures = repo_dir(cfg["outputs"]["tables_dir"]), repo_dir(cfg["outputs"]["figures_dir"])
    rows = []
    for pol, lin in {"VV": vv, "VH": vh}.items():
        det_file = repo_path(f"{cfg['outputs']['detections_dir']}/pilot_detections_{pol}.geojson")
        if lin is None or not det_file.exists():
            continue
        ours = gpd.read_file(det_file).to_crs(crs)
        ours = ours[ours.within(aoi_utm)]
        if footprint is not None:
            ours = ours[ours.within(footprint)]
        ours = ours.reset_index(drop=True)

        for d in v["match_distances_m"]:
            res = match_detections(ours, ref, d)
            tp = len(res["pairs"])
            row = {"polarisation": pol, "max_distance_m": d, "n_ours": len(ours), "n_gfw": len(ref),
                   **prf(tp, len(ours) - tp, len(ref) - tp)}
            if ref["matched"].notna().any():
                found = set(res["pairs"]["ref_idx"])
                for flag, name in [(True, "recall_ais_matched"), (False, "recall_ais_unmatched")]:
                    idx = ref.index[ref["matched"].eq(flag).fillna(False).to_numpy(dtype=bool)]
                    row[name] = len(found & set(idx)) / len(idx) if len(idx) else float("nan")
            rows.append(row)
            if d == v["primary_distance_m"]:
                res["pairs"].to_csv(tables / f"pilot_matches_{pol}.csv", index=False)
                plot_overview(to_db(lin), transform, ours, ref, res,
                              figures / f"pilot_overview_{pol}.png", f"{pol}, match radius {d} m")

    metrics = pd.DataFrame(rows)
    metrics.to_csv(tables / "pilot_metrics.csv", index=False)
    print(metrics.round(3).to_string(index=False))


STEPS = {"gfw-summary": cmd_gfw_summary, "search": cmd_search, "fetch": cmd_fetch,
         "detect": cmd_detect, "validate": cmd_validate}


def main():
    parser = argparse.ArgumentParser(description="Dark-vessel pilot pipeline")
    parser.add_argument("step", choices=STEPS)
    args = parser.parse_args()
    STEPS[args.step](load_config())


if __name__ == "__main__":
    main()
