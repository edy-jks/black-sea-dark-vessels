"""Load study configuration and resolve paths relative to the repository root."""
import json
from pathlib import Path

import yaml
from shapely.geometry import shape

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path="config/study.yaml"):
    with open(REPO_ROOT / path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def repo_path(relative):
    """Absolute path inside the repo; creates the parent folder."""
    p = REPO_ROOT / relative
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def repo_dir(relative):
    """Absolute path of a folder inside the repo; creates it."""
    p = REPO_ROOT / relative
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_aoi(cfg):
    """Return the study area as a shapely geometry in EPSG:4326 (.bounds = west, south, east, north)."""
    with open(REPO_ROOT / cfg["aoi_file"], "r", encoding="utf-8") as f:
        gj = json.load(f)
    return shape(gj["features"][0]["geometry"])