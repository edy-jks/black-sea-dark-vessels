"""Search the CDSE STAC catalogue for Sentinel-1 GRD scenes (metadata only, nothing is downloaded)."""
from pystac_client import Client
from shapely.geometry import shape


def search_s1_scenes(stac_url, collection, aoi, start, end):
    client = Client.open(stac_url)
    search = client.search(
        collections=[collection],
        bbox=list(aoi.bounds),
        datetime=f"{start}/{end}",
        max_items=500,
    )
    rows = []
    for item in search.items():
        p = item.properties
        rows.append({
            "scene_id": item.id,
            "platform": p.get("platform"),
            "start": p.get("start_datetime") or p.get("datetime"),
            "end": p.get("end_datetime") or p.get("datetime"),
            "mode": p.get("sar:instrument_mode"),
            "polarizations": ",".join(p.get("sar:polarizations", []) or []),
            "orbit_state": p.get("sat:orbit_state"),
            "covers_aoi": bool(shape(item.geometry).contains(aoi)),
        })
    return rows