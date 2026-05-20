"""NASA CMR (Common Metadata Repository) granule-search helper.

Used by the cloud-OPeNDAP-per-granule collectors to discover which granules
intersect an AOI + temporal range and to obtain their cloud OPeNDAP URLs.

CMR docs: https://cmr.earthdata.nasa.gov/search/site/docs/search/api.html
"""

import datetime as _dt
import re

import requests

CMR_GRANULES_URL = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"

# Matches the AYYYYDDD acquisition-date token in MODIS granule URs, e.g.
# "MOD11A1.A2019060.h20v06.061.2020346005755" -> "2019060".
_DOY_RE = re.compile(r"\.A(\d{7})\.")
# Matches MODIS tile hHHvVV, e.g. "h20v06".
_TILE_RE = re.compile(r"\.h(\d{2})v(\d{2})\.")


def _modis_date_from_ur(granule_ur):
    m = _DOY_RE.search(granule_ur)
    if not m:
        return None
    return _dt.datetime.strptime(m.group(1), "%Y%j").date()


def _modis_tile_from_ur(granule_ur):
    m = _TILE_RE.search(granule_ur)
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)))


def _pick_opendap_url(related_urls):
    """Return the cloud OPeNDAP service URL for a granule, if any."""
    for r in related_urls or []:
        url = r.get("URL", "")
        if url.startswith("https://opendap.earthdata.nasa.gov/"):
            return url
    return None


def search_granules(
    short_name,
    version,
    latlim,
    lonlim,
    timelim,
    page_size=2000,
    session=None,
):
    """Search CMR for granules and return a list of dicts.

    Each returned item has:
      - `granule_ur`: CMR GranuleUR string
      - `url`: cloud OPeNDAP base URL (or None if the granule has none)
      - `tile`: (h, v) tuple for MODIS-tile-style URs, else None
      - `date`: datetime.date parsed from the granule UR, else None
      - `temporal`: (start, end) ISO strings from CMR if present

    Parameters
    ----------
    short_name : str
        e.g. "MOD11A1".
    version : str
        e.g. "061".
    latlim, lonlim : list of float
        [min, max] in EPSG:4326.
    timelim : list of (date | datetime | str)
        [start, end]. `str` should be ISO 8601.
    """
    def _iso(t):
        if isinstance(t, str):
            return t if t.endswith("Z") else f"{t}T00:00:00Z"
        if isinstance(t, _dt.datetime):
            return t.strftime("%Y-%m-%dT%H:%M:%SZ")
        if isinstance(t, _dt.date):
            return t.strftime("%Y-%m-%dT00:00:00Z")
        # numpy datetime, pandas Timestamp
        return _dt.datetime.fromisoformat(str(t).replace("Z", "")).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

    params = {
        "short_name": short_name,
        "version": version,
        "temporal": f"{_iso(timelim[0])},{_iso(timelim[1])}",
        "bounding_box": f"{lonlim[0]},{latlim[0]},{lonlim[1]},{latlim[1]}",
        "page_size": page_size,
    }

    sess = session or requests.Session()
    items_out = []
    search_after = None
    while True:
        headers = {}
        if search_after:
            headers["CMR-Search-After"] = search_after
        r = sess.get(CMR_GRANULES_URL, params=params, headers=headers, timeout=60)
        r.raise_for_status()
        data = r.json()
        for it in data.get("items", []):
            umm = it.get("umm", {})
            ur = umm.get("GranuleUR", "")
            url = _pick_opendap_url(umm.get("RelatedUrls", []))
            temporal = umm.get("TemporalExtent", {}).get("RangeDateTime", {})
            items_out.append({
                "granule_ur": ur,
                "url": url,
                "tile": _modis_tile_from_ur(ur),
                "date": _modis_date_from_ur(ur),
                "temporal": (
                    temporal.get("BeginningDateTime"),
                    temporal.get("EndingDateTime"),
                ),
            })
        search_after = r.headers.get("CMR-Search-After")
        if not search_after or not data.get("items"):
            break
    return items_out


if __name__ == "__main__":
    res = search_granules(
        "MOD11A1", "061",
        latlim=[29.4, 29.5], lonlim=[30.7, 30.8],
        timelim=["2019-03-01", "2019-03-03"],
    )
    print(f"found {len(res)} granules")
    for r in res[:5]:
        print(r)
