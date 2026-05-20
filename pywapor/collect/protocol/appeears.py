"""AppEEARS REST API protocol layer.

AppEEARS (https://appeears.earthdatacloud.nasa.gov) is NASA's recommended
replacement for the LP DAAC on-prem OPeNDAP Hyrax service that was retired in
September 2025. It performs server-side spatial/temporal subsetting and
reprojection, returning a small bundle of files (NetCDF or GeoTIFF) per task.

This module wraps the REST API into a small set of building blocks. It does
not know anything about specific products; that is the job of the per-product
collectors that use this module.
"""

import os
import time

import requests

from pywapor.general.logger import log

BASE_URL = "https://appeears.earthdatacloud.nasa.gov/api"

DEFAULT_TIMEOUT = 60
DOWNLOAD_TIMEOUT = 600

TERMINAL_STATUSES = {"done", "error", "expired", "deleted"}


def login(username, password):
    """Exchange Earthdata Login credentials for an AppEEARS bearer token.

    The token is valid for ~48h.
    """
    r = requests.post(
        f"{BASE_URL}/login",
        auth=(username, password),
        headers={"Content-Length": "0"},
        timeout=DEFAULT_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()["token"]


def logout(token):
    requests.post(
        f"{BASE_URL}/logout",
        headers={"Authorization": f"Bearer {token}"},
        timeout=DEFAULT_TIMEOUT,
    )


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def build_area_payload(
    task_name,
    product_layers,
    latlim,
    lonlim,
    timelim,
    projection="geographic",
    out_format="netcdf4",
):
    """Build the JSON body for an AppEEARS area task.

    Parameters
    ----------
    task_name : str
        Human-readable task name.
    product_layers : list of (product, layer)
        e.g. [("MOD11A1.061", "LST_Day_1km"), ("MOD11A1.061", "QC_Day")].
    latlim, lonlim : list of float
        [min, max] in EPSG:4326.
    timelim : list of date-like
        [start, end]. Must support strftime.
    projection : str
        AppEEARS projection id. "geographic" yields EPSG:4326.
    out_format : {"netcdf4", "geotiff"}
    """
    start = timelim[0].strftime("%m-%d-%Y")
    end = timelim[1].strftime("%m-%d-%Y")
    polygon = [[
        [lonlim[0], latlim[0]],
        [lonlim[0], latlim[1]],
        [lonlim[1], latlim[1]],
        [lonlim[1], latlim[0]],
        [lonlim[0], latlim[0]],
    ]]
    return {
        "task_type": "area",
        "task_name": task_name,
        "params": {
            "dates": [{"startDate": start, "endDate": end, "recurring": False}],
            "layers": [
                {"product": product, "layer": layer}
                for product, layer in product_layers
            ],
            "geo": {
                "type": "FeatureCollection",
                "fileName": "pywapor-aoi",
                "features": [{
                    "type": "Feature",
                    "properties": {},
                    "geometry": {"type": "Polygon", "coordinates": polygon},
                }],
            },
            "output": {
                "format": {"type": out_format},
                "projection": projection,
            },
        },
    }


def submit_task(token, payload):
    r = requests.post(
        f"{BASE_URL}/task",
        json=payload,
        headers={**_auth(token), "Content-Type": "application/json"},
        timeout=DEFAULT_TIMEOUT,
    )
    if not r.ok:
        raise requests.HTTPError(
            f"{r.status_code} {r.reason} for {r.url}: {r.text}", response=r
        )
    return r.json()["task_id"]


def task_status(token, task_id):
    r = requests.get(
        f"{BASE_URL}/task/{task_id}",
        headers=_auth(token),
        timeout=DEFAULT_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def wait_for_completion(token, task_id, poll_interval=30, timeout=7200):
    """Block until the task reaches a terminal state.

    Raises RuntimeError on non-`done` terminal states, TimeoutError if the
    poll budget is exhausted.
    """
    deadline = time.time() + timeout
    last_status = None
    while time.time() < deadline:
        info = task_status(token, task_id)
        status = info.get("status")
        if status != last_status:
            log.info(f"--> AppEEARS task {task_id}: {status}")
            last_status = status
        if status == "done":
            return info
        if status in TERMINAL_STATUSES:
            raise RuntimeError(
                f"AppEEARS task {task_id} ended with status={status!r}: {info}"
            )
        time.sleep(poll_interval)
    raise TimeoutError(
        f"AppEEARS task {task_id} did not finish within {timeout}s "
        f"(last status: {last_status})"
    )


def list_bundle(token, task_id):
    """Return the list of files produced by a completed task."""
    r = requests.get(
        f"{BASE_URL}/bundle/{task_id}",
        headers=_auth(token),
        timeout=DEFAULT_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()["files"]


def list_tasks(token, limit=1000):
    """Return all of this user's AppEEARS tasks (summary records).

    Each item carries `task_id`, `task_name`, `task_type`, `status`,
    timestamps, and some summary fields. Use `task_status(token, task_id)`
    to fetch full params for any individual task.
    """
    r = requests.get(
        f"{BASE_URL}/task",
        params={"limit": limit},
        headers=_auth(token),
        timeout=DEFAULT_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


# Status values AppEEARS uses for in-flight or completed tasks. `done` means
# the bundle is ready; the others mean we can wait. `error`/`expired`/`deleted`
# are intentionally excluded — a match with those statuses is not reusable.
REUSABLE_STATUSES = ("done", "pending", "queued", "processing")


def _normalise_layers(layers):
    return sorted((l["product"], l["layer"]) for l in layers)


def _normalise_bbox(geo):
    coords = geo["features"][0]["geometry"]["coordinates"][0]
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return (min(xs), min(ys), max(xs), max(ys))


def _normalise_dates(dates):
    return sorted(
        (d.get("startDate"), d.get("endDate"), bool(d.get("recurring", False)))
        for d in dates
    )


def payloads_match(existing_params, intended_payload, bbox_tol=1e-5):
    """Decide whether an existing task's params satisfy a new request.

    Compares task_type, dates, layers (as a set), bbox (within tolerance),
    output format and output projection. Ignores task_name and any
    AppEEARS-side bookkeeping fields.
    """
    if existing_params.get("task_type") != intended_payload.get("task_type"):
        return False
    ep = existing_params.get("params", {})
    ip = intended_payload.get("params", {})
    if _normalise_layers(ep.get("layers", [])) != _normalise_layers(ip.get("layers", [])):
        return False
    if _normalise_dates(ep.get("dates", [])) != _normalise_dates(ip.get("dates", [])):
        return False
    a = _normalise_bbox(ep.get("geo", {"features": [{"geometry": {"coordinates": [[]]}}]}))
    b = _normalise_bbox(ip.get("geo", {"features": [{"geometry": {"coordinates": [[]]}}]}))
    if any(abs(x - y) > bbox_tol for x, y in zip(a, b)):
        return False
    eo = ep.get("output", {})
    io = ip.get("output", {})
    if eo.get("format", {}).get("type") != io.get("format", {}).get("type"):
        return False
    if eo.get("projection") != io.get("projection"):
        return False
    return True


def find_matching_task(token, payload, allowed_statuses=REUSABLE_STATUSES):
    """Return a task summary if one of this user's existing tasks satisfies
    `payload`, else None.

    Newest match wins (AppEEARS returns tasks newest-first). Tasks in
    `error`/`expired`/`deleted` are skipped by default.
    """
    for t in list_tasks(token):
        if t.get("status") not in allowed_statuses:
            continue
        if t.get("task_type") != payload.get("task_type"):
            continue
        try:
            details = task_status(token, t["task_id"])
        except requests.HTTPError:
            continue
        if payloads_match(details, payload):
            return details
    return None


def download_file(token, task_id, file_id, dest_path):
    """Stream one bundle file to `dest_path`. Follows the S3 redirect."""
    with requests.get(
        f"{BASE_URL}/bundle/{task_id}/{file_id}",
        headers=_auth(token),
        stream=True,
        allow_redirects=True,
        timeout=DOWNLOAD_TIMEOUT,
    ) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return dest_path


def run_area_task(
    username,
    password,
    task_name,
    product_layers,
    latlim,
    lonlim,
    timelim,
    out_dir,
    projection="geographic",
    out_format="netcdf4",
    file_filter=None,
    poll_interval=30,
    timeout=7200,
    reuse_existing=True,
):
    """Submit an area task, wait for it, download the bundle to `out_dir`.

    Returns a list of locally-downloaded file paths.

    `file_filter`, if provided, is called as `file_filter(file_metadata_dict)`
    and only files for which it returns truthy are downloaded.

    If `reuse_existing` is True (default), the user's existing AppEEARS tasks
    are scanned first; if one matches the intended payload (same layers,
    dates, bbox, format, projection) and is `done` or still in flight, that
    task is reused instead of submitting a new one. A typical area task takes
    ~10 min to process server-side, so this skip can be substantial.
    """
    os.makedirs(out_dir, exist_ok=True)
    token = login(username, password)
    try:
        payload = build_area_payload(
            task_name=task_name,
            product_layers=product_layers,
            latlim=latlim,
            lonlim=lonlim,
            timelim=timelim,
            projection=projection,
            out_format=out_format,
        )
        task_id = None
        if reuse_existing:
            existing = find_matching_task(token, payload)
            if existing is not None:
                task_id = existing["task_id"]
                log.info(
                    f"--> Reusing existing AppEEARS task {task_id} "
                    f"(status={existing.get('status')!r}, "
                    f"name={existing.get('task_name')!r})."
                )
        if task_id is None:
            task_id = submit_task(token, payload)
            log.info(f"--> AppEEARS task submitted (task_id={task_id}).")
        wait_for_completion(
            token, task_id, poll_interval=poll_interval, timeout=timeout
        )
        files = list_bundle(token, task_id)
        downloaded = []
        for meta in files:
            if file_filter is not None and not file_filter(meta):
                continue
            name = meta["file_name"]
            file_id = meta["file_id"]
            dest = os.path.join(out_dir, os.path.basename(name))
            log.info(f"--> Downloading `{os.path.basename(name)}`.")
            download_file(token, task_id, file_id, dest)
            downloaded.append(dest)
        return downloaded
    finally:
        try:
            logout(token)
        except Exception:
            pass


def list_products():
    """Return AppEEARS's full product catalog (no auth required)."""
    r = requests.get(f"{BASE_URL}/product", timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    return r.json()


def find_products(query):
    """Return catalog entries whose ProductAndVersion contains `query` (case-insensitive)."""
    q = query.lower()
    return [p for p in list_products() if q in p.get("ProductAndVersion", "").lower()]


def list_layers(product_and_version):
    """Return the layer dict for a given `ProductAndVersion`, e.g. "SRTMGL1.003"."""
    r = requests.get(
        f"{BASE_URL}/product/{product_and_version}", timeout=DEFAULT_TIMEOUT
    )
    r.raise_for_status()
    return r.json()


if __name__ == "__main__":
    import json as _json
    import sys

    if len(sys.argv) > 1:
        query = sys.argv[1]
        matches = find_products(query)
        print(f"Products matching {query!r}:")
        for p in matches:
            print(f"  {p['ProductAndVersion']}  —  {p.get('Description', '')}")
        if len(matches) == 1:
            print()
            print(f"Layers for {matches[0]['ProductAndVersion']}:")
            print(_json.dumps(list_layers(matches[0]["ProductAndVersion"]), indent=2))
