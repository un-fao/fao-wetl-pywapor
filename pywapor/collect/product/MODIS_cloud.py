"""MODIS collection via per-granule cloud OPeNDAP — prototype (BROKEN UPSTREAM).

Status (verified 2026-05-14)
----------------------------
**Not production-viable.** The implementation is functionally correct — CMR
discovery, URS auth, HDF-EOS group reading, mosaicking and reprojection all
work — but ~60% of MOD11A1.061 granules are unfetchable from NASA's cloud
OPeNDAP service due to an unresolved server-side bug. Preserved as a working
code reference for if/when NASA fixes it.

The bug
-------
`opendap.earthdata.nasa.gov` serves data by reading a precomputed `.dmrpp`
sidecar from the LP DAAC S3 bucket. For granules whose CMR `RelatedUrls` list
a BROWSE `.jpg` before the `.hdf`, the service resolves the wrong S3 object
and returns 404 with a body like:

    The HTTP GET request for the source URL:
    https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-public/MOD11A1.061/
        MOD11A1.A2019061.h20v06.061.2020346011503/
        BROWSE.MOD11A1.A2019061.h20v06.061.2020346011506.2.jpg.dmrpp FAILED
    ... NgapOwnedContainer::dmrpp_read_from_daac_bucket() failed to read the
    DMR++ from S3.

i.e. it tried to load the DMR++ sidecar for a browse-image URL that has no
such sidecar. Empirically reproducible (10 consecutive March 2019 granules):

    2019-03-01  OK            2019-03-06  404
    2019-03-02  404           2019-03-07  OK
    2019-03-03  404           2019-03-08  404
    2019-03-04  404           2019-03-09  404
    2019-03-05  OK            2019-03-10  OK

The 404 set correlates with granules whose CMR record was later updated to
include browse imagery; the originally-archived granules without browse work.

Production path
---------------
Use `MODIS` (server-side subsetting via NASA's AppEEARS) instead.
A direct-HDF download from `data.lpdaac.earthdatacloud.nasa.gov` is also a
reliable fallback that routes around cloud OPeNDAP entirely.

Original design (kept for reference)
------------------------------------
Replaces the retired on-prem LP DAAC Hyrax aggregation (`opendap.cr.usgs.gov`)
with NASA's cloud OPeNDAP at `opendap.earthdata.nasa.gov`. Each MODIS granule
is one date+tile, exposed at its own URL discovered via CMR.

Flow:
  1. CMR granule search for the AOI + temporal range.
  2. URS Earthdata Login session (reuses existing `opendap.setup_session`).
  3. Per granule: download `<granule>.dap.nc4` (full granule, no constraint
     expression — MODIS HDF-EOS data lives in nested groups that DAP2 cannot
     subset cleanly; the per-tile granule is small (~6 MB) so we do the
     spatial subset locally after `rioxarray.merge` and `rio.reproject`).
  4. Open the HDF-EOS grid group + data fields group, attach XDim/YDim coords.
  5. Group by date, spatially mosaic the tiles per date, concatenate over time.
  6. Reproject MODIS sinusoidal -> EPSG:4326, clip to AOI, post-process, save.

`download()` mirrors `pywapor.collect.product.MODIS.download` so this module
is a drop-in replacement for testing.
"""

import copy
import itertools
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import rioxarray.merge
import xarray as xr
from rasterio.crs import CRS

import pywapor.collect.accounts as accounts
import pywapor.collect.protocol.cmr as cmr
import pywapor.collect.protocol.opendap as opendap
from pywapor.collect.product.MODIS import (
    default_post_processors,
    default_vars,
)
from pywapor.collect.protocol.crawler import download_url
from pywapor.collect.protocol.projections import get_crss
from pywapor.enhancers.apply_enhancers import apply_enhancers
from pywapor.general.logger import log
from pywapor.general.processing_functions import (
    adjust_timelim_dtype,
    open_ds,
    save_ds,
)

# HDF-EOS grid-group name per product, as exposed by the cloud OPeNDAP DMR.
GRID_GROUP = {
    "MOD11A1.061": "MODIS_Grid_Daily_1km_LST",
    "MYD11A1.061": "MODIS_Grid_Daily_1km_LST",
    "MOD13Q1.061": "MODIS_Grid_16DAY_250m_500m_VI",
    "MYD13Q1.061": "MODIS_Grid_16DAY_250m_500m_VI",
    "MCD43A3.061": "MOD_Grid_BRDF",
}

SPATIAL_BUFFER = {
    "MOD11A1.061": True,
    "MYD11A1.061": True,
    "MCD43A3.061": True,
    "MOD13Q1.061": False,
    "MYD13Q1.061": False,
}


def _short_and_version(product_name):
    short, version = product_name.split(".")
    return short, version


def _load_granule(fp, grid_group, layer_vars):
    """Open `fp` (a .dap.nc4 granule) and return an xr.Dataset with the
    requested data layers and `XDim`/`YDim` 1-D coordinates attached.
    """
    parent = xr.open_dataset(fp, group=grid_group)
    data = xr.open_dataset(fp, group=f"{grid_group}/Data_Fields")

    keep = [v for v in layer_vars if v in data.data_vars]
    if not keep:
        raise ValueError(
            f"None of requested layers {layer_vars} found in {fp} "
            f"(have: {list(data.data_vars)})."
        )
    data = data[keep]

    # Cloud OPeNDAP exposes per-group dims as `<axis>_<group>`; rename to plain
    # XDim / YDim so the rest of the pipeline can treat the dataset uniformly.
    rename = {}
    for d in data.dims:
        if d.startswith("XDim"):
            rename[d] = "XDim"
        elif d.startswith("YDim"):
            rename[d] = "YDim"
    if rename:
        data = data.rename(rename)

    data = data.assign_coords({"XDim": parent["XDim"], "YDim": parent["YDim"]})
    return data


def _standardise(ds, variables, data_source_crs):
    """Rename layer vars + spatial dims; attach CRS; sort axes."""
    keep = {k: v for k, v in variables.items() if k in ds.data_vars}
    if not keep:
        raise ValueError(
            f"granule is missing all requested layers {list(variables)} "
            f"(have: {list(ds.data_vars)})."
        )
    ds = ds[list(keep.keys())].rename({k: v[1] for k, v in keep.items()})
    ds = ds.rename({"XDim": "x", "YDim": "y"})
    ds = ds.rio.write_crs(data_source_crs).rio.write_grid_mapping("spatial_ref")
    ds = ds.sortby("y", ascending=False).sortby("x")
    ds.attrs = {}
    return ds


def most_recent(product_name, latlim, lonlim):
    short, version = _short_and_version(product_name)
    res = cmr.search_granules(
        short, version,
        latlim=latlim, lonlim=lonlim,
        timelim=[
            (pd.Timestamp.utcnow() - pd.Timedelta(days=60)).to_pydatetime(),
            pd.Timestamp.utcnow().to_pydatetime(),
        ],
    )
    dates = sorted({r["date"] for r in res if r["date"] is not None})
    if not dates:
        return None
    return pd.Timestamp(dates[-1]).to_pydatetime()


def download(
    folder,
    latlim,
    lonlim,
    timelim,
    product_name,
    req_vars,
    variables=None,
    post_processors=None,
):
    """Cloud-OPeNDAP-per-granule replacement for `MODIS.download`."""
    if product_name not in GRID_GROUP:
        raise NotImplementedError(
            f"MODIS_cloud prototype supports {sorted(GRID_GROUP)}; "
            f"got {product_name!r}."
        )

    folder = os.path.join(folder, "MODIS")
    os.makedirs(folder, exist_ok=True)
    fn = os.path.join(folder, f"{product_name}.nc")

    req_vars_orig = copy.deepcopy(req_vars)
    if os.path.isfile(fn):
        existing_ds = open_ds(fn)
        req_vars_new = list(set(req_vars).difference(set(existing_ds.data_vars)))
        if not req_vars_new:
            return existing_ds[req_vars_orig]
        existing_ds.close()
        req_vars = req_vars_new

    if SPATIAL_BUFFER.get(product_name, False):
        latlim = [latlim[0] - 0.01, latlim[1] + 0.01]
        lonlim = [lonlim[0] - 0.01, lonlim[1] + 0.01]

    timelim = adjust_timelim_dtype(timelim)
    if product_name in ("MOD13Q1.061", "MYD13Q1.061"):
        timedelta = np.timedelta64(8, "D")
        timelim = [timelim[0] - pd.Timedelta(timedelta), timelim[1]]
    elif product_name == "MCD43A3.061":
        timedelta = np.timedelta64(12, "h")
        timelim = [timelim[0] - pd.Timedelta(timedelta), timelim[1]]
    else:
        timedelta = None

    if variables is None:
        variables = default_vars(product_name, req_vars)
    if post_processors is None:
        post_processors = default_post_processors(product_name, req_vars)
    else:
        defaults = default_post_processors(product_name, req_vars)
        post_processors = {
            k: (defaults[k] if v == "default" else v)
            for k, v in post_processors.items()
            if k in req_vars
        }

    # The MODIS.default_vars table includes coordinate / projection helper
    # entries (e.g. `time`, `MODIS_Grid_..._eos_cf_projection`). Strip them
    # to the actual HDF-EOS data-field layers we need to load.
    layer_vars = [
        k for k, v in variables.items()
        if isinstance(v, list) and len(v) >= 1
        and isinstance(v[0], tuple)
        and any(d in v[0] for d in ("XDim", "YDim"))
    ]
    # Only keep the data-field entries in `variables` for renaming downstream.
    variables = {k: v for k, v in variables.items() if k in layer_vars}

    data_source_crs = get_crss("MODIS")
    grid_group = GRID_GROUP[product_name]

    short, version = _short_and_version(product_name)
    log.info(f"--> Searching CMR for {short}.{version} granules.")
    granules = cmr.search_granules(
        short, version,
        latlim=latlim, lonlim=lonlim, timelim=timelim,
    )
    granules = [g for g in granules if g["url"] and g["date"] is not None]
    if not granules:
        raise RuntimeError(
            f"CMR returned 0 granules with cloud OPeNDAP URLs for "
            f"{product_name} over {latlim}/{lonlim}/{timelim}."
        )
    log.info(f"--> CMR returned {len(granules)} granule(s).")

    un_pw = accounts.get("NASA")
    sample_url = granules[0]["url"] + ".dap.nc4"
    session = opendap.setup_session(
        "https://urs.earthdata.nasa.gov",
        username=un_pw[0], password=un_pw[1],
        check_url=sample_url,
    )

    staging = tempfile.mkdtemp(prefix="modis_cloud_", dir=folder)
    try:
        per_granule = []
        for g in granules:
            url = g["url"] + ".dap.nc4"
            local_fp = os.path.join(staging, f"{g['granule_ur']}.nc")
            log.info(f"--> Downloading granule `{g['granule_ur']}`.")
            download_url(url, local_fp, session, waitbar=False)
            raw = _load_granule(local_fp, grid_group, layer_vars)
            std = _standardise(raw, variables, data_source_crs)
            std = std.expand_dims(time=[pd.Timestamp(g["date"]).to_datetime64()])
            per_granule.append((g["date"], std))

        per_granule.sort(key=lambda x: x[0])
        daily = []
        for date, group in itertools.groupby(per_granule, key=lambda x: x[0]):
            dss = [item[1] for item in group]
            if len(dss) == 1:
                day = dss[0]
            else:
                day = rioxarray.merge.merge_datasets(dss)
            daily.append(day)
        ds = xr.concat(daily, dim="time").sortby("time")

        if ds.rio.crs.to_epsg() != 4326:
            ds = ds.rio.reproject(CRS.from_epsg(4326))

        ds = ds.rio.clip_box(lonlim[0], latlim[0], lonlim[1], latlim[1])

        ds = apply_enhancers(post_processors, ds)
        if isinstance(timedelta, np.timedelta64):
            ds["time"] = ds["time"] + timedelta

        ds = ds[list(post_processors.keys())]
        ds.attrs = {}
        for var in ds.data_vars:
            ds[var].attrs = {}

        ds = save_ds(ds, fn, encoding="initiate",
                     label="Saving cloud-OPeNDAP merged data.")
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    return ds[req_vars_orig]


if __name__ == "__main__":
    from pywapor.general.logger import adjust_logger

    out_folder = "/tmp/pywapor_cloud_test"
    os.makedirs(out_folder, exist_ok=True)
    adjust_logger(True, out_folder, "INFO")

    ds = download(
        folder=out_folder,
        latlim=[29.4, 29.5],
        lonlim=[30.7, 30.8],
        timelim=["2019-03-01", "2019-03-03"],
        product_name="MOD11A1.061",
        req_vars=["lst"],
    )
    print(ds)
    print("CRS:", ds.rio.crs)
    print("Vars:", list(ds.data_vars))
