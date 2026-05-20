"""SRTM (30 m) collection via NASA AppEEARS.

Replaces the previous OPeNDAP-backed implementation; the LP DAAC Hyrax
service that hosted `SRTMGL1_NC.003` was retired on 2025-09-19. AppEEARS is
NASA's recommended successor for MODIS/SRTM subsetting.

The single AppEEARS layer `SRTMGL1.003.SRTM_DEM` provides elevation; `slope`
and `aspect` are derived from `z` via `pywapor.enhancers.dem.calc_slope_or_aspect`.
"""

import copy
import datetime
import os
import shutil
import tempfile
import time

import numpy as np
import xarray as xr

import pywapor.collect.accounts as accounts
import pywapor.collect.protocol.appeears as appeears
from pywapor.enhancers.apply_enhancers import apply_enhancers
from pywapor.enhancers.dem import calc_slope_or_aspect
from pywapor.general.logger import log
from pywapor.general.processing_functions import open_ds, save_ds

# Pywapor product_name -> AppEEARS product id.
APPEEARS_PRODUCT = {
    "30M": "SRTMGL1_NC.003",
}

# {pywapor_product: {req_var: {appeears_layer: pywapor_var}}}.
# All three req_vars need the same DEM layer; slope/aspect are derived from z
# by the post-processor.
LAYERS = {
    "30M": {
        "z": {"SRTMGL1_DEM": "z"},
        "slope": {"SRTMGL1_DEM": "z"},
        "aspect": {"SRTMGL1_DEM": "z"},
    },
}

# SRTM was acquired in February 2000; AppEEARS still requires a date range.
SRTM_TIMELIM = [datetime.date(2000, 2, 10), datetime.date(2000, 2, 12)]


def default_vars(product_name, req_vars=["z"]):
    """Resolve req_vars to {appeears_layer: [(), pywapor_var]} for `product_name`.

    Used by `pywapor.main.Configuration.has_var` to validate variable support;
    invalid req_vars must raise `TypeError` (see main.py:656).
    """
    supported = LAYERS[product_name]
    out = {}
    for v in req_vars:
        if v not in supported:
            raise TypeError(f"SRTM.{product_name} has no req_var {v!r}")
        for layer, name in supported[v].items():
            out[layer] = [(), name]
    return out


def default_post_processors(product_name, req_vars=["z"]):
    """Return `{req_var: [callables]}` for the requested variables."""
    post_processors = {
        "30M": {
            "z": [],
            "aspect": [calc_slope_or_aspect],
            "slope": [calc_slope_or_aspect],
        }
    }
    return {k: v for k, v in post_processors[product_name].items() if k in req_vars}


def _resolve_layer_map(product_name, req_vars):
    """{appeears_layer: pywapor_var} for all req_vars of one product."""
    out = {}
    for v in req_vars:
        out.update(LAYERS[product_name][v])
    return out


def _find_dim(ds, candidates):
    for c in candidates:
        if c in ds.dims or c in ds.coords:
            return c
    raise KeyError(f"None of {candidates} present in dataset (dims={list(ds.dims)}).")


def _standardise(ds, layer_map):
    """Coerce an AppEEARS-produced NetCDF into the schema pywapor expects."""
    for v in ("crs", "spatial_ref"):
        if v in ds.variables and v not in ds.dims:
            ds = ds.drop_vars(v, errors="ignore")

    keep = {k: v for k, v in layer_map.items() if k in ds.variables}
    if not keep:
        raise ValueError(
            f"AppEEARS output is missing all requested layers {list(layer_map)} "
            f"(found vars: {list(ds.data_vars)})."
        )
    ds = ds[list(keep.keys())].rename(keep)

    y_name = _find_dim(ds, ["lat", "latitude", "YDim", "y"])
    x_name = _find_dim(ds, ["lon", "longitude", "XDim", "x"])
    rename = {}
    if y_name != "y":
        rename[y_name] = "y"
    if x_name != "x":
        rename[x_name] = "x"
    if rename:
        ds = ds.rename(rename)

    if ds["y"].values[0] < ds["y"].values[-1]:
        ds = ds.isel(y=slice(None, None, -1))
    if ds["x"].values[0] > ds["x"].values[-1]:
        ds = ds.isel(x=slice(None, None, -1))

    if "time" in ds.coords and not np.issubdtype(ds["time"].dtype, np.datetime64):
        ds = ds.assign_coords(
            time=np.array([np.datetime64(t.isoformat()) for t in ds["time"].values])
        )

    for var in list(ds.data_vars):
        for key in ("grid_mapping", "coordinates"):
            ds[var].attrs.pop(key, None)
            ds[var].encoding.pop(key, None)

    ds = ds.rio.write_crs(4326).rio.write_grid_mapping("spatial_ref")
    return ds


def _drop_time(ds):
    """Collapse the singleton time dim left over from the AppEEARS bundle."""
    if "time" in ds.dims:
        ds = ds.isel(time=0)
    for coord in ("time",):
        if coord in ds.coords:
            ds = ds.drop_vars(coord, errors="ignore")
    return ds


def _open_bundle(nc_files):
    if len(nc_files) == 1:
        return xr.open_dataset(nc_files[0], decode_coords="all")
    dss = [xr.open_dataset(f, decode_coords="all") for f in nc_files]
    return xr.merge(dss, compat="override", join="outer")


def most_recent(product_name, *args):
    return None


def download(
    folder,
    latlim,
    lonlim,
    product_name="30M",
    req_vars=["z"],
    variables=None,
    post_processors=None,
    reuse_existing=True,
    **kwargs,
):
    """AppEEARS-backed replacement for the OPeNDAP SRTM download.

    `timelim` is accepted via **kwargs and ignored — SRTM is a single-epoch
    DEM, so the request is always pinned to the acquisition window.
    """
    if variables is not None:
        log.warning(
            "SRTM: `variables` kwarg is ignored — layer selection is driven by "
            "`req_vars` against the LAYERS table."
        )

    if product_name not in LAYERS:
        raise NotImplementedError(
            f"SRTM only supports {sorted(LAYERS)}; got {product_name!r}."
        )

    folder = os.path.join(folder, "SRTM")
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

    dx = dy = 0.0002777777777777768
    latlim = [latlim[0] - dy, latlim[1] + dy]
    lonlim = [lonlim[0] - dx, lonlim[1] + dx]

    layer_map = _resolve_layer_map(product_name, req_vars)
    appeears_product = APPEEARS_PRODUCT[product_name]
    product_layers = [(appeears_product, layer) for layer in layer_map.keys()]

    if post_processors is None:
        post_processors = default_post_processors(product_name, req_vars)
    else:
        defaults = default_post_processors(product_name, req_vars)
        post_processors = {
            k: (defaults[k] if v == "default" else v)
            for k, v in post_processors.items()
            if k in req_vars
        }

    un_pw = accounts.get("NASA")

    staging = tempfile.mkdtemp(prefix="appeears_", dir=folder)
    raw = None
    try:
        files = appeears.run_area_task(
            username=un_pw[0],
            password=un_pw[1],
            task_name=f"pywapor_SRTM_{product_name}_{int(time.time())}",
            product_layers=product_layers,
            latlim=latlim,
            lonlim=lonlim,
            timelim=SRTM_TIMELIM,
            out_dir=staging,
            projection="geographic",
            out_format="netcdf4",
            file_filter=lambda m: m["file_name"].lower().endswith(".nc"),
            reuse_existing=reuse_existing,
        )
        if not files:
            raise RuntimeError(
                "AppEEARS task produced no NetCDF files for this request."
            )

        raw = _open_bundle(files)
        ds = _standardise(raw, layer_map)
        ds = ds.rio.clip_box(lonlim[0], latlim[0], lonlim[1], latlim[1])
        ds = _drop_time(ds)

        ds = apply_enhancers(post_processors, ds)

        ds = ds[list(post_processors.keys())]

        ds.attrs = {}
        for var in ds.data_vars:
            ds[var].attrs = {}

        ds = save_ds(ds, fn, encoding="initiate", label="Saving AppEEARS bundle.")
    finally:
        if raw is not None:
            try:
                raw.close()
            except Exception:
                pass
        shutil.rmtree(staging, ignore_errors=True)

    return ds[req_vars_orig]


if __name__ == "__main__":
    from pywapor.general.logger import adjust_logger

    out_folder = "/Users/hmcoerver/Local/srtm_test"
    os.makedirs(out_folder, exist_ok=True)
    adjust_logger(True, out_folder, "INFO")

    latlim = [28.9, 29.7]
    lonlim = [30.2, 31.2]

    ds = download(out_folder, latlim, lonlim, req_vars=["z", "slope", "aspect"])
    print(ds)
    print("CRS:", ds.rio.crs)
    print("Vars:", list(ds.data_vars))
