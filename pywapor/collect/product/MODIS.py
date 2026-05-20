"""MODIS collection via NASA AppEEARS.

The on-prem LP DAAC OPeNDAP Hyrax service that this module previously targeted
was retired on 2025-09-19; AppEEARS is NASA's recommended replacement for MODIS
subsetting workflows.

`default_vars` and `default_post_processors` are preserved so that
`pywapor.main.Configuration.has_var` (validation) and `MODIS_cloud` (which
reuses the same layer/post-processor tables) keep working.
"""

import copy
import os
import shutil
import tempfile
import time
import warnings
from functools import partial

import numpy as np
import pandas as pd
import xarray as xr

import pywapor.collect.accounts as accounts
import pywapor.collect.protocol.appeears as appeears
from pywapor.enhancers.apply_enhancers import apply_enhancers
from pywapor.general import bitmasks
from pywapor.general.logger import log
from pywapor.general.processing_functions import (
    adjust_timelim_dtype,
    open_ds,
    save_ds,
)

# Per-product, per-req-var: AppEEARS layer name -> pywapor variable name.
# These must match the layers expected by `default_post_processors` so the
# masking/expansion functions keep working unchanged.
LAYERS = {
    "MOD11A1.061": {
        "lst": {
            "LST_Day_1km": "lst",
            "Day_view_time": "lst_hour",
            "QC_Day": "lst_qa",
        },
    },
    "MYD11A1.061": {
        "lst": {
            "LST_Day_1km": "lst",
            "Day_view_time": "lst_hour",
            "QC_Day": "lst_qa",
        },
    },
    "MOD13Q1.061": {
        "ndvi": {
            "_250m_16_days_NDVI": "ndvi",
            "_250m_16_days_pixel_reliability": "ndvi_qa",
        },
    },
    "MYD13Q1.061": {
        "ndvi": {
            "_250m_16_days_NDVI": "ndvi",
            "_250m_16_days_pixel_reliability": "ndvi_qa",
        },
    },
    "MCD43A3.061": {
        "r0": {
            "Albedo_WSA_shortwave": "white_r0",
            "Albedo_BSA_shortwave": "black_r0",
            "BRDF_Albedo_Band_Mandatory_Quality_shortwave": "r0_qa",
        },
    },
}

SPATIAL_BUFFER = {
    "MOD11A1.061": True,
    "MYD11A1.061": True,
    "MCD43A3.061": True,
    "MOD13Q1.061": False,
    "MYD13Q1.061": False,
}


def shortwave_r0(ds, *args):
    ds["r0"] = 0.3 * ds["white_r0"] + 0.7 * ds["black_r0"]
    ds = ds.drop_vars(["white_r0", "black_r0"])
    return ds


def expand_time_dim(ds, *args):
    """MODIS lst data comes with a variable specifying the acquisition decimal time per pixel, This function
    expands the "date" dimension of the data with "time", i.e. afterwards each temporal-slice in the dataset
    contains data at one specific datetime.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset, should have `lst_hour` variable.

    Returns
    -------
    xr.Dataset
        Expanded dataset.
    """
    groups = ds.groupby(ds.lst_hour)

    def _expand_hour_dim(x):
        hour = np.timedelta64(int(np.nanmedian(x.lst_hour.values) * 3600 * 1e9), "ns")
        x = x.assign_coords({"hour": hour}).expand_dims("hour")
        return x

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Slicing with an out-of-order index")

        ds_expand = groups.map(_expand_hour_dim)

        ds_expand = ds_expand.stack({"datetime": ("hour", "time")})

        new_coords = [
            time + hour
            for time, hour in zip(ds_expand.time.values, ds_expand.hour.values)
        ]

        try:  # new versions of xarray require to drop all dimensions of a multi-index
            ds_expand = ds_expand.drop_vars(["datetime", "hour", "time"])
        except ValueError:  # old versions throw an error when trying to drop sub-dimensions of a multiindex.
            ds_expand = ds_expand.drop_vars(["datetime"])

        ds_expand = (
            ds_expand.assign_coords({"datetime": new_coords})
            .rename({"datetime": "time"})
            .sortby("time")
        )
        ds_expand = ds_expand.drop_vars(["lst_hour"])
        ds_expand = ds_expand.transpose("time", "y", "x")
        ds_expand = ds_expand.dropna("time", how="all")

    return ds_expand


def mask_bitwise_qa(
    ds, var, masker="lst_qa", product_name="MOD11A1.061", flags=["good_qa"]
):
    """Mask MODIS data using a qa variable.

    Parameters
    ----------
    ds : xr.Dataset
        Input data.
    var : str
        Variable in `ds` to mask.
    masker : str, optional
        Variable in `ds` to use for masking, by default "lst_qa".
    product_name : str, optional
        Name of the product, by default "MOD11A1.061".
    flags : list, optional
        Which flags not to mask, by default ["good_qa"].

    Returns
    -------
    xr.Dataset
        Masked dataset.
    """

    new_data = ds[var]

    flag_bits = bitmasks.MODIS_qa_translator(product_name)
    mask = bitmasks.get_mask(ds[masker].astype("uint8"), flags, flag_bits)
    new_data = ds[var].where(mask, np.nan)
    ds = ds.drop_vars([masker])

    ds[var] = new_data

    return ds


def mask_qa(ds, var, masker=("ndvi_qa", 1.0)):
    """Mask MODIS data using a qa variable.

    Parameters
    ----------
    ds : xr.Dataset
        Input data
    var : str
        Variable name in `ds` to be masked.
    masker : tuple, optional
        Variable in `ds` to use for masking, second value defines which value in mask to
        use as valid data, by default ("ndvi_qa", 1.0).

    Returns
    -------
    xr.Dataset
        Masked dataset.
    """

    new_data = ds[var].where(ds[masker[0]] != masker[1], np.nan)
    ds = ds.drop_vars(masker[0])

    ds[var] = new_data

    return ds


def default_vars(product_name, req_vars):
    """Given a `product_name` and a list of requested variables, returns a dictionary
    with metadata on which exact layers need to be requested, how they should
    be renamed, and how their dimensions are defined.

    The AppEEARS-backed `download` in this module derives its layer list from
    `LAYERS` directly and does not consume this dictionary, but it is preserved
    in the legacy (dim-tuple, pywapor-name) shape for two consumers:

    - `pywapor.main.Configuration.has_var` — uses a `TypeError` from this
      function to flag an invalid `req_vars` entry against `product_name`.
    - `pywapor.collect.product.MODIS_cloud.download` — reads the dim tuples
      to identify which entries are HDF-EOS data fields vs. coordinate /
      projection helpers.

    Parameters
    ----------
    product_name : str
        Name of the product.
    req_vars : list
        List of variables to be collected.

    Returns
    -------
    dict
        Metadata on which exact layers need to be requested from the server.
    """

    variables = {
        "MOD13Q1.061": {
            "_250m_16_days_NDVI": [("time", "YDim", "XDim"), "ndvi"],
            "_250m_16_days_pixel_reliability": [("time", "YDim", "XDim"), "ndvi_qa"],
            "MODIS_Grid_16DAY_250m_500m_VI_eos_cf_projection": [(), "spatial_ref"],
        },
        "MYD13Q1.061": {
            "_250m_16_days_NDVI": [("time", "YDim", "XDim"), "ndvi"],
            "_250m_16_days_pixel_reliability": [("time", "YDim", "XDim"), "ndvi_qa"],
            "MODIS_Grid_16DAY_250m_500m_VI_eos_cf_projection": [(), "spatial_ref"],
        },
        "MOD11A1.061": {
            "LST_Day_1km": [("time", "YDim", "XDim"), "lst"],
            "Day_view_time": [("time", "YDim", "XDim"), "lst_hour"],
            "QC_Day": [("time", "YDim", "XDim"), "lst_qa"],
            "MODIS_Grid_Daily_1km_LST_eos_cf_projection": [(), "spatial_ref"],
        },
        "MYD11A1.061": {
            "LST_Day_1km": [("time", "YDim", "XDim"), "lst"],
            "Day_view_time": [("time", "YDim", "XDim"), "lst_hour"],
            "QC_Day": [("time", "YDim", "XDim"), "lst_qa"],
            "MODIS_Grid_Daily_1km_LST_eos_cf_projection": [(), "spatial_ref"],
        },
        "MCD43A3.061": {
            "Albedo_WSA_shortwave": [("time", "YDim", "XDim"), "white_r0"],
            "Albedo_BSA_shortwave": [("time", "YDim", "XDim"), "black_r0"],
            "BRDF_Albedo_Band_Mandatory_Quality_shortwave": [
                ("time", "YDim", "XDim"),
                "r0_qa",
            ],
            "MOD_Grid_BRDF_eos_cf_projection": [(), "spatial_ref"],
        },
    }

    req_dl_vars = {
        "MOD13Q1.061": {
            "ndvi": [
                "_250m_16_days_NDVI",
                "_250m_16_days_pixel_reliability",
                "MODIS_Grid_16DAY_250m_500m_VI_eos_cf_projection",
            ],
        },
        "MYD13Q1.061": {
            "ndvi": [
                "_250m_16_days_NDVI",
                "_250m_16_days_pixel_reliability",
                "MODIS_Grid_16DAY_250m_500m_VI_eos_cf_projection",
            ],
        },
        "MOD11A1.061": {
            "lst": [
                "LST_Day_1km",
                "Day_view_time",
                "QC_Day",
                "MODIS_Grid_Daily_1km_LST_eos_cf_projection",
            ],
        },
        "MYD11A1.061": {
            "lst": [
                "LST_Day_1km",
                "Day_view_time",
                "QC_Day",
                "MODIS_Grid_Daily_1km_LST_eos_cf_projection",
            ],
        },
        "MCD43A3.061": {
            "r0": [
                "Albedo_WSA_shortwave",
                "Albedo_BSA_shortwave",
                "BRDF_Albedo_Band_Mandatory_Quality_shortwave",
                "MOD_Grid_BRDF_eos_cf_projection",
            ],
        },
    }

    out = {
        val: variables[product_name][val]
        for sublist in map(req_dl_vars[product_name].get, req_vars)
        for val in sublist
    }

    return out


def default_post_processors(product_name, req_vars=None):
    """Given a `product_name` and a list of requested variables, returns a dictionary with a
    list of functions per variable that should be applied after having collected the data
    from a server.

    Parameters
    ----------
    product_name : str
        Name of the product.
    req_vars : list
        List of variables to be collected.

    Returns
    -------
    dict
        Functions per variable that should be applied to the variable.
    """

    post_processors = {
        "MOD13Q1.061": {"ndvi": [mask_qa]},
        "MYD13Q1.061": {"ndvi": [mask_qa]},
        "MOD11A1.061": {"lst": [mask_bitwise_qa, expand_time_dim]},
        "MYD11A1.061": {"lst": [mask_bitwise_qa, expand_time_dim]},
        "MCD43A3.061": {
            "r0": [
                shortwave_r0,
                partial(mask_qa, masker=("r0_qa", 1.0)),
            ]
        },
    }

    out = {k: v for k, v in post_processors[product_name].items() if k in req_vars}

    return out


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
    """Coerce an AppEEARS-produced NetCDF into the schema pywapor expects.

    - Rename layer variables to pywapor names.
    - Rename spatial dims/coords to `x` / `y`.
    - Ensure `y` is decreasing, `x` is increasing.
    - Attach EPSG:4326 via rioxarray (adds `spatial_ref` coord).
    """
    # Drop AppEEARS-internal scalar CRS variable if present; rioxarray will
    # re-create a clean `spatial_ref` coord below.
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

    # AppEEARS MODIS NetCDF uses a Julian calendar that xarray decodes to
    # `cftime.DatetimeJulian`. Downstream code (e.g. expand_time_dim) does
    # arithmetic with `np.timedelta64`, which only works on `np.datetime64`.
    if "time" in ds.coords and not np.issubdtype(ds["time"].dtype, np.datetime64):
        ds = ds.assign_coords(
            time=np.array([np.datetime64(t.isoformat()) for t in ds["time"].values])
        )

    # AppEEARS bundles carry their own `grid_mapping` / `coordinates`
    # references (pointing at `crs`, `lat`, `lon`) in both attrs and
    # encoding. After dropping the bundle's CRS var and renaming the spatial
    # dims, those references go stale; rioxarray then refuses to attach a
    # fresh `spatial_ref` via `write_crs`. Clear them on every data var.
    for var in list(ds.data_vars):
        for key in ("grid_mapping", "coordinates"):
            ds[var].attrs.pop(key, None)
            ds[var].encoding.pop(key, None)

    ds = ds.rio.write_crs(4326).rio.write_grid_mapping("spatial_ref")
    return ds


def _open_bundle(nc_files):
    """Open AppEEARS bundle NetCDFs and merge.

    AppEEARS area+netcdf4 output is typically one file per product with all
    requested layers and dates concatenated. To stay robust against multiple
    files (e.g. multiple products) we merge by intersection of dims.
    """
    if len(nc_files) == 1:
        return xr.open_dataset(nc_files[0], decode_coords="all")
    dss = [xr.open_dataset(f, decode_coords="all") for f in nc_files]
    return xr.merge(dss, compat="override", join="outer")


def most_recent(product_name, latlim, lonlim):
    # AppEEARS does not expose a cheap "latest granule" probe analogous to
    # the OPeNDAP NCML aggregation. Return None to signal "unknown" until a
    # CMR-based lookup is wired in.
    return None


def download(
    folder,
    latlim,
    lonlim,
    timelim,
    product_name,
    req_vars,
    variables=None,
    post_processors=None,
    reuse_existing=True,
):
    """Download MODIS data via NASA AppEEARS and store it in a single netCDF file.

    Parameters
    ----------
    folder : str
        Path to folder in which to store results.
    latlim : list
        Latitude limits of area of interest.
    lonlim : list
        Longitude limits of area of interest.
    timelim : list
        Period for which to prepare data.
    product_name : str
        Name of the product to download.
    req_vars : list
        Which variables to download for the selected product.
    variables : dict, optional
        OPeNDAP-era layer-selection dict; ignored. Layer selection is driven by
        `req_vars` against `LAYERS`.
    post_processors : dict, optional
        Functions per variable that should be applied to the variable, by
        default None.
    reuse_existing : bool, optional
        If True (default), checks the user's AppEEARS task history for a
        matching prior order before submitting; matches are reused. Set to
        False to force a fresh order.

    Returns
    -------
    xr.Dataset
        Downloaded data.
    """
    if variables is not None:
        log.warning(
            "MODIS: `variables` kwarg is ignored — layer selection is driven "
            "by `req_vars` against the LAYERS table."
        )

    if product_name not in LAYERS:
        raise NotImplementedError(
            f"MODIS AppEEARS backend supports {sorted(LAYERS)}; "
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

    layer_map = _resolve_layer_map(product_name, req_vars)
    product_layers = [(product_name, layer) for layer in layer_map.keys()]

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
    try:
        files = appeears.run_area_task(
            username=un_pw[0],
            password=un_pw[1],
            task_name=f"pywapor_{product_name}_{int(time.time())}",
            product_layers=product_layers,
            latlim=latlim,
            lonlim=lonlim,
            timelim=[pd.Timestamp(t) for t in timelim],
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

        ds = apply_enhancers(post_processors, ds)

        if isinstance(timedelta, np.timedelta64):
            ds["time"] = ds["time"] + timedelta

        ds = ds[list(post_processors.keys())]

        ds.attrs = {}
        for var in ds.data_vars:
            ds[var].attrs = {}

        ds = save_ds(ds, fn, encoding="initiate", label="Saving AppEEARS bundle.")
    finally:
        try:
            raw.close()
        except Exception:
            pass
        shutil.rmtree(staging, ignore_errors=True)

    return ds[req_vars_orig]


if __name__ == "__main__":
    from pywapor.general.logger import adjust_logger

    out_folder = "/Users/hmcoerver/Local/modis_test"
    os.makedirs(out_folder, exist_ok=True)
    adjust_logger(True, out_folder, "INFO")

    for x in LAYERS.keys():
        vars = LAYERS[x].keys()
        for var in vars:

            ds = download(
                folder=out_folder,
                latlim=[29.4, 29.5],
                lonlim=[30.7, 30.8],
                timelim=["2019-03-01", "2019-03-05"],
                product_name=x,
                req_vars=[var],
            )

            print(ds)
            print("CRS:", ds.rio.crs)
            print("Vars:", list(ds.data_vars))
