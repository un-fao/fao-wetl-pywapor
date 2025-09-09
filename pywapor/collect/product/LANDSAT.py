"""
LXSS_LLLL_PPPRRR_YYYYMMDD_yyyymmdd_CX_TX
(e.g., LC08_L2SP_039037_20150728_20200318_02_T1)
L Landsat
X Sensor (“O” = OLI; “T” = TIRS; “C” = OLI/TIRS)
SS Satellite (“08” = Landsat 8, “09” = Landsat 9)
LLLL Processing correction level (“L2SP” if SR and ST are generated or “L2SR”
if ST could not be generated) PPP Path
RRR Row
YYYY Year of acquisition
MM Month of acquisition
DD Day of acquisition
yyyy Year of Level 2 processing
mm Month of Level 2 processing
dd Day of Level 2 processing
CX Collection number (“02”)
TX Collection category ( “T1” = Tier 1; “T2” = Tier 2)
"""

import copy
import datetime
import glob
import json
import os
import re
import shutil
import tarfile
import time
import warnings
from functools import partial

import numpy as np
import pandas as pd
import rasterio
import requests as r
import urllib3
import xarray as xr
import xmltodict

import pywapor
from pywapor.collect.protocol.crawler import download_urls
from pywapor.enhancers.apply_enhancers import apply_enhancer, apply_enhancers
from pywapor.enhancers.gap_fill import gap_fill
from pywapor.general import bitmasks
from pywapor.general.logger import adjust_logger, log
from pywapor.general.processing_functions import (
    adjust_timelim_dtype,
    make_example_ds,
    open_ds,
    remove_ds,
    save_ds,
)


def calc_r0(ds, var, product_name=None):
    """Calculate the Albedo.

    Parameters
    ----------
    ds : xr.Dataset
        Input data.
    var : str
        Name of the variable in which to store the normalized difference.

    Returns
    -------
    xr.Dataset
        Output data.
    """

    weights = {
        "LT05_SR": {
            "blue": 0.116,
            "green": 0.010,
            "red": 0.364,
            "nir": 0.360,
            "offset": 0.032,
        },
        "LE07_SR": {
            "blue": 0.085,
            "green": 0.057,
            "red": 0.349,
            "nir": 0.359,
            "offset": 0.033,
        },
        "LC08_SR": {
            "blue": 0.141,
            "green": 0.114,
            "red": 0.322,
            "nir": 0.364,
            "offset": 0.018,
        },
        "LC09_SR": {
            "blue": 0.141,
            "green": 0.114,
            "red": 0.322,
            "nir": 0.364,
            "offset": 0.018,
        },
    }[product_name]

    reqs = ["blue", "green", "red", "nir"]
    if np.all([x in ds.data_vars for x in reqs]):
        ds["offset"] = xr.ones_like(ds["blue"])
        weights_da = xr.DataArray(
            data=list(weights.values()), coords={"band": list(weights.keys())}
        )
        ds["r0"] = (
            ds[reqs + ["offset"]]
            .to_array("band")
            .weighted(weights_da)
            .sum("band", skipna=False)
        )
    else:
        log.warning(
            f"--> Couldn't calculate `{var}`, `{'` and `'.join([x for x in reqs if x not in ds.data_vars])}` is missing."
        )
    return ds


def calc_normalized_difference(ds, var, bands=["nir", "red"]):
    """Calculate the normalized difference of two bands.

    Parameters
    ----------
    ds : xr.Dataset
        Input data.
    var : str
        Name of the variable in which to store the normalized difference.
    bands : list, optional
        The two bands to use to calculate the norm. difference, by default ["nir", "red"].

    Returns
    -------
    xr.Dataset
        Output data.
    """
    if np.all([x in ds.data_vars for x in bands]):
        da = (ds[bands[0]] - ds[bands[1]]) / (ds[bands[0]] + ds[bands[1]])
        ds[var] = da.clip(-1, 1)
    else:
        log.warning(
            f"--> Couldn't calculate `{var}`, `{'` and `'.join([x for x in bands if x not in ds.data_vars])}` is missing."
        )
    return ds


def mask_uncertainty(ds, var, max_uncertainty=3):
    if "lst_qa" in ds.data_vars:
        ds[var] = ds[var].where(ds["lst_qa"] <= max_uncertainty, np.nan)
    else:
        log.warning("--> Couldn't mask uncertain values.")
    return ds


def mask_invalid(ds, var):
    if "valid_range" in ds[var].attrs.keys():
        ds[var] = xr.where(
            (ds[var] >= ds[var].valid_range[0])
            & (ds[var] <= ds[var].valid_range[1])
            & (ds[var] != ds[var]._FillValue),
            ds[var],
            np.nan,
            keep_attrs=True,
        )
    else:
        log.warning(
            "--> Couldn't mask invalid values, since `valid_range` is not defined."
        )
    return ds


def apply_qa(ds, var, pixel_qa_flags=None, radsat_qa_flags=None, product_name=None):
    masks = list()

    if ("pixel_qa" in ds.data_vars) and not isinstance(pixel_qa_flags, type(None)):
        pixel_qa_bits = bitmasks.get_pixel_qa_bits(
            2, int(product_name.split("_")[0][-1]), 2
        )
        mask1 = bitmasks.get_mask(ds["pixel_qa"], pixel_qa_flags, pixel_qa_bits)
        masks.append(mask1)
        # Don't allow negative reflectances
        ds[var] = ds[var].clip(0, np.inf)

    if ("radsat_qa" in ds.data_vars) and not isinstance(radsat_qa_flags, type(None)):
        radsat_qa_bits = bitmasks.get_radsat_qa_bits(
            2, int(product_name.split("_")[0][-1]), 2
        )
        radsat_qa_flags = list(radsat_qa_bits.keys())
        mask2 = bitmasks.get_mask(ds["radsat_qa"], radsat_qa_flags, radsat_qa_bits)
        masks.append(mask2)

    if len(masks) >= 1:
        mask = np.invert(np.any(masks, axis=0))
        ds[var] = ds[var].where(mask)

    return ds


def scale_data(ds, var):
    scale = getattr(ds[var], "scale_factor", 1.0)
    offset = getattr(ds[var], "add_offset", 0.0)
    ds[var] = ds[var] * scale + offset
    return ds


def default_vars(product_name, req_vars):
    # {x: [ds[x].dims, ds[x].long_name] for x in ds.data_vars}

    pixel_qa_flags_89 = ["dilated_cloud", "cirrus", "cloud", "cloud_shadow", "snow"]
    pixel_qa_flags_457 = ["dilated_cloud", "cloud", "cloud_shadow", "snow"]

    variables = {
        "LT05_SR": {
            "sr_band1": [
                ("YDim_sr_band1", "XDim_sr_band1"),
                "blue",
                [
                    mask_invalid,
                    scale_data,
                    partial(
                        apply_qa,
                        product_name="LT05_SR",
                        pixel_qa_flags=pixel_qa_flags_457,
                        radsat_qa_flags=["terrain_occlusion", "saturated_band1"],
                    ),
                ],
            ],
            "sr_band2": [
                ("YDim_sr_band2", "XDim_sr_band2"),
                "green",
                [
                    mask_invalid,
                    scale_data,
                    partial(
                        apply_qa,
                        product_name="LT05_SR",
                        pixel_qa_flags=pixel_qa_flags_457,
                        radsat_qa_flags=["terrain_occlusion", "saturated_band2"],
                    ),
                ],
            ],
            "sr_band3": [
                ("YDim_sr_band3", "XDim_sr_band3"),
                "red",
                [
                    mask_invalid,
                    scale_data,
                    partial(
                        apply_qa,
                        product_name="LT05_SR",
                        pixel_qa_flags=pixel_qa_flags_457,
                        radsat_qa_flags=["terrain_occlusion", "saturated_band3"],
                    ),
                ],
            ],
            "sr_band4": [
                ("YDim_sr_band4", "XDim_sr_band4"),
                "nir",
                [
                    mask_invalid,
                    scale_data,
                    partial(
                        apply_qa,
                        product_name="LT05_SR",
                        pixel_qa_flags=pixel_qa_flags_457,
                        radsat_qa_flags=["terrain_occlusion", "saturated_band4"],
                    ),
                ],
            ],
            "sr_band5": [
                ("YDim_sr_band5", "XDim_sr_band5"),
                "swir1",
                [
                    mask_invalid,
                    scale_data,
                    partial(
                        apply_qa,
                        product_name="LT05_SR",
                        pixel_qa_flags=pixel_qa_flags_457,
                        radsat_qa_flags=["terrain_occlusion", "saturated_band5"],
                    ),
                ],
            ],
            "sr_band7": [
                ("YDim_sr_band7", "XDim_sr_band7"),
                "swir2",
                [
                    mask_invalid,
                    scale_data,
                    partial(
                        apply_qa,
                        product_name="LT05_SR",
                        pixel_qa_flags=pixel_qa_flags_457,
                        radsat_qa_flags=["terrain_occlusion", "saturated_band7"],
                    ),
                ],
            ],
            "qa_pixel": [("YDim_qa_pixel", "XDim_qa_pixel"), "pixel_qa", []],
            "qa_radsat": [("YDim_qa_radsat", "XDim_qa_radsat"), "radsat_qa", []],
        },
        "LT05_ST": {
            "st_band6": [
                ("YDim_st_band6", "XDim_st_band6"),
                "lst",
                [mask_invalid, scale_data],
            ],
            "st_qa": [
                ("YDim_st_qa", "XDim_st_qa"),
                "lst_qa",
                [mask_invalid, scale_data],
            ],
        },
        "LE07_SR": {
            "sr_band1": [
                ("YDim_sr_band1", "XDim_sr_band1"),
                "blue",
                [
                    mask_invalid,
                    scale_data,
                    partial(
                        apply_qa,
                        product_name="LE07_SR",
                        pixel_qa_flags=pixel_qa_flags_457,
                        radsat_qa_flags=["terrain_occlusion", "saturated_band1"],
                    ),
                ],
            ],
            "sr_band2": [
                ("YDim_sr_band2", "XDim_sr_band2"),
                "green",
                [
                    mask_invalid,
                    scale_data,
                    partial(
                        apply_qa,
                        product_name="LE07_SR",
                        pixel_qa_flags=pixel_qa_flags_457,
                        radsat_qa_flags=["terrain_occlusion", "saturated_band2"],
                    ),
                ],
            ],
            "sr_band3": [
                ("YDim_sr_band3", "XDim_sr_band3"),
                "red",
                [
                    mask_invalid,
                    scale_data,
                    partial(
                        apply_qa,
                        product_name="LE07_SR",
                        pixel_qa_flags=pixel_qa_flags_457,
                        radsat_qa_flags=["terrain_occlusion", "saturated_band3"],
                    ),
                ],
            ],
            "sr_band4": [
                ("YDim_sr_band4", "XDim_sr_band4"),
                "nir",
                [
                    mask_invalid,
                    scale_data,
                    partial(
                        apply_qa,
                        product_name="LE07_SR",
                        pixel_qa_flags=pixel_qa_flags_457,
                        radsat_qa_flags=["terrain_occlusion", "saturated_band4"],
                    ),
                ],
            ],
            "sr_band5": [
                ("YDim_sr_band5", "XDim_sr_band5"),
                "swir1",
                [
                    mask_invalid,
                    scale_data,
                    partial(
                        apply_qa,
                        product_name="LE07_SR",
                        pixel_qa_flags=pixel_qa_flags_457,
                        radsat_qa_flags=["terrain_occlusion", "saturated_band5"],
                    ),
                ],
            ],
            "sr_band7": [
                ("YDim_sr_band7", "XDim_sr_band7"),
                "swir2",
                [
                    mask_invalid,
                    scale_data,
                    partial(
                        apply_qa,
                        product_name="LE07_SR",
                        pixel_qa_flags=pixel_qa_flags_457,
                        radsat_qa_flags=["terrain_occlusion", "saturated_band7"],
                    ),
                ],
            ],
            "qa_pixel": [("YDim_qa_pixel", "XDim_qa_pixel"), "pixel_qa", []],
            "qa_radsat": [("YDim_qa_radsat", "XDim_qa_radsat"), "radsat_qa", []],
        },
        "LE07_ST": {
            "st_band6": [
                ("YDim_st_band6", "XDim_st_band6"),
                "lst",
                [mask_invalid, scale_data],
            ],
            "st_qa": [
                ("YDim_st_qa", "XDim_st_qa"),
                "lst_qa",
                [mask_invalid, scale_data],
            ],
        },
        "LC08_SR": {
            "sr_band1": [
                ("YDim_sr_band1", "XDim_sr_band1"),
                "coastal",
                [
                    mask_invalid,
                    scale_data,
                    partial(
                        apply_qa,
                        product_name="LC08_SR",
                        pixel_qa_flags=pixel_qa_flags_89,
                        radsat_qa_flags=["terrain_occlusion", "saturated_band1"],
                    ),
                ],
            ],
            "sr_band2": [
                ("YDim_sr_band2", "XDim_sr_band2"),
                "blue",
                [
                    mask_invalid,
                    scale_data,
                    partial(
                        apply_qa,
                        product_name="LC08_SR",
                        pixel_qa_flags=pixel_qa_flags_89,
                        radsat_qa_flags=["terrain_occlusion", "saturated_band2"],
                    ),
                ],
            ],
            "sr_band3": [
                ("YDim_sr_band3", "XDim_sr_band3"),
                "green",
                [
                    mask_invalid,
                    scale_data,
                    partial(
                        apply_qa,
                        product_name="LC08_SR",
                        pixel_qa_flags=pixel_qa_flags_89,
                        radsat_qa_flags=["terrain_occlusion", "saturated_band3"],
                    ),
                ],
            ],
            "sr_band4": [
                ("YDim_sr_band4", "XDim_sr_band4"),
                "red",
                [
                    mask_invalid,
                    scale_data,
                    partial(
                        apply_qa,
                        product_name="LC08_SR",
                        pixel_qa_flags=pixel_qa_flags_89,
                        radsat_qa_flags=["terrain_occlusion", "saturated_band4"],
                    ),
                ],
            ],
            "sr_band5": [
                ("YDim_sr_band5", "XDim_sr_band5"),
                "nir",
                [
                    mask_invalid,
                    scale_data,
                    partial(
                        apply_qa,
                        product_name="LC08_SR",
                        pixel_qa_flags=pixel_qa_flags_89,
                        radsat_qa_flags=["terrain_occlusion", "saturated_band5"],
                    ),
                ],
            ],
            "sr_band6": [
                ("YDim_sr_band6", "XDim_sr_band6"),
                "swir1",
                [
                    mask_invalid,
                    scale_data,
                    partial(
                        apply_qa,
                        product_name="LC08_SR",
                        pixel_qa_flags=pixel_qa_flags_89,
                        radsat_qa_flags=["terrain_occlusion", "saturated_band6"],
                    ),
                ],
            ],
            "sr_band7": [
                ("YDim_sr_band7", "XDim_sr_band7"),
                "swir2",
                [
                    mask_invalid,
                    scale_data,
                    partial(
                        apply_qa,
                        product_name="LC08_SR",
                        pixel_qa_flags=pixel_qa_flags_89,
                        radsat_qa_flags=["terrain_occlusion", "saturated_band7"],
                    ),
                ],
            ],
            "qa_pixel": [("YDim_qa_pixel", "XDim_qa_pixel"), "pixel_qa", []],
            "qa_radsat": [("YDim_qa_radsat", "XDim_qa_radsat"), "radsat_qa", []],
        },
        "LC08_ST": {
            "st_band10": [
                ("YDim_st_band10", "XDim_st_band10"),
                "lst",
                [mask_invalid, scale_data],
            ],
            "st_qa": [
                ("YDim_st_qa", "XDim_st_qa"),
                "lst_qa",
                [mask_invalid, scale_data],
            ],
        },
        "LC09_SR": {
            "sr_band1": [
                ("YDim_sr_band1", "XDim_sr_band1"),
                "coastal",
                [
                    mask_invalid,
                    scale_data,
                    partial(
                        apply_qa,
                        product_name="LC09_SR",
                        pixel_qa_flags=pixel_qa_flags_89,
                        radsat_qa_flags=["terrain_occlusion", "saturated_band1"],
                    ),
                ],
            ],
            "sr_band2": [
                ("YDim_sr_band2", "XDim_sr_band2"),
                "blue",
                [
                    mask_invalid,
                    scale_data,
                    partial(
                        apply_qa,
                        product_name="LC09_SR",
                        pixel_qa_flags=pixel_qa_flags_89,
                        radsat_qa_flags=["terrain_occlusion", "saturated_band2"],
                    ),
                ],
            ],
            "sr_band3": [
                ("YDim_sr_band3", "XDim_sr_band3"),
                "green",
                [
                    mask_invalid,
                    scale_data,
                    partial(
                        apply_qa,
                        product_name="LC09_SR",
                        pixel_qa_flags=pixel_qa_flags_89,
                        radsat_qa_flags=["terrain_occlusion", "saturated_band3"],
                    ),
                ],
            ],
            "sr_band4": [
                ("YDim_sr_band4", "XDim_sr_band4"),
                "red",
                [
                    mask_invalid,
                    scale_data,
                    partial(
                        apply_qa,
                        product_name="LC09_SR",
                        pixel_qa_flags=pixel_qa_flags_89,
                        radsat_qa_flags=["terrain_occlusion", "saturated_band4"],
                    ),
                ],
            ],
            "sr_band5": [
                ("YDim_sr_band5", "XDim_sr_band5"),
                "nir",
                [
                    mask_invalid,
                    scale_data,
                    partial(
                        apply_qa,
                        product_name="LC09_SR",
                        pixel_qa_flags=pixel_qa_flags_89,
                        radsat_qa_flags=["terrain_occlusion", "saturated_band5"],
                    ),
                ],
            ],
            "sr_band6": [
                ("YDim_sr_band6", "XDim_sr_band6"),
                "swir1",
                [
                    mask_invalid,
                    scale_data,
                    partial(
                        apply_qa,
                        product_name="LC09_SR",
                        pixel_qa_flags=pixel_qa_flags_89,
                        radsat_qa_flags=["terrain_occlusion", "saturated_band6"],
                    ),
                ],
            ],
            "sr_band7": [
                ("YDim_sr_band7", "XDim_sr_band7"),
                "swir2",
                [
                    mask_invalid,
                    scale_data,
                    partial(
                        apply_qa,
                        product_name="LC09_SR",
                        pixel_qa_flags=pixel_qa_flags_89,
                        radsat_qa_flags=["terrain_occlusion", "saturated_band7"],
                    ),
                ],
            ],
            "qa_pixel": [("YDim_qa_pixel", "XDim_qa_pixel"), "pixel_qa", []],
            "qa_radsat": [("YDim_qa_radsat", "XDim_qa_radsat"), "radsat_qa", []],
        },
        "LC09_ST": {
            "st_band10": [
                ("YDim_st_band10", "XDim_st_band10"),
                "lst",
                [mask_invalid, scale_data],
            ],
            "st_qa": [
                ("YDim_st_qa", "XDim_st_qa"),
                "lst_qa",
                [mask_invalid, scale_data],
            ],
        },
    }

    req_dl_vars = {
        "LT05_SR": {
            "blue": ["sr_band1", "qa_pixel", "qa_radsat"],
            "green": ["sr_band2", "qa_pixel", "qa_radsat"],
            "red": ["sr_band3", "qa_pixel", "qa_radsat"],
            "nir": ["sr_band4", "qa_pixel", "qa_radsat"],
            "swir1": ["sr_band5", "qa_pixel", "qa_radsat"],
            "swir2": ["sr_band7", "qa_pixel", "qa_radsat"],
            "pixel_qa": ["qa_pixel"],
            "radsat_qa": ["qa_radsat"],
            "ndvi": ["sr_band3", "sr_band4", "qa_pixel", "qa_radsat"],
            "r0": [
                "sr_band1",
                "sr_band2",
                "sr_band3",
                "sr_band4",
                "qa_pixel",
                "qa_radsat",
            ],
        },
        "LT05_ST": {
            "lst": ["st_band6", "st_qa"],
            "lst_qa": ["st_qa"],
        },
        "LE07_SR": {
            "blue": ["sr_band1", "qa_pixel", "qa_radsat"],
            "green": ["sr_band2", "qa_pixel", "qa_radsat"],
            "red": ["sr_band3", "qa_pixel", "qa_radsat"],
            "nir": ["sr_band4", "qa_pixel", "qa_radsat"],
            "swir1": ["sr_band5", "qa_pixel", "qa_radsat"],
            "swir2": ["sr_band7", "qa_pixel", "qa_radsat"],
            "pixel_qa": ["qa_pixel"],
            "radsat_qa": ["qa_radsat"],
            "ndvi": ["sr_band3", "sr_band4", "qa_pixel", "qa_radsat"],
            "r0": [
                "sr_band1",
                "sr_band2",
                "sr_band3",
                "sr_band4",
                "qa_pixel",
                "qa_radsat",
            ],
        },
        "LE07_ST": {
            "lst": ["st_band6", "st_qa"],
            "lst_qa": ["st_qa"],
        },
        "LC08_SR": {
            "coastal": ["sr_band1", "qa_pixel", "qa_radsat"],
            "blue": ["sr_band2", "qa_pixel", "qa_radsat"],
            "green": ["sr_band3", "qa_pixel", "qa_radsat"],
            "red": ["sr_band4", "qa_pixel", "qa_radsat"],
            "nir": ["sr_band5", "qa_pixel", "qa_radsat"],
            "swir1": ["sr_band6", "qa_pixel", "qa_radsat"],
            "swir2": ["sr_band7", "qa_pixel", "qa_radsat"],
            "pixel_qa": ["qa_pixel"],
            "radsat_qa": ["qa_radsat"],
            "ndvi": ["sr_band4", "sr_band5", "qa_pixel", "qa_radsat"],
            "r0": [
                "sr_band2",
                "sr_band3",
                "sr_band4",
                "sr_band5",
                "qa_pixel",
                "qa_radsat",
            ],
        },
        "LC08_ST": {
            "lst": ["st_band10", "st_qa"],
            "lst_qa": ["st_qa"],
        },
        "LC09_SR": {
            "coastal": ["sr_band1", "qa_pixel", "qa_radsat"],
            "blue": ["sr_band2", "qa_pixel", "qa_radsat"],
            "green": ["sr_band3", "qa_pixel", "qa_radsat"],
            "red": ["sr_band4", "qa_pixel", "qa_radsat"],
            "nir": ["sr_band5", "qa_pixel", "qa_radsat"],
            "swir1": ["sr_band6", "qa_pixel", "qa_radsat"],
            "swir2": ["sr_band7", "qa_pixel", "qa_radsat"],
            "pixel_qa": ["qa_pixel"],
            "radsat_qa": ["qa_radsat"],
            "ndvi": ["sr_band4", "sr_band5", "qa_pixel", "qa_radsat"],
            "r0": [
                "sr_band2",
                "sr_band3",
                "sr_band4",
                "sr_band5",
                "qa_pixel",
                "qa_radsat",
            ],
        },
        "LC09_ST": {
            "lst": ["st_band10", "st_qa"],
            "lst_qa": ["st_qa"],
        },
    }

    out = {
        val: variables[product_name][val]
        for sublist in map(req_dl_vars[product_name].get, req_vars)
        for val in sublist
    }

    return out


def default_post_processors(product_name, req_vars):
    post_processors = {
        "LT05_SR": {
            "coastal": [gap_fill],
            "blue": [gap_fill],
            "green": [gap_fill],
            "red": [gap_fill],
            "nir": [gap_fill],
            "swir1": [gap_fill],
            "swir2": [gap_fill],
            "pixel_qa": [],
            "radsat_qa": [],
            "ndvi": [calc_normalized_difference, gap_fill],
            "r0": [partial(calc_r0, product_name="LT05_SR"), gap_fill],
        },
        "LT05_ST": {
            "lst": [mask_uncertainty, gap_fill],
            "lst_qa": [],
        },
        "LE07_SR": {
            "coastal": [gap_fill],
            "blue": [gap_fill],
            "green": [gap_fill],
            "red": [gap_fill],
            "nir": [gap_fill],
            "swir1": [gap_fill],
            "swir2": [gap_fill],
            "pixel_qa": [],
            "radsat_qa": [],
            "ndvi": [calc_normalized_difference, gap_fill],
            "r0": [partial(calc_r0, product_name="LE07_SR"), gap_fill],
        },
        "LE07_ST": {
            "lst": [mask_uncertainty, gap_fill],
            "lst_qa": [],
        },
        "LC08_SR": {
            "coastal": [gap_fill],
            "blue": [gap_fill],
            "green": [gap_fill],
            "red": [gap_fill],
            "nir": [gap_fill],
            "swir1": [gap_fill],
            "swir2": [gap_fill],
            "pixel_qa": [],
            "radsat_qa": [],
            "ndvi": [calc_normalized_difference, gap_fill],
            "r0": [partial(calc_r0, product_name="LC08_SR"), gap_fill],
        },
        "LC08_ST": {
            "lst": [mask_uncertainty, gap_fill],
            "lst_qa": [],
        },
        "LC09_SR": {
            "coastal": [gap_fill],
            "blue": [gap_fill],
            "green": [gap_fill],
            "red": [gap_fill],
            "nir": [gap_fill],
            "swir1": [gap_fill],
            "swir2": [gap_fill],
            "pixel_qa": [],
            "radsat_qa": [],
            "ndvi": [calc_normalized_difference, gap_fill],
            "r0": [partial(calc_r0, product_name="LC09_SR"), gap_fill],
        },
        "LC09_ST": {
            "lst": [mask_uncertainty, gap_fill],
            "lst_qa": [],
        },
    }

    out = {k: v for k, v in post_processors[product_name].items() if k in req_vars}

    return out


def espa_api(endpoint, verb="get", body=None, uauth=None):
    """Suggested simple way to interact with the ESPA JSON REST API"""
    # auth_tup = uauth if uauth else (username, password)
    host = "https://espa.cr.usgs.gov/api/v1/"
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=urllib3.exceptions.InsecureRequestWarning
        )
        response = getattr(r, verb)(
            host + endpoint, auth=uauth, json=body, verify=False
        )
    data = response.json()
    if isinstance(data, dict):
        messages = data.pop("messages", None)
        if messages:
            log.info(json.dumps(messages, indent=4))
    try:
        response.raise_for_status()
    except Exception as e:
        log.warning(str(e))
        return None
    else:
        return data


def search_stac(latlim, lonlim, timelim, product_name, extra_search_kwargs, limit=1000):
    bb = [lonlim[0], latlim[0], lonlim[1], latlim[1]]

    search_kwargs = extra_search_kwargs

    stac = "https://landsatlook.usgs.gov/stac-server"  # Landsat STAC API Endpoint
    stac_response = r.get(stac).json()
    catalog_links = stac_response["links"]
    search = [link["href"] for link in catalog_links if link["rel"] == "search"][
        0
    ]  # retreive search endpoint from STAC Catalog

    params = dict()
    if not isinstance(timelim, type(None)):
        timelim = adjust_timelim_dtype(timelim)
        sd = datetime.datetime.strftime(timelim[0], "%Y-%m-%dT00:00:00Z")
        ed = datetime.datetime.strftime(timelim[1], "%Y-%m-%dT23:59:59Z")
        search_dates = f"{sd}/{ed}"
        params["datetime"] = search_dates
    params["collections"] = ["landsat-c2l2-sr", "landsat-c2l2-st"]
    params["limit"] = limit
    params["bbox"] = bb

    params["query"] = search_kwargs

    query = r.post(
        search,
        json=params,
    ).json()  # send POST request to the stac-search endpoint with params passed in

    ids = np.unique(
        [
            x["id"].replace("_ST", "").replace("_SR", "")
            for x in query["features"]
            if product_name.split("_")[0] in x["id"]
        ]
    ).tolist()

    log.info(f"--> Found {len(ids)} scenes.")

    return ids, query


def request_scenes(ids, image_extents):
    uauth = pywapor.collect.accounts.get("EARTHEXPLORER")

    order = espa_api("available-products", body={"inputs": list(ids)}, uauth=uauth)

    if "not implemented" in order.keys():
        missing = order.pop("not implemented")
        log.warning(f"--> Some scenes could not be found (`{'`, `'.join(missing)}`).")

    req_prods = ["l1"]
    for sensor in order.keys():
        order[sensor]["products"] = [
            x for x in req_prods if x in order[sensor]["products"]
        ]

    order["projection"] = {"lonlat": None}
    order["format"] = "netcdf"
    order["resampling_method"] = "nn"
    order["note"] = f"pyWaPOR_{pywapor.__version__}"
    order["image_extents"] = image_extents

    log.info(f"--> Placing order for {len(ids)} scenes.")

    order_response = espa_api("order", verb="post", body=order, uauth=uauth)

    time.sleep(10)

    return order_response


def unpack(fp, folder):
    fn, _ = os.path.splitext(os.path.split(fp)[-1])
    subfolder = os.path.join(folder, fn)
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)
    shutil.unpack_archive(fp, subfolder)


def check_availabilty(product_folder, product_name, scene_ids):
    paths = glob.glob(os.path.join(product_folder, "**", "*.nc"), recursive=True)
    regex_str = r"_.{4}_\d{6}_\d{8}_\d{8}_\d{2}_T\d.nc"
    unpacked_scenes = set(
        [
            os.path.splitext(os.path.split(f)[-1])[0]
            for f in paths
            if re.search(f"{product_name.split('_')[0]}{regex_str}", f)
        ]
    )
    packed_scenes = {
        [
            y.name
            for y in tarfile.open(x, encoding="utf-8").getmembers()
            if ".nc" in y.name
        ][0][:-3]: x
        for x in glob.glob(
            os.path.join(product_folder, "**", f"{product_name.split('_')[0]}*.tar.gz"),
            recursive=True,
        )
    }
    to_unpack = [
        fp
        for k, fp in packed_scenes.items()
        if (k in scene_ids) and (k not in unpacked_scenes)
    ]
    for fp in to_unpack:
        unpack(fp, product_folder)
    available_scenes = unpacked_scenes.union(set(packed_scenes.keys())).intersection(
        scene_ids
    )
    return available_scenes


def update_order_statuses(
    scene_ids,
    product_folder,
    product_name,
    to_download,
    to_wait,
    to_request,
    uauth,
    image_extents,
    verbose=True,
):
    all_orders__ = espa_api("item-status", uauth=uauth)
    all_orders_ = {
        k: [x for x in v if x.get("status") not in ("purged", "cancelled")]
        for k, v in all_orders__.items()
    }
    all_orders = {k: v for k, v in all_orders_.items() if len(v) != 0}

    available_scenes = check_availabilty(product_folder, product_name, scene_ids)

    for order_id, order in all_orders.items():
        # Get specific order details.
        order_folder = os.path.join(product_folder, "orders")
        order_details_fp = os.path.join(order_folder, f"{order_id}.json")
        if os.path.isfile(order_details_fp):
            with open(order_details_fp, "r") as fp:
                order_details = json.load(fp)
        else:
            order_details = espa_api(f"order/{order_id}", uauth=uauth)
            with open(order_details_fp, "w") as fp:
                json.dump(order_details, fp)

        if order_details["product_opts"].get("image_extents") != image_extents:
            continue

        for scene in order:
            if (
                (scene["status"] == "complete")
                and (scene["name"] in scene_ids)
                and (scene["name"] not in available_scenes)
            ):
                to_download[scene["name"]] = scene["product_dload_url"]
                to_request.discard(scene["name"])
                to_wait.discard(scene["name"])
            elif (
                (scene["status"] == "oncache")
                and (scene["name"] in scene_ids)
                and (scene["name"] not in available_scenes)
            ):
                to_request.discard(scene["name"])
                to_wait.add(scene["name"])
            elif (
                (scene["status"] == "onorder")
                and (scene["name"] in scene_ids)
                and (scene["name"] not in available_scenes)
            ):
                to_request.discard(scene["name"])
                to_wait.add(scene["name"])
            elif (
                (scene["status"] == "queued")
                and (scene["name"] in scene_ids)
                and (scene["name"] not in available_scenes)
            ):
                to_request.discard(scene["name"])
                to_wait.add(scene["name"])
            elif (
                (scene["status"] == "processing")
                and (scene["name"] in scene_ids)
                and (scene["name"] not in available_scenes)
            ):
                to_request.discard(scene["name"])
                to_wait.add(scene["name"])
            elif (
                (scene["status"] == "error")
                and (scene["name"] in scene_ids)
                and (scene["name"] not in available_scenes)
            ):
                to_wait.discard(scene["name"])
                to_request.discard(scene["name"])
                if not verbose:
                    log.info(
                        f"--> Error in `{scene['name']}` request, `{scene['note']}`."
                    )
            elif (
                (scene["status"] == "retry")
                and (scene["name"] in scene_ids)
                and (scene["name"] not in available_scenes)
            ):
                to_request.discard(scene["name"])
                to_wait.add(scene["name"])
            elif (
                (scene["status"] == "unavailable")
                and (scene["name"] in scene_ids)
                and (scene["name"] not in available_scenes)
            ):
                to_request.discard(scene["name"])
                to_wait.discard(scene["name"])
                if not verbose:
                    log.info(
                        f"--> Scene `{scene['name']}` is unavailable, `{scene['note']}`."
                    )
            elif (
                (scene["status"] == "cancelled")
                and (scene["name"] in scene_ids)
                and (scene["name"] not in available_scenes)
            ):
                to_request.add(scene["name"])
                to_wait.discard(scene["name"])
                if not verbose:
                    log.info(
                        f"Scene `{scene['name']}` order is cancelled, `{scene['note']}`. Ordering again."
                    )
            elif (scene["name"] in scene_ids) and (
                scene["name"] not in available_scenes
            ):
                to_request.add(scene["name"])
                to_wait.discard(scene["name"])
            else:
                ...

    return available_scenes, to_download, to_wait, to_request


def download_scenes(
    scene_ids,
    product_folder,
    product_name,
    latlim,
    lonlim,
    max_attempts=5,
    wait_time=300,
):
    image_extents = {
        "east": lonlim[1],
        "north": latlim[1],
        "south": latlim[0],
        "west": lonlim[0],
        "units": "dd",
    }

    uauth = pywapor.collect.accounts.get("EARTHEXPLORER")

    attempt = 0

    # Check which scenes already exist locally
    available_scenes = check_availabilty(product_folder, product_name, scene_ids)
    to_request = set(scene_ids).difference(available_scenes)
    to_wait = set()
    to_download = dict()

    while (
        len(to_download) + len(to_wait) + len(to_request) > 0 and attempt < max_attempts
    ):
        if attempt > 0:
            log.info(f"--> Waiting {wait_time} seconds before trying again.")
            log.info(
                "--> For more info on order status, go to ` https://espa.cr.usgs.gov/ordering/status `."
            )
            time.sleep(wait_time)

        # Update statutes
        available_scenes, to_download, to_wait, to_request = update_order_statuses(
            scene_ids,
            product_folder,
            product_name,
            to_download,
            to_wait,
            to_request,
            uauth,
            image_extents,
            verbose=True,
        )
        # log.info(f"--> {len(available_scenes)} scenes available, {len(to_download)} ready for download, {len(to_request)} need to be requested and {len(to_wait)} are being processed.")

        # Request missing scenes on (ESPA)
        if len(to_request) > 0:
            _ = request_scenes(to_request, image_extents)

        # Download completed scenes (ESPA)
        if len(to_download) > 0:
            log.info(f"--> Downloading {len(to_download)} scenes.")
            fps = download_urls(list(to_download.values()), product_folder)
            for fp in fps:
                unpack(fp, product_folder)
            to_download = dict()

        available_scenes, to_download, to_wait, to_request = update_order_statuses(
            scene_ids,
            product_folder,
            product_name,
            to_download,
            to_wait,
            to_request,
            uauth,
            image_extents,
            verbose=False,
        )
        # log.info(f"--> {len(available_scenes)} scenes available, {len(to_download)} ready for download, {len(to_request)} need to be requested and {len(to_wait)} are being processed.")

        attempt += 1
        log.info(
            f"--> {len(available_scenes)} scenes collected in attempt {attempt}/{max_attempts}, waiting for {len(to_wait) + len(to_request)} more scenes."
        )

    return available_scenes, to_download, to_request, to_wait


def _process_scene(scene, product_folder, product_name, variables, example_ds=None):
    ds = xr.open_dataset(scene, mask_and_scale=False, chunks="auto")[
        list(variables.keys())
    ]

    if len(set([ds[x].shape for x in ds.data_vars])) > 1:
        log.warning("--> Not all variables have identical shapes.")

    xdim = ds[list(variables.values())[0][0][1]].values
    ydim = ds[list(variables.values())[0][0][0]].values

    renames1 = {k: v[1] for k, v in variables.items()}
    renames2 = {v[0][0]: "y" for v in variables.values()}
    renames3 = {v[0][1]: "x" for v in variables.values()}

    ds = ds.rename_dims({**renames2, **renames3})
    ds = ds.drop(list(ds.coords.keys())).assign_coords({"x": xdim, "y": ydim})
    ds = ds.rename(renames1)

    crs = rasterio.crs.CRS.from_epsg(4326)
    ds = ds.rio.write_crs(crs)
    ds = ds.rio.write_grid_mapping("spatial_ref")

    ds = ds.sortby("y", ascending=False)
    ds = ds.sortby("x")

    mtl_fp = glob.glob(
        os.path.join(
            product_folder, "**", f"*{ds.LPGSMetadataFile.replace('.txt', '.xml')}*"
        ),
        recursive=True,
    )[0]
    with open(mtl_fp, "r") as f:
        xml_content = f.read()
    mtl = xmltodict.parse(xml_content)

    # Clip and pad to bounding-box
    if isinstance(example_ds, type(None)):
        example_ds = make_example_ds(
            ds,
            product_folder,
            crs,
            bb=None,
            example_ds_fp=os.path.join(product_folder, f"example_ds_{product_name}.nc"),
        )  # [lonlim[0], latlim[0], lonlim[1], latlim[1]])
    ds = ds.rio.reproject_match(example_ds).chunk("auto")
    ds = ds.assign_coords({"x": example_ds.x, "y": example_ds.y})

    # Add time dimension to data arrays.
    ds = ds.expand_dims({"time": 1})

    # Set the correct time.
    date_str = mtl["LANDSAT_METADATA_FILE"]["IMAGE_ATTRIBUTES"]["DATE_ACQUIRED"]
    time_str = mtl["LANDSAT_METADATA_FILE"]["IMAGE_ATTRIBUTES"]["SCENE_CENTER_TIME"]
    datetime_str = date_str + " " + time_str.replace("Z", "")
    ds = ds.assign_coords({"time": [np.datetime64(datetime_str)]})

    # Apply variable specific functions.
    for vars in variables.values():
        for func in vars[2]:
            ds, _ = apply_enhancer(ds, vars[1], func)

    # Cleanup attributes
    for var in ["x", "y", "time"]:
        ds[var].attrs = {}

    return ds, example_ds


def process_scenes(
    fp, scene_paths, product_folder, product_name, variables, post_processors
):
    dss = list()

    log.info(f"--> Processing {len(scene_paths)} scenes.").add()

    chunks = {"time": 1, "x": 1000, "y": 1000}

    example_ds = None
    for i, scene in enumerate(scene_paths):
        ds, example_ds = _process_scene(
            scene, product_folder, product_name, variables, example_ds=example_ds
        )
        fp_temp = scene.replace(".nc", "_temp.nc")
        ds = save_ds(
            ds,
            fp_temp,
            chunks=chunks,
            encoding="initiate",
            label=f"({i + 1}/{len(scene_paths)}) Processing `{os.path.split(scene)[-1]}`.",
        )
        dss.append(ds)
    remove_ds(example_ds)

    ds = xr.concat(dss, "time")

    # Apply general product functions.
    ds = apply_enhancers(post_processors, ds)

    # Remove unrequested variables.
    ds = ds[list(post_processors.keys())]

    for var in ds.data_vars:
        ds[var].attrs = {}

    ds = ds.sortby("time")

    # Save final netcdf.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="invalid value encountered in true_divide"
        )
        warnings.filterwarnings(
            "ignore", message="divide by zero encountered in true_divide"
        )
        ds = save_ds(ds, fp, chunks="auto", encoding="initiate", label="Merging files.")

    # Remove intermediate files.
    for x in dss:
        remove_ds(x)

    return ds


def most_recent(product_name, latlim, lonlim):
    extra_search_kwargs = {
        "sortby": [{"field": "properties.datetime", "direction": "desc"}]
    }

    _, query = search_stac(
        latlim, lonlim, None, product_name, extra_search_kwargs, limit=30
    )

    return pd.Timestamp(query["features"][0]["properties"]["datetime"]).to_pydatetime()


def download(
    folder,
    latlim,
    lonlim,
    timelim,
    product_name,
    req_vars,
    variables=None,
    post_processors=None,
    extra_search_kwargs={"eo:cloud_cover": {"gte": 0, "lt": 30}},
    max_attempts=24,
    wait_time=300,
):
    """Order, Download and preprocess Landsat scenes.

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
        Metadata on which exact layers need to be requested from the server, by default None.
    post_processors : dict, optional
        Functions per variable that should be applied to the variable, by default None.
    extra_search_kwargs : dict, optional
        Extra keywords passed to the scene searching API, by default {'eo:cloud_cover': {'gte': 0, 'lt': 30}}
    max_attempts : int, optional
        Maximum number of retries, by default 24.
    wait_time : int, optional
        Wait time in seconds between retries, by default 300.

    Returns
    -------
    xr.Dataset
        Dataset with the requested variables.

    Raises
    ------
    ValueError
        Raised when not all found scenes could be downloaded within the max_attempts.
    """
    adjust_logger(True, folder, "INFO")

    product_folder = os.path.join(folder, "LANDSAT")
    order_folder = os.path.join(product_folder, "orders")

    if not os.path.exists(product_folder):
        os.makedirs(product_folder)

    if not os.path.exists(order_folder):
        os.makedirs(order_folder)

    fn = os.path.join(product_folder, f"{product_name}.nc")
    req_vars_orig = copy.deepcopy(req_vars)
    if os.path.isfile(fn):
        existing_ds = open_ds(fn)
        req_vars_new = list(set(req_vars).difference(set(existing_ds.data_vars)))
        if len(req_vars_new) > 0:
            req_vars = req_vars_new
            existing_ds = existing_ds.close()
        else:
            return existing_ds[req_vars_orig]

    if isinstance(variables, type(None)):
        variables = default_vars(product_name, req_vars)

    if isinstance(post_processors, type(None)):
        post_processors = default_post_processors(product_name, req_vars)
    else:
        default_processors = default_post_processors(product_name, req_vars)
        post_processors = {
            k: {True: default_processors[k], False: v}[v == "default"]
            for k, v in post_processors.items()
            if k in req_vars
        }

    # Search scene IDs (STAC)
    scene_ids, _ = search_stac(
        latlim, lonlim, timelim, product_name, extra_search_kwargs
    )

    if len(scene_ids) == 0:
        return None

    # Order and download scenes (ESPA)
    available_scenes, to_download, to_request, to_wait = download_scenes(
        scene_ids,
        product_folder,
        product_name,
        latlim,
        lonlim,
        max_attempts=max_attempts,
        wait_time=wait_time,
    )
    # log.info(f"available_scenes = {available_scenes}")

    # log.info(f"available_scenes: {len(available_scenes)}, scene_ids: {len(scene_ids)}, to_wait: {len(to_wait)}, to_download: {len(to_download)}, to_request: {len(to_request)} \n\n {available_scenes} \n {scene_ids} \n {to_wait}")
    if len(available_scenes) < len(scene_ids) and len(to_wait) > 0:
        raise ValueError(
            f"Waiting for order of {len(to_wait)} scenes to finish.", len(to_wait)
        )

    # Process scenes.
    scene_paths = [
        glob.glob(os.path.join(product_folder, "**", f"*{x}.nc"), recursive=True)[0]
        for x in available_scenes
    ]
    # log.info(f"scene_paths = {scene_paths}")
    ds = process_scenes(
        fn, scene_paths, product_folder, product_name, variables, post_processors
    )

    # Remove unpacked zips.
    rmve = {"NO": False, "YES": True}.get(os.environ.get("PYWAPOR_REMOVE_TEMP_FILES", "YES"), True)
    if rmve:
        subfolders = [x for x in glob.glob(os.path.join(product_folder, "*.tar")) if os.path.isdir(x)]
        for subfolder in subfolders:
            if os.path.isdir(subfolder):
                try:
                    shutil.rmtree(subfolder)
                except PermissionError:
                    ... # Windows...
    

    return ds[req_vars_orig]


if __name__ == "__main__":
    sources = "level_3"
    period = 5
    area = "fayoum"

    lonlim, latlim = {
        "fayoum": ([30.2, 31.2], [28.9, 29.7]),
        "pakistan_south": ([67.70, 67.90], [26.35, 26.55]),
        "pakistan_hydera": ([68.35, 68.71], [25.49, 25.73]),
    }[area]

    timelim = {
        0: [datetime.date(2019, 10, 1), datetime.date(2019, 10, 11)],
        1: [datetime.date(2022, 5, 1), datetime.date(2022, 5, 11)],
        2: [datetime.date(2022, 10, 1), datetime.date(2022, 10, 11)],
        3: [datetime.date(2022, 8, 1), datetime.date(2022, 10, 1)],
        4: [datetime.date(2021, 8, 1), datetime.date(2021, 10, 1)],
        5: [datetime.date(2015, 5, 15), datetime.date(2015, 7, 21)],
    }[period]

    folder = f"/Users/hmcoerver/Local/{area}_{sources}_{period}"  #

    adjust_logger(True, folder, "INFO")

    product_name = "LC08_SR"
    req_vars = ["ndvi"]
    variables = None
    post_processors = None
    extra_search_kwargs = {"eo:cloud_cover": {"gte": 0, "lt": 30}}
    max_attempts = 24
    wait_time = 300

    # for product_name in ["LC08_SR", "LT05_SR", "LE07_SR"]:
    #     ds = download(
    #         folder,
    #         latlim,
    #         lonlim,
    #         timelim,
    #         product_name,
    #         req_vars,
    #         variables=variables,
    #         post_processors=post_processors,
    #         extra_search_kwargs=extra_search_kwargs,
    #     )
