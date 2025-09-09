import copy
import glob
import os
import warnings
from datetime import datetime as dt
from functools import partial

import numpy as np
import xarray as xr
from lxml import etree

import pywapor.collect.protocol.copernicus_odata as copernicus_odata
from pywapor.enhancers.gap_fill import gap_fill
from pywapor.general.logger import adjust_logger, log
from pywapor.general.processing_functions import adjust_timelim_dtype, open_ds


def apply_qa(ds, var):
    """Mask SENTINEL2 data using a qa variable.

    Parameters
    ----------
    ds : xr.Dataset
        Input data, should contain `"qa"` and `var` as variables.
    var : str
        Variable name in `ds` to be masked.

    Returns
    -------
    xr.Dataset
        Masked dataset.
    """
    # 0 SC_NODATA # 1 SC_SATURATED_DEFECTIVE # 2 SC_DARK_FEATURE_SHADOW
    # 3 SC_CLOUD_SHADOW # 4 SC_VEGETATION # 5 SC_NOT_VEGETATED
    # 6 SC_WATER # 7 SC_UNCLASSIFIED # 8 SC_CLOUD_MEDIUM_PROBA
    # 9 SC_CLOUD_HIGH_PROBA # 10 SC_THIN_CIRRUS # 11 SC_SNOW_ICE
    if "qa" in ds.data_vars:
        pixel_qa_flags = [0, 1, 2, 3, 7, 8, 9, 10, 11]
        keep = np.invert(ds["qa"].isin(pixel_qa_flags))
        ds[var] = ds[var].where(keep)
    else:
        log.warning(
            f"--> Couldn't apply qa, since `qa` doesn't exist in this dataset ({list(ds.data_vars)})."
        )
    return ds


def mask_invalid(ds, var, valid_range=(1, 65534)):
    """Mask invalid data.

    Parameters
    ----------
    ds : xr.Dataset
        Input data.
    var : str
        Variable to mask.
    valid_range : tuple, optional
        Range of valid values in `var`, by default (1, 65534).

    Returns
    -------
    xr.Dataset
        Masked data.
    """
    # 0 = NODATA, 65535 = SATURATED
    ds[var] = ds[var].where((ds[var] >= valid_range[0]) & (ds[var] <= valid_range[1]))
    return ds


def scale_data(ds, var):
    """Apply a scale and offset factor to `ds`.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    var : str
        Variable to scale and offset.

    Returns
    -------
    xr.Dataset
        Output data.
    """

    scale = 1.0 / ds.scale_factor  # BOA_QUANTIFICATION_VALUE
    offset = ds.offset_factor  # BOA_ADD_OFFSET
    ds[var] = (ds[var] + offset) * scale
    ds[var] = ds[var].where((ds[var] <= 1.00) & (ds[var] >= 0.00))
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
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            da = (ds[bands[0]] - ds[bands[1]]) / (ds[bands[0]] + ds[bands[1]])
        ds[var] = da.clip(-1, 1)
    else:
        log.warning(
            f"--> Couldn't calculate `{var}`, `{'` and `'.join([x for x in bands if x not in ds.data_vars])}` is missing."
        )
    return ds


def calc_psri(ds, var):
    """Calculate the PSRI as ("red" - "blue)/"red_edge_740".

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
    reqs = ["red", "blue", "red_edge_740"]
    if np.all([x in ds.data_vars for x in reqs]):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            da = (ds["red"] - ds["blue"]) / ds["red_edge_740"]
        ds[var] = da.clip(-1, 1)
    else:
        log.warning(
            f"--> Couldn't calculate `{var}`, `{'` and `'.join([x for x in reqs if x not in ds.data_vars])}` is missing."
        )
    return ds


def calc_nmdi(ds, var):
    """Calculate the NMDI.

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
    reqs = ["swir1", "swir2", "nir"]
    if np.all([x in ds.data_vars for x in reqs]):
        ds["nominator"] = ds["swir1"] - ds["swir2"]
        ds = calc_normalized_difference(ds, var, bands=["nominator", "nir"])
        ds = ds.drop_vars(["nominator"])
    else:
        log.warning(
            f"--> Couldn't calculate `{var}`, `{'` and `'.join([x for x in reqs if x not in ds.data_vars])}` is missing."
        )
    return ds


def calc_bsi(ds, var):
    """Calculate the BSI.

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
    reqs = ["nir", "swir1", "red", "blue"]
    if np.all([x in ds.data_vars for x in reqs]):
        ds["nominator"] = ds["nir"] + ds["blue"]
        ds["denominator"] = ds["swir1"] + ds["red"]
        ds = calc_normalized_difference(ds, var, bands=["nominator", "denominator"])
        ds = ds.drop_vars(["nominator", "denominator"])
    else:
        log.warning(
            f"--> Couldn't calculate `{var}`, `{'` and `'.join([x for x in reqs if x not in ds.data_vars])}` is missing."
        )
    return ds


def calc_vari_red_egde(ds, var):
    """Calculate the VARI_RED_EDGE.

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
    reqs = ["red_edge_740", "blue", "red"]
    if np.all([x in ds.data_vars for x in reqs]):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            n1 = ds["red_edge_740"] - 1.7 * ds["red"] + 0.7 * ds["blue"]
            n2 = ds["red_edge_740"] + 2.3 * ds["red"] - 1.3 * ds["blue"]
            da = n1 / n2
        ds[var] = da.clip(-1, 1)
    else:
        log.warning(
            f"--> Couldn't calculate `{var}`, `{'` and `'.join([x for x in reqs if x not in ds.data_vars])}` is missing."
        )
    return ds


def calc_r0(ds, var):
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
        "blue": 0.171,
        "green": 0.060,
        "red": 0.334,
        "nir": 0.371,
        "offset": 0.018,
    }

    reqs = ["blue", "green", "red", "nir"]
    if np.all([x in ds.data_vars for x in reqs]):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
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


def default_vars(product_name, req_vars):
    """Given a `product_name` and a list of requested variables, returns a dictionary
    with metadata on which exact layers need to be requested from the server, how they should
    be renamed, and how their dimensions are defined.

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
        "S2MSI2A_R10m": {
            "_B02_10m.jp2": [(), "blue", [mask_invalid, apply_qa, scale_data]],
            "_B03_10m.jp2": [(), "green", [mask_invalid, apply_qa, scale_data]],
            "_B04_10m.jp2": [(), "red", [mask_invalid, apply_qa, scale_data]],
            "_B08_10m.jp2": [(), "nir", [mask_invalid, apply_qa, scale_data]],
            "_SCL_20m.jp2": [(), "qa", []],
        },
        "S2MSI2A_R20m": {
            "_B01_20m.jp2": [
                (),
                "coastal_aerosol",
                [mask_invalid, apply_qa, scale_data],
            ],
            "_B02_20m.jp2": [(), "blue", [mask_invalid, apply_qa, scale_data]],
            "_B03_20m.jp2": [(), "green", [mask_invalid, apply_qa, scale_data]],
            "_B04_20m.jp2": [(), "red", [mask_invalid, apply_qa, scale_data]],
            "_B05_20m.jp2": [(), "red_edge_703", [mask_invalid, apply_qa, scale_data]],
            "_B06_20m.jp2": [(), "red_edge_740", [mask_invalid, apply_qa, scale_data]],
            "_B07_20m.jp2": [(), "red_edge_782", [mask_invalid, apply_qa, scale_data]],
            "_B8A_20m.jp2": [(), "nir", [mask_invalid, apply_qa, scale_data]],
            "_B11_20m.jp2": [(), "swir1", [mask_invalid, apply_qa, scale_data]],
            "_B12_20m.jp2": [(), "swir2", [mask_invalid, apply_qa, scale_data]],
            "_SCL_20m.jp2": [(), "qa", []],
        },
        "S2MSI2A_R60m": {
            "_B01_60m.jp2": [
                (),
                "coastal_aerosol",
                [mask_invalid, apply_qa, scale_data],
            ],
            "_B02_60m.jp2": [(), "blue", [mask_invalid, apply_qa, scale_data]],
            "_B03_60m.jp2": [(), "green", [mask_invalid, apply_qa, scale_data]],
            "_B04_60m.jp2": [(), "red", [mask_invalid, apply_qa, scale_data]],
            "_B05_60m.jp2": [(), "red_edge_703", [mask_invalid, apply_qa, scale_data]],
            "_B06_60m.jp2": [(), "red_edge_740", [mask_invalid, apply_qa, scale_data]],
            "_B07_60m.jp2": [(), "red_edge_782", [mask_invalid, apply_qa, scale_data]],
            "_B8A_60m.jp2": [(), "nir", [mask_invalid, apply_qa, scale_data]],
            "_B09_60m.jp2": [(), "narrow_nir", [mask_invalid, apply_qa, scale_data]],
            "_B11_60m.jp2": [(), "swir1", [mask_invalid, apply_qa, scale_data]],
            "_B12_60m.jp2": [(), "swir2", [mask_invalid, apply_qa, scale_data]],
            "_SCL_60m.jp2": [(), "qa", []],
        },
    }

    req_dl_vars = {
        "S2MSI2A_R10m": {
            "blue": ["_B02_10m.jp2", "_SCL_20m.jp2"],
            "green": ["_B03_10m.jp2", "_SCL_20m.jp2"],
            "red": ["_B04_10m.jp2", "_SCL_20m.jp2"],
            "nir": ["_B08_10m.jp2", "_SCL_20m.jp2"],
            "qa": ["_SCL_20m.jp2"],
            "ndvi": ["_B04_10m.jp2", "_B08_10m.jp2", "_SCL_20m.jp2"],
            "r0": [
                "_B02_10m.jp2",
                "_B03_10m.jp2",
                "_B04_10m.jp2",
                "_B08_10m.jp2",
                "_SCL_20m.jp2",
            ],
        },
        "S2MSI2A_R20m": {
            "coastal_aerosol": ["_B01_20m.jp2", "_SCL_20m.jp2"],
            "blue": ["_B02_20m.jp2", "_SCL_20m.jp2"],
            "green": ["_B03_20m.jp2", "_SCL_20m.jp2"],
            "red": ["_B04_20m.jp2", "_SCL_20m.jp2"],
            "red_edge_703": ["_B05_20m.jp2", "_SCL_20m.jp2"],
            "red_edge_740": ["_B06_20m.jp2", "_SCL_20m.jp2"],
            "red_edge_782": ["_B07_20m.jp2", "_SCL_20m.jp2"],
            "nir": ["_B8A_20m.jp2", "_SCL_20m.jp2"],
            "swir1": ["_B11_20m.jp2", "_SCL_20m.jp2"],
            "swir2": ["_B12_20m.jp2", "_SCL_20m.jp2"],
            "qa": ["_SCL_20m.jp2"],
            "ndvi": ["_B04_20m.jp2", "_B8A_20m.jp2", "_SCL_20m.jp2"],
            "mndwi": ["_B03_20m.jp2", "_B11_20m.jp2", "_SCL_20m.jp2"],
            "vari_red_edge": [
                "_B06_20m.jp2",
                "_B02_20m.jp2",
                "_B04_20m.jp2",
                "_SCL_20m.jp2",
            ],
            "nmdi": ["_B11_20m.jp2", "_B12_20m.jp2", "_B8A_20m.jp2", "_SCL_20m.jp2"],
            "psri": ["_B02_20m.jp2", "_B04_20m.jp2", "_B06_20m.jp2", "_SCL_20m.jp2"],
            "bsi": [
                "_B8A_20m.jp2",
                "_B11_20m.jp2",
                "_B04_20m.jp2",
                "_B02_20m.jp2",
                "_SCL_20m.jp2",
            ],
            "r0": [
                "_B02_20m.jp2",
                "_B03_20m.jp2",
                "_B04_20m.jp2",
                "_B8A_20m.jp2",
                "_SCL_20m.jp2",
            ],
        },
        "S2MSI2A_R60m": {
            "coastal_aerosol": ["_B01_60m.jp2", "_SCL_60m.jp2"],
            "blue": ["_B02_60m.jp2", "_SCL_60m.jp2"],
            "green": ["_B03_60m.jp2", "_SCL_60m.jp2"],
            "red": ["_B04_60m.jp2", "_SCL_60m.jp2"],
            "red_edge_703": ["_B05_60m.jp2", "_SCL_60m.jp2"],
            "red_edge_740": ["_B06_60m.jp2", "_SCL_60m.jp2"],
            "red_edge_782": ["_B07_60m.jp2", "_SCL_60m.jp2"],
            "nir": ["_B8A_60m.jp2", "_SCL_60m.jp2"],
            "narrow_nir": ["_B09_60m.jp2", "_SCL_60m.jp2"],
            "swir1": ["_B11_60m.jp2", "_SCL_60m.jp2"],
            "swir2": ["_B12_60m.jp2", "_SCL_60m.jp2"],
            "qa": ["_SCL_60m.jp2"],
            "ndvi": ["_B04_60m.jp2", "_B8A_60m.jp2", "_SCL_60m.jp2"],
            "mndwi": ["_B03_60m.jp2", "_B11_60m.jp2", "_SCL_60m.jp2"],
            "vari_red_edge": [
                "_B06_60m.jp2",
                "_B02_60m.jp2",
                "_B04_60m.jp2",
                "_SCL_60m.jp2",
            ],
            "nmdi": ["_B11_60m.jp2", "_B12_60m.jp2", "_B8A_60m.jp2", "_SCL_60m.jp2"],
            "psri": ["_B02_60m.jp2", "_B04_60m.jp2", "_B06_60m.jp2", "_SCL_60m.jp2"],
            "bsi": [
                "_B8A_60m.jp2",
                "_B11_60m.jp2",
                "_B04_60m.jp2",
                "_B02_60m.jp2",
                "_SCL_60m.jp2",
            ],
            "r0": [
                "_B02_60m.jp2",
                "_B03_60m.jp2",
                "_B04_60m.jp2",
                "_B8A_60m.jp2",
                "_SCL_60m.jp2",
            ],
        },
    }

    out = {
        val: variables[product_name][val]
        for sublist in map(req_dl_vars[product_name].get, req_vars)
        for val in sublist
    }

    return out


def default_post_processors(product_name, req_vars):
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
        "S2MSI2A_R10m": {
            "blue": [gap_fill],
            "green": [gap_fill],
            "red": [gap_fill],
            "nir": [gap_fill],
            "qa": [],
            "ndvi": [calc_normalized_difference, gap_fill],
            "r0": [calc_r0, gap_fill],
        },
        "S2MSI2A_R20m": {
            "coastal_aerosol": [gap_fill],
            "blue": [gap_fill],
            "green": [gap_fill],
            "red": [gap_fill],
            "red_edge_703": [gap_fill],
            "red_edge_740": [gap_fill],
            "red_edge_782": [gap_fill],
            "nir": [gap_fill],
            "qa": [],
            "swir1": [gap_fill],
            "swir2": [gap_fill],
            "psri": [calc_psri],
            "ndvi": [calc_normalized_difference, gap_fill],
            "nmdi": [calc_nmdi, gap_fill],
            "vari_red_edge": [calc_vari_red_egde, gap_fill],
            "bsi": [calc_bsi, gap_fill],
            "mndwi": [
                partial(calc_normalized_difference, bands=["swir1", "green"]),
                gap_fill,
            ],
            "r0": [calc_r0, gap_fill],
        },
        "S2MSI2A_R60m": {
            "coastal_aerosol": [gap_fill],
            "blue": [gap_fill],
            "green": [gap_fill],
            "red": [gap_fill],
            "red_edge_703": [gap_fill],
            "red_edge_740": [gap_fill],
            "red_edge_782": [gap_fill],
            "nir": [gap_fill],
            "narrow_nir": [gap_fill],
            "qa": [],
            "swir1": [gap_fill],
            "swir2": [gap_fill],
            "psri": [calc_psri, gap_fill],
            "ndvi": [calc_normalized_difference, gap_fill],
            "nmdi": [calc_nmdi, gap_fill],
            "vari_red_edge": [calc_vari_red_egde, gap_fill],
            "bsi": [calc_bsi, gap_fill],
            "mndwi": [
                partial(calc_normalized_difference, bands=["swir1", "green"]),
                gap_fill,
            ],
            "r0": [calc_r0, gap_fill],
        },
    }

    out = {k: v for k, v in post_processors[product_name].items() if k in req_vars}

    return out


def time_func(fn):
    """Return a np.datetime64 given a filename.

    Parameters
    ----------
    fn : str
        Filename.

    Returns
    -------
    np.datetime64
        Date as described in the filename.
    """
    dtime = np.datetime64(dt.strptime(fn.split("_")[2], "%Y%m%dT%H%M%S"), "ns")
    return dtime


def s2_processor(scene_folder, variables, **kwargs):
    has_10m = np.any(["_10m" in x for x in variables.keys()])

    dss = list()
    dss__ = list()
    lowres_vars_idxs = list()
    for i, (k, v) in enumerate(variables.items()):
        ds__ = open_ds(
            glob.glob(os.path.join(scene_folder, "**", "*" + k), recursive=True)[0],
            decode_coords=None,
        )
        ds_ = ds__.isel(band=0).rename({"band_data": v[1]})
        dss__.append(ds__)
        dss.append(ds_)
        if has_10m and "_20m" in k:
            lowres_vars_idxs.append(i)

    if len(lowres_vars_idxs) < len(dss) and has_10m:
        highres_ds = dss[[i for i in range(len(dss)) if i not in lowres_vars_idxs][0]]
        for idx in lowres_vars_idxs:
            dss[idx] = dss[idx].interp_like(
                highres_ds, method="nearest", kwargs={"fill_value": "extrapolate"}
            )

    ds = xr.merge(dss).drop_vars("band")

    meta_fps = glob.glob(
        os.path.join(scene_folder, "**", "MTD_MSIL2A.xml"), recursive=True
    )

    if len(meta_fps) >= 1:
        tree = etree.parse(meta_fps[0])
        root = tree.getroot()
        baseline = [float(x.text) for x in root.iter("PROCESSING_BASELINE")][0]
        # NOTE https://sentinels.copernicus.eu/en/web/sentinel/-/copernicus-sentinel-2-major-products-upgrade-upcoming
        # NOTE https://stackoverflow.com/questions/72566760/how-to-correct-sentinel-2-baseline-v0400-offset
        if baseline >= 4.0:
            offsets = [float(x.text) for x in root.iter("BOA_ADD_OFFSET")]
            scales = [float(x.text) for x in root.iter("BOA_QUANTIFICATION_VALUE")]
            scale = np.median(scales)
            offset = np.median(offsets)
            ds.attrs = {"scale_factor": scale, "offset_factor": offset}
        else:
            offsets = [0.0]
            offset = offsets[0]
            scales = [float(x.text) for x in root.iter("BOA_QUANTIFICATION_VALUE")]
            scale = np.median(scales)
            ds.attrs = {"scale_factor": scale, "offset_factor": offset}
        if len(np.unique(offsets)) != 1:
            log.warning(
                f"--> Multiple offsets found for `{scene_folder}`, using `{offset}`, check `{meta_fps[0]}` for more info."
            )
        if len(scales) != 1:
            log.warning(
                f"--> Multiple scales found for `{scene_folder}`, using `{scale}`, check `{meta_fps[0]}` for more info."
            )
    else:
        t = time_func(os.path.split(scene_folder)[-1])
        if t < np.datetime64("2022-01-25"):
            ds.attrs = {"scale_factor": 10000.0, "offset_factor": 0.0}
        else:
            ds.attrs = {"scale_factor": 10000.0, "offset_factor": -1000.0}
        log.warning(
            f"--> No scale/offset found for `{scene_folder}`, using `{ds.attrs}`."
        )

    return ds, dss__


def most_recent(product_name, latlim, lonlim):
    return copernicus_odata.most_recent(product_name, latlim, lonlim, "SENTINEL-2")


def download(
    folder,
    latlim,
    lonlim,
    timelim,
    product_name,
    req_vars,
    variables=None,
    post_processors=None,
    extra_search_kwargs={"cloudcoverpercentage": (0, 30)},
):
    """Download SENTINEL2 data and store it in a single netCDF file.

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
    extra_search_kwargs : dict
        Extra search kwargs passed to SentinelAPI, by default {"cloudcoverpercentage": (0, 30)}.

    Returns
    -------
    xr.Dataset
        Downloaded data.
    """

    product_folder = os.path.join(folder, "SENTINEL2")

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

    timelim = adjust_timelim_dtype(timelim)
    bb = [lonlim[0], latlim[0], lonlim[1], latlim[1]]

    def node_filter(node_info):
        fn = os.path.split(node_info)[-1]
        to_dl = list(variables.keys()) + ["MTD_MSIL2A.xml"]
        return np.any([x in fn for x in to_dl])

    final_fn = os.path.split(fn)[-1]
    processor = s2_processor
    scenes = copernicus_odata.download(
        product_folder,
        latlim,
        lonlim,
        timelim,
        "SENTINEL2",
        product_name.split("_")[0],
        node_filter=node_filter,
    )
    ds = copernicus_odata.process_sentinel(
        scenes, variables, time_func, final_fn, post_processors, processor, bb=bb
    )

    return ds[req_vars_orig]


if __name__ == "__main__":
    os.environ["PYWAPOR_REMOVE_TEMP_FILES"] = "NO"

    folder = r"/Users/hmcoerver/Local/s2_test"
    adjust_logger(True, folder, "INFO")
    timelim = ["2022-03-29", "2022-03-31"]
    latlim = [28.9, 29.1]
    lonlim = [30.2, 30.4]

    product_name = "S2MSI2A_R10m"
    # req_vars = ["mndwi", "psri", "vari_red_edge", "bsi", "nmdi", "green", "nir"]
    req_vars = ["ndvi", "r0"]
    post_processors = None
    variables = None

    # ds = download(folder, latlim, lonlim, timelim, product_name,
    #             req_vars, variables = variables, post_processors = post_processors)

    # variables = {'_B02_10m.jp2': [(), 'blue', []], '_SCL_20m.jp2': [(), 'qa', []]}

    # scene_folder = r"/Users/hmcoerver/Local/s2_test/SENTINEL2/S2A_MSIL2A_20220329T082601_N0400_R021_T36RTS_20220329T125458.SAFE"
