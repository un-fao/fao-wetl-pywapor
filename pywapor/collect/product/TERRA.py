"""
https://docs.terrascope.be/#/Developers/WebServices/TerraCatalogue/ProductDownload
"""

import copy
import datetime
import os
import time
from functools import partial

import numpy as np
import requests
import xarray as xr
from joblib import Memory
from osgeo import gdal
from requests.exceptions import ConnectionError

import pywapor.collect.accounts as accounts
import pywapor.collect.protocol.cog as cog
import pywapor.general.bitmasks as bm
from pywapor.enhancers.apply_enhancers import apply_enhancers
from pywapor.general.logger import adjust_logger, log
from pywapor.general.processing_functions import (
    adjust_timelim_dtype,
    is_corrupt_or_empty,
    open_ds,
    process_ds,
    remove_ds,
    save_ds,
)


def default_post_processors(product_name, req_vars=["ndvi", "r0"]):
    """Given a `product_name` and a list of requested variables, returns a dictionary with a
    list of functions per variable that should be applied after having collected the data
    from a server.

    Parameters
    ----------
    product_name : str
        Name of the product.
    req_vars : list, optional
        List of variables to be collected, by default ["ndvi", "r0"].

    Returns
    -------
    dict
        Functions per variable that should be applied to the variable.
    """

    post_processors = {
        "urn:eop:VITO:PROBAV_S5_TOC_100M_COG_V2": {
            "r0": [
                calc_r0,
                partial(
                    mask_bitwise_qa,
                    flags=[
                        "bad BLUE",
                        "bad RED",
                        "bad NIR",
                        "bad SWIR",
                        "sea",
                        "undefined",
                        "cloud",
                        "ice/snow",
                        "shadow",
                    ],
                ),
            ],
            "ndvi": [
                partial(
                    mask_bitwise_qa,
                    flags=[
                        "bad RED",
                        "bad NIR",
                        "sea",
                        "undefined",
                        "cloud",
                        "ice/snow",
                        "shadow",
                    ],
                ),
            ],
        }
    }

    out = {k: v for k, v in post_processors[product_name].items() if k in req_vars}

    return out


def default_vars(product_name, req_vars=["ndvi", "r0"]):
    """Given a `product_name` and a list of requested variables, returns a dictionary
    with metadata on which exact layers need to be requested from the server, how they should
    be renamed, and how their dimensions are defined.

    Parameters
    ----------
    product_name : str
        Name of the product.
    req_vars : list, optional
        List of variables to be collected, by default ["ndvi", "r0"].

    Returns
    -------
    dict
        Metadata on which exact layers need to be requested from the server.
    """

    variables = {
        "urn:eop:VITO:PROBAV_S5_TOC_100M_COG_V2": {
            "NIR": [("lon", "lat"), "nir"],
            "RED": [("lon", "lat"), "red"],
            "BLUE": [("lon", "lat"), "blue"],
            "SWIR": [("lon", "lat"), "swir"],
            "SM": [("lon", "lat"), "qa"],
            "GEOMETRY": [("lon", "lat"), "geometry"],
            "NDVI": [("lon", "lat"), "ndvi"],
            "TIME": [("lon", "lat"), "time"],
        },
    }

    req_dl_vars = {
        "urn:eop:VITO:PROBAV_S5_TOC_100M_COG_V2": {
            "ndvi": ["NDVI", "SM"],
            "r0": ["BLUE", "NIR", "RED", "SWIR", "SM"],
        }
    }

    out = {
        val: variables[product_name][val]
        for sublist in map(req_dl_vars[product_name].get, req_vars)
        for val in sublist
    }

    return out


def mask_bitwise_qa(ds, var, flags):
    """Mask PROBAV data using a qa variable.

    Parameters
    ----------
    ds : xr.Dataset
        Input data.
    var : str
        Variable in `ds` to mask.
    flags : list
        Which flags not to mask.

    Returns
    -------
    xr.Dataset
        Masked dataset.
    """
    flag_bits = bm.PROBAV_qa_translator()
    mask = bm.get_mask(ds["qa"].astype("uint8"), flags, flag_bits)
    ds[var] = ds[var].where(~mask, np.nan)
    return ds


def calc_r0(ds, *args):
    ds["r0"] = (
        0.429 * ds["blue"] + 0.333 * ds["red"] + 0.133 * ds["nir"] + 0.105 * ds["swir"]
    )
    return ds


def most_recent(product_name, latlim, lonlim):
    bb = [lonlim[0], latlim[0], lonlim[1], latlim[1]]
    params = dict()
    params["limit"] = 250
    params["bbox"] = bb
    params["collections"] = [product_name]
    products = search_stac(params, None)
    return datetime.datetime.strptime(
        products[-1]["properties"]["datetime"], "%Y-%m-%dT%H:%M:%SZ"
    )


def download(
    folder,
    latlim,
    lonlim,
    timelim,
    product_name,
    req_vars=["ndvi", "r0"],
    variables=None,
    post_processors=None,
    timedelta=np.timedelta64(60, "h"),
):
    """Download MODIS data and store it in a single netCDF file.

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
    req_vars : list, optional
        Which variables to download for the selected product, by default ["ndvi", "r0"].
    variables : dict, optional
        Metadata on which exact layers need to be requested from the server, by default None.
    post_processors : dict, optional
        Functions per variable that should be applied to the variable, by default None.

    Returns
    -------
    xr.Dataset
        Downloaded data.
    """
    folder = os.path.join(folder, "TERRA")

    if not os.path.isdir(folder):
        os.makedirs(folder)

    fn = os.path.join(folder, f"{product_name.replace(':', '_')}.nc")
    req_vars_orig = copy.deepcopy(req_vars)
    if os.path.isfile(fn):
        existing_ds = open_ds(fn)
        req_vars_new = list(set(req_vars).difference(set(existing_ds.data_vars)))
        if len(req_vars_new) > 0:
            req_vars = req_vars_new
            existing_ds = existing_ds.close()
        else:
            return existing_ds[req_vars_orig]

    spatial_buffer = True
    if spatial_buffer:
        dx = dy = 0.0033
        latlim = [latlim[0] - dy, latlim[1] + dy]
        lonlim = [lonlim[0] - dx, lonlim[1] + dx]

    timelim = adjust_timelim_dtype(timelim)

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

    bb = [lonlim[0], latlim[0], lonlim[1], latlim[1]]

    coords = {"x": ["lon", lonlim], "y": ["lat", latlim]}

    sd = datetime.datetime.strftime(timelim[0], "%Y-%m-%dT00:00:00Z")
    ed = datetime.datetime.strftime(timelim[1], "%Y-%m-%dT23:59:59Z")
    search_dates = f"{sd}/{ed}"
    params = dict()
    params["limit"] = 250
    params["bbox"] = bb
    params["datetime"] = search_dates
    params["collections"] = [product_name]

    # NOTE paths on windows have a max length, this extends the max length, see
    # here for more info https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=registry
    if os.name == "nt":
        cachedir = "\\\\?\\" + os.path.join(os.path.abspath(folder), "cache")
    else:
        cachedir = os.path.join(folder, "cache")

    products = search_stac(params, cachedir)

    dss = list()

    log.info(f"--> Processing {len(products)} scenes.").add()

    un, pw = accounts.get("TERRA")

    gdal_config_options = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    }

    warp_kwargs = dict()
    warp_kwargs["outputBounds"] = [lonlim[0], latlim[0], lonlim[1], latlim[1]]
    warp_kwargs["outputBoundsSRS"] = "epsg:4326"

    outs = list()

    for i, product in enumerate(products):
        out_fp = os.path.join(folder, f"{product['properties']['title']}.nc")

        log.info(
            f"--> ({i + 1}/{len(products)}) Processing `{product['properties']['title']}`."
        ).add()

        if os.path.isfile(out_fp):
            corrupt = is_corrupt_or_empty(out_fp)
            if corrupt:
                remove_ds(out_fp)
            else:
                out = open_ds(out_fp)
                outs.append(out)
        else:
            all_cog_urls = [v.get("href") for k, v in product["assets"].items()]
            cog_urls = [
                (key, all_cog_urls[np.argmax([key in x for x in all_cog_urls])])
                for key in variables.keys()
            ]
            dl_urls = [
                (
                    x[0],
                    x[1].replace(
                        "https://services", f"/vsicurl/https://{un}:{pw}@services"
                    ),
                )
                for x in cog_urls
            ]

            dss = list()
            cleanup = list()
            for j, (var, url) in enumerate(dl_urls):
                var_file = out_fp.replace(".nc", f"_{var}_temp.nc")
                log.info(f"--> ({j + 1}/{len(dl_urls)}) Downloading `{var}`.").add()

                if os.path.isfile(var_file):
                    corrupt = is_corrupt_or_empty(var_file)
                    if corrupt:
                        remove_ds(var_file)
                    else:
                        ds_ = open_ds(var_file)
                        cleanup.append(ds_)
                        ds = ds_.rename({"Band1": var})
                        dss.append(ds)
                else:
                    try:
                        for k, v in gdal_config_options.items():
                            gdal.SetConfigOption(k, v)
                        var_file = cog.cog_dl([url], var_file, warp_kwargs=warp_kwargs)
                    except Exception as e:
                        raise e
                    finally:
                        for k, v in gdal_config_options.items():
                            gdal.SetConfigOption(k, None)
                    ds_ = open_ds(var_file)
                    cleanup.append(ds_)
                    ds = ds_.rename({"Band1": var})
                    dss.append(ds)
                    if var != dl_urls[-1][0]:
                        # NOTE VERY IMPORTANT, otherwise easily hitting "429" errors,
                        # resulting in very strange behaviour.
                        time.sleep(10)
                log.sub()

        ds = xr.merge(dss)
        ds = process_ds(ds, coords, variables)
        ds = ds.expand_dims({"time": 1}).assign_coords(
            {
                "time": [
                    datetime.datetime.strptime(
                        product["properties"]["datetime"], "%Y-%m-%dT%H:%M:%SZ"
                    )
                ]
            }
        )

        # # Apply product specific functions.
        ds = apply_enhancers(post_processors, ds)

        out = save_ds(
            ds,
            out_fp,
            encoding="initiate",
            label=f"Saving {os.path.split(out_fp)[-1]}.",
        )
        outs.append(out)

        for x in cleanup:
            remove_ds(x)

        log.sub()

    log.sub()

    ds = xr.merge(outs)

    if not isinstance(timedelta, type(None)):
        ds["time"] = ds["time"] + timedelta

    ds = ds[req_vars]
    ds = ds.rio.write_crs("epsg:4326")

    ds = save_ds(ds, fn, label="Merging files.")

    for nc in outs:
        remove_ds(nc)

    return ds[req_vars_orig]


def search_stac(params, cachedir):
    memory = Memory(cachedir, verbose=0)

    search = "https://services.terrascope.be/stac/search"

    @memory.cache()
    def _post_query(params):
        all_scenes = list()
        links = [{"body": params, "rel": "next"}]
        while "next" in [x.get("rel", None) for x in links]:
            params_ = [x.get("body") for x in links if x.get("rel") == "next"][0]
            try:
                query = requests.post(search, json=params_)
            except ConnectionError as e:
                log.info(f"--> The TERRA server is not responding (check `{search}`).")
                raise (e)
            query.raise_for_status()
            out = query.json()
            all_scenes += out["features"]
            links = out["links"]
        return all_scenes

    all_scenes = _post_query(params)

    return all_scenes


if __name__ == "__main__":
    variables = None
    post_processors = None

    folder = "/Users/hmcoerver/local/test_dl_TERRA_0"
    latlim = [29.4, 29.7]
    lonlim = [30.7, 31.0]
    timelim = [datetime.date(2020, 7, 2), datetime.date(2020, 7, 9)]
    product_name = "urn:eop:VITO:PROBAV_S5_TOC_100M_COG_V2"
    req_vars = ["r0"]
    timedelta = np.timedelta64(60, "h")

    adjust_logger(True, folder, "INFO")
