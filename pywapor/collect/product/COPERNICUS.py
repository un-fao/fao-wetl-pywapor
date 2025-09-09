import copy
import os
import glob
from functools import partial
from itertools import product

import numpy as np
import xarray as xr

import pywapor
from pywapor.collect.protocol.crawler import download_urls
from pywapor.enhancers.apply_enhancers import apply_enhancers
from pywapor.enhancers.dem import calc_slope_or_aspect
from pywapor.general.logger import adjust_logger
from pywapor.general.processing_functions import open_ds, remove_ds, save_ds


def tiles_intersect(latlim, lonlim, product_name):
    """Creates a list of server-side filenames for tiles that intersect with `latlim` and
    `lonlim` for the selected product.

    Parameters
    ----------
    latlim : list
        Latitude limits of area of interest.
    lonlim : list
        Longitude limits of area of interest.
    product_name : str
        Name of the product to download.

    Returns
    -------
    list
        Server-side filenames for tiles.
    """
    with open(
        os.path.join(pywapor.collect.__path__[0], f"product/{product_name}.txt")
    ) as src:
        all_tiles = src.read().split("\n")
    we_tiles = (
        np.arange(np.floor(lonlim[0]), np.ceil(lonlim[1]), 1).astype(int).tolist()
    )
    ns_tiles = (
        np.arange(np.floor(latlim[0]), np.ceil(latlim[1]), 1).astype(int).tolist()
    )
    tiles = list(product(we_tiles, ns_tiles))
    dl_tiles = list()
    res = {"GLO30": 10, "GLO90": 30}[product_name]
    for tile in tiles:
        we_sign = {-1: "W", 1: "E", 0: "E"}[np.sign(tile[0])]
        ns_sign = {-1: "S", 1: "N", 0: "N"}[np.sign(tile[1])]
        fn = f"Copernicus_DSM_COG_{res}_{ns_sign}{abs(tile[1]):02}_00_{we_sign}{abs(tile[0]):03}_00_DEM"
        if fn in all_tiles:
            dl_tiles.append(fn)
    return dl_tiles


def url_func(product_name, fn):
    """Returns a url at which to collect COPERNICUS data.

    Parameters
    ----------
    product_name : str
        Name of the product to download.
    fn : str
        Name of the server-side filename to download.

    Returns
    -------
    str
        The url.
    """
    url = {
        "GLO30": f"https://copernicus-dem-30m.s3.amazonaws.com/{fn}/{fn}.tif",
        "GLO90": f"https://copernicus-dem-90m.s3.amazonaws.com/{fn}/{fn}.tif",
    }[product_name]
    return url


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
        "GLO30": {
            "Band1": [("lat", "lon"), "z"],
            "crs": [(), "spatial_ref"],
        },
        "GLO90": {
            "Band1": [("lat", "lon"), "z"],
            "crs": [(), "spatial_ref"],
        },
    }

    req_dl_vars = {
        "GLO30": {
            "z": ["Band1", "crs"],
            "slope": ["Band1", "crs"],
            "aspect": ["Band1", "crs"],
        },
        "GLO90": {
            "z": ["Band1", "crs"],
            "slope": ["Band1", "crs"],
            "aspect": ["Band1", "crs"],
        },
    }

    out = {
        val: variables[product_name][val]
        for sublist in map(req_dl_vars[product_name].get, req_vars)
        for val in sublist
    }

    return out


def default_post_processors(product_name, req_vars=["z"]):
    """Given a `product_name` and a list of requested variables, returns a dictionary with a
    list of functions per variable that should be applied after having collected the data
    from a server.

    Parameters
    ----------
    product_name : str
        Name of the product.
    req_vars : list, optional
        List of variables to be collected, by default ["z"].

    Returns
    -------
    dict
        Functions per variable that should be applied to the variable.
    """

    post_processors = {
        "GLO30": {
            "z": [],
            "slope": [partial(calc_slope_or_aspect, write_init=False)],
            "aspect": [partial(calc_slope_or_aspect, write_init=False)],
        },
        "GLO90": {
            "z": [],
            "slope": [partial(calc_slope_or_aspect, write_init=False)],
            "aspect": [partial(calc_slope_or_aspect, write_init=False)],
        },
    }

    out = {k: v for k, v in post_processors[product_name].items() if k in req_vars}

    return out


def most_recent(product_name, *args):
    if product_name == "GLO30":
        return None
    elif product_name == "GLO90":
        return None


def download(
    folder,
    latlim,
    lonlim,
    product_name="GLO30",
    req_vars=["z"],
    variables=None,
    post_processors=None,
    **kwargs,
):
    """Download COPERNICUS data and store it in a single netCDF file.

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
    product_name : str, optional
        Name of the product to download, by default "GLO30".
    req_vars : list, optional
        Which variables to download for the selected product, by default ["z"].
    variables : dict, optional
        Metadata on which exact layers need to be requested from the server, by default None.
    post_processors : dict, optional
        Functions per variable that should be applied to the variable, by default None.

    Returns
    -------
    xr.Dataset
        Downloaded data.
    """

    folder = os.path.join(folder, "COPERNICUS")

    final_fp = os.path.join(folder, f"{product_name}.nc")
    req_vars_orig = copy.deepcopy(req_vars)
    if os.path.isfile(final_fp):
        existing_ds = open_ds(final_fp)
        req_vars_new = list(set(req_vars).difference(set(existing_ds.data_vars)))
        if len(req_vars_new) > 0:
            req_vars = req_vars_new
            existing_ds = existing_ds.close()
        else:
            return existing_ds[req_vars_orig]

    spatial_buffer = True
    if spatial_buffer:
        dx = dy = 0.00027778
        latlim = [latlim[0] - dy, latlim[1] + dy]
        lonlim = [lonlim[0] - dx, lonlim[1] + dx]

    dl_tiles = tiles_intersect(latlim, lonlim, product_name)

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

    coords = {"x": ("lon", lonlim), "y": ("lat", latlim)}

    urls = [url_func(product_name, x) for x in dl_tiles]
    fns = download_urls(urls, folder)

    dss = [
        xr.open_dataset(x, chunks={"y": -1, "x": -1}).isel(band=0, drop=True)
        for x in fns
    ]
    ds = xr.combine_by_coords(dss).rename({"band_data": "z"})
    ds = ds.rio.clip_box(
        coords["x"][1][0], coords["y"][1][0], coords["x"][1][1], coords["y"][1][1]
    )

    ds__ = save_ds(
        ds,
        final_fp.replace(".nc", "_stitched.nc"),
        encoding="initiate",
        label="Merging tiles.",
    )

    # Apply product specific functions.
    ds_ = apply_enhancers(post_processors, ds__)

    # Remove unrequested variables.
    ds_ = ds_[list(post_processors.keys())]

    # Save final output.
    ds = save_ds(ds_, final_fp, encoding="initiate", label=f"Saving {product_name}.nc")

    for nc in dss:
        remove_ds(nc)

    remove_ds(ds__)

    fhs = glob.glob(os.path.join(folder, "*_aspect.tif")) + \
    glob.glob(os.path.join(folder, "*_slope.tif"))
    for fh in fhs:
        remove_ds(fh)

    return ds[req_vars_orig]


if __name__ == "__main__":
    folder = r"/Users/hmcoerver/Local/cog_test"
    product_name = r"GLO90"  # r"GLO90" r"GLO30
    # latlim = [28.9, 29.7]
    # lonlim = [30.2, 31.2]
    # latlim = [24.2, 30.2]
    # lonlim = [24.8, 33.7]

    lonlim = [-75.25, -75.15]
    latlim = [-13.6, -13.5]

    req_vars = ["z", "slope", "aspect"]
    variables = None
    post_processors = None

    adjust_logger(True, folder, "INFO")

    ds = download(
        folder,
        latlim,
        lonlim,
        product_name=product_name,
        req_vars=req_vars,
        variables=variables,
        post_processors=post_processors,
    )
