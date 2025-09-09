import copy
import datetime
import fnmatch
import os
from functools import partial
from urllib.error import HTTPError

import numpy as np
import pandas as pd

import pywapor.collect.accounts as accounts
import pywapor.collect.protocol.opendap as opendap
from pywapor.collect.protocol.crawler import find_paths
from pywapor.collect.protocol.projections import get_crss
from pywapor.enhancers.pressure import pa_to_kpa
from pywapor.enhancers.temperature import kelvin_to_celsius
from pywapor.general.processing_functions import open_ds


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
        "M2I1NXASM.5.12.4": {
            "T2M": [("time", "lat", "lon"), "t_air"],
            "U2M": [("time", "lat", "lon"), "u2m"],
            "V2M": [("time", "lat", "lon"), "v2m"],
            "QV2M": [("time", "lat", "lon"), "qv"],
            "TQV": [("time", "lat", "lon"), "wv"],
            "PS": [("time", "lat", "lon"), "p_air"],
            "SLP": [("time", "lat", "lon"), "p_air_0"],
        },
        "M2T1NXRAD.5.12.4": {
            "SWGDN": [("time", "lat", "lon"), "ra_flat"],
        },
    }

    req_dl_vars = {
        "M2I1NXASM.5.12.4": {
            "t_air": ["T2M"],
            "t_air_max": ["T2M"],
            "t_air_min": ["T2M"],
            "u2m": ["U2M"],
            "v2m": ["V2M"],
            "qv": ["QV2M"],
            "wv": ["TQV"],
            "p_air": ["PS"],
            "p_air_0": ["SLP"],
        },
        "M2T1NXRAD.5.12.4": {
            "ra_flat": ["SWGDN"],
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
        "M2I1NXASM.5.12.4": {
            "t_air": [kelvin_to_celsius],
            "t_air_max": [
                partial(kelvin_to_celsius, in_var="t_air", out_var="t_air_max")
            ],
            "t_air_min": [
                partial(kelvin_to_celsius, in_var="t_air", out_var="t_air_min")
            ],
            "u2m": [],
            "v2m": [],
            "qv": [],
            "wv": [],
            "p_air": [pa_to_kpa],
            "p_air_0": [pa_to_kpa],
        },
        "M2T1NXRAD.5.12.4": {
            "ra_flat": [],
        },
    }

    out = {k: v for k, v in post_processors[product_name].items() if k in req_vars}

    return out


def fn_func(product_name, tile):
    """Returns a client-side filename at which to store data.

    Parameters
    ----------
    product_name : str
        Name of the product to download.
    tile : str
        Name of the server-side tile to download.

    Returns
    -------
    str
        Filename.
    """
    fn = f"{product_name}_{tile.strftime('%Y%m%d')}.nc"
    return fn


def url_func(product_name, tile):
    """Returns a url at which to collect MERRA2 data.

    Parameters
    ----------
    product_name : str
        Name of the product to download.
    tile : str
        Name of the server-side tile to download.

    Returns
    -------
    str
        The url.
    """

    def _filter(tag):
        tag_value = tag["href"]
        if tag_value[-5:] == ".html":
            tag_value = tag_value[:-5]
        return tag_value

    # Find the existing tiles for the given year and month, this is necessary
    # because the version number (`\d{3}`) in the filenames is irregular.
    regex = r"MERRA2_\d{3}\..*\.\d{8}.nc4.html"
    url = f"https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2/{product_name}/{tile.year}/{tile.month:02d}/contents.html"
    tile_names = find_paths(url, regex, filter=_filter)

    # Find which of the existing tiles matches with the date.
    fn_pattern = f"MERRA2_*.*.{tile.strftime('%Y%m%d')}.nc4"
    fn = fnmatch.filter(tile_names, fn_pattern)[0]

    # Create the final url.
    url = f"https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2/{product_name}/{tile.year}/{tile.month:02d}/{fn}.nc4?"
    return url


def most_recent(product_name, *args):
    today = pd.Timestamp.today().date()
    month_year = today + pd.offsets.MonthBegin()

    def _filter(tag):
        tag_value = tag["href"]
        if tag_value[-5:] == ".html":
            tag_value = tag_value[:-5]
        return tag_value

    regex = r"MERRA2_\d{3}\..*\.\d{8}.nc4.html"

    found = False
    iter = 0
    while not found and iter < 1000:
        url = f"https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2/{product_name}/{month_year.year}/{month_year.month:02d}/contents.html"
        try:
            tile_names = find_paths(url, regex, filter=_filter)
        except HTTPError:
            tile_names = []
        if len(tile_names) == 0:
            month_year -= pd.offsets.MonthBegin()
        else:
            found = True
        iter += 1

    x = datetime.datetime.strptime(sorted(tile_names)[-1].split(".")[-2], "%Y%m%d")
    return x


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
    """Download MERRA2 data and store it in a single netCDF file. See
    https://gmao.gsfc.nasa.gov/pubs/docs/Bosilovich785.pdf for docs.

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

    Returns
    -------
    xr.Dataset
        Downloaded data.
    """

    folder = os.path.join(folder, "MERRA2")

    fn = os.path.join(folder, f"{product_name}.nc")
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
        latlim = [latlim[0] - 0.5, latlim[1] + 0.5]
        lonlim = [lonlim[0] - 0.625, lonlim[1] + 0.625]

    tiles = pd.date_range(timelim[0], timelim[1], freq="D")
    coords = {"x": ["lon", lonlim], "y": ["lat", latlim], "t": ["time", timelim]}
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

    timedelta = np.timedelta64(30, "m")
    data_source_crs = get_crss("WGS84")
    parallel = False
    spatial_tiles = False
    un_pw = accounts.get("NASA")
    request_dims = True

    ds = opendap.download(
        fn,
        product_name,
        coords,
        variables,
        post_processors,
        fn_func,
        url_func,
        un_pw=un_pw,
        tiles=tiles,
        data_source_crs=data_source_crs,
        parallel=parallel,
        spatial_tiles=spatial_tiles,
        request_dims=request_dims,
        timedelta=timedelta,
    )

    return ds[req_vars_orig]


if __name__ == "__main__":
    folder = r"/Users/hmcoerver/Downloads/pywapor_test"
    # latlim = [26.9, 33.7]
    # lonlim = [25.2, 37.2]
    latlim = [28.9, 29.7]
    lonlim = [30.2, 31.2]
    timelim = [datetime.date(2020, 7, 1), datetime.date(2020, 7, 11)]

    variables = None
    post_processors = None

    # # MERRA2.
    # wanted = [
    #             ("M2I1NXASM.5.12.4", ["t_air", "u2m", "v2m", "qv", "wv", "p_air", "p_air_0"]),
    #             # ("M2T1NXRAD.5.12.4", ["ra"]),
    #         ]

    product_name = "M2I1NXASM.5.12.4"
    req_vars = ["t_air", "u2m", "v2m", "qv", "wv", "p_air", "p_air_0"]

    # for product_name, req_vars in wanted:
    # ds = download(folder, latlim, lonlim, timelim, product_name, req_vars)
    # print(ds.rio.crs, ds.rio.grid_mapping)
