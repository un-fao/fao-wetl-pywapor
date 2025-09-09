"""
https://opendap.cr.usgs.gov/opendap/hyrax/SRTMGL1_NUMNC.003/contents.html
"""

import copy
import datetime
import json
import os

from shapely.geometry import shape
from shapely.geometry.polygon import Polygon

import pywapor.collect
import pywapor.collect.accounts as accounts
import pywapor.collect.protocol.opendap as opendap
from pywapor.enhancers.dem import calc_slope_or_aspect
from pywapor.general.processing_functions import open_ds


def tiles_intersect(latlim, lonlim):
    """Creates a list of server-side filenames for tiles that intersect with `latlim` and
    `lonlim` for the selected product.

    Parameters
    ----------
    latlim : list
        Latitude limits of area of interest.
    lonlim : list
        Longitude limits of area of interest.

    Returns
    -------
    list
        Server-side filenames for tiles.
    """
    with open(
        os.path.join(pywapor.collect.__path__[0], "product/SRTM30_tiles.geojson")
    ) as f:
        features = json.load(f)["features"]
    aoi = Polygon.from_bounds(lonlim[0], latlim[0], lonlim[1], latlim[1])
    tiles = list()
    for feature in features:
        shp = shape(feature["geometry"])
        tile = feature["properties"]["dataFile"]
        if shp.intersects(aoi):
            tiles.append(tile.split(".")[0])
    return tiles


def default_vars(product_name, req_vars=["z"]):
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
        "30M": {
            "SRTMGL1_DEM": [("time", "lat", "lon"), "z"],
            "crs": [(), "spatial_ref"],
        }
    }

    req_dl_vars = {
        "30M": {
            "z": ["SRTMGL1_DEM", "crs"],
            "slope": ["SRTMGL1_DEM", "crs"],
            "aspect": ["SRTMGL1_DEM", "crs"],
        }
    }

    out = {
        val: variables[product_name][val]
        for sublist in map(req_dl_vars[product_name].get, req_vars)
        for val in sublist
    }

    return out


def drop_time(ds):
    return ds.isel(time=0).drop("time")


def default_post_processors(product_name, req_vars=["z"]):
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
        "30M": {
            "z": [],
            "aspect": [calc_slope_or_aspect],
            "slope": [calc_slope_or_aspect],
        }
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
    fn = f"{product_name}_{tile}.nc"
    return fn


def url_func(product_name, tile):
    """Returns a url at which to collect MERRA2 data.

    Parameters
    ----------
    product_name : None
        Not used.
    tile : str
        Name of the server-side tile to download.

    Returns
    -------
    str
        The url.
    """
    url = f"https://opendap.cr.usgs.gov/opendap/hyrax/SRTMGL1_NC.003/{tile}.SRTMGL1_NC.ncml.nc4?"
    return url

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
    **kwargs,
):
    """Download SRTM data and store it in a single netCDF file.

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
        Name of the product to download, by default "30M".
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
    folder = os.path.join(folder, "SRTM")

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
        dx = dy = 0.0002777777777777768
        latlim = [latlim[0] - dy, latlim[1] + dy]
        lonlim = [lonlim[0] - dx, lonlim[1] + dx]

    timelim = [datetime.date(2000, 2, 10), datetime.date(2000, 2, 12)]
    tiles = tiles_intersect(latlim, lonlim)

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

    data_source_crs = None
    parallel = False
    spatial_tiles = True
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
    )

    return ds[req_vars_orig]


if __name__ == "__main__":
    folder = r"/Users/hmcoerver/Local/cog2_test"
    # latlim = [26.9, 33.7]
    # lonlim = [25.2, 37.2]
    latlim = [28.9, 29.7]
    lonlim = [30.2, 31.2]
    timelim = [datetime.date(2020, 7, 1), datetime.date(2020, 7, 11)]
    req_vars = ["z", "aspect", "slope"]

    variables = None
    post_processors = None

    # fn = os.path.join(os.path.join(folder, "SRTM"), "30M.nc")
    # if os.path.isfile(fn):
    #     os.remove(fn)

    # SRTM.
    ds = download(folder, latlim, lonlim, req_vars=req_vars)
    print(ds.rio.crs, ds.rio.grid_mapping)
