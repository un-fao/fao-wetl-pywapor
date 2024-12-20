import datetime
import os
import json
import pywapor
import numpy as np
from functools import partial
import pywapor.collect.accounts as accounts
from shapely.geometry.polygon import Polygon
from pywapor.general.logger import log
from shapely.geometry import shape
from pywapor.collect.protocol.projections import get_crss
import pywapor.collect.protocol.opendap as opendap
import xarray as xr
from pywapor.general.processing_functions import open_ds, remove_ds, save_ds
from pywapor.general import bitmasks
import pandas as pd
import warnings
import copy

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
    fn = f"{product_name}_h{tile[0]:02d}v{tile[1]:02d}.nc"
    return fn

def url_func(product_name, tile):
    """Returns a url at which to collect MODIS data.

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
    url = f"https://opendap.cr.usgs.gov/opendap/hyrax/{product_name}/h{tile[0]:02d}v{tile[1]:02d}.ncml.nc4?"
    return url

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
    with open(os.path.join(pywapor.collect.__path__[0], "product/MODIS_tiles.geojson")) as f:
        features = json.load(f)["features"]
    aoi = Polygon.from_bounds(lonlim[0], latlim[0], lonlim[1], latlim[1])
    tiles = list()
    for feature in features:
        shp = shape(feature["geometry"])
        tile = feature["properties"]["Name"]
        if shp.intersects(aoi):
            h, v = tile.split(" ")
            htile = int(h.split(":")[-1])
            vtile = int(v.split(":")[-1])
            tiles.append((htile, vtile))
    return tiles

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

        ds_expand = ds_expand.stack({"datetime": ("hour","time")})

        new_coords = [time + hour for time, hour in zip(ds_expand.time.values, ds_expand.hour.values)]
        
        try: # new versions of xarray require to drop all dimensions of a multi-index
            ds_expand = ds_expand.drop_vars(["datetime", "hour", "time"])
        except ValueError: # old versions throw an error when trying to drop sub-dimensions of a multiindex.
            ds_expand = ds_expand.drop_vars(["datetime"])
        
        ds_expand = ds_expand.assign_coords({"datetime": new_coords}).rename({"datetime": "time"}).sortby("time")
        ds_expand = ds_expand.drop_vars(["lst_hour"])
        ds_expand = ds_expand.transpose("time", "y", "x")
        ds_expand = ds_expand.dropna("time", how="all")

    return ds_expand

def mask_bitwise_qa(ds, var, masker = "lst_qa", 
                product_name = "MOD11A1.061", flags = ["good_qa"]):
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

def mask_qa(ds, var, masker = ("ndvi_qa", 1.0)):
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
                    "BRDF_Albedo_Band_Mandatory_Quality_shortwave": [("time", "YDim", "XDim"), "r0_qa"],
                    "MOD_Grid_BRDF_eos_cf_projection": [(), "spatial_ref"]
        }
    }

    req_dl_vars = {
        "MOD13Q1.061": {
            "ndvi": ["_250m_16_days_NDVI", "_250m_16_days_pixel_reliability", "MODIS_Grid_16DAY_250m_500m_VI_eos_cf_projection"],
        },
        "MYD13Q1.061": {
            "ndvi": ["_250m_16_days_NDVI", "_250m_16_days_pixel_reliability", "MODIS_Grid_16DAY_250m_500m_VI_eos_cf_projection"],
        },
        "MOD11A1.061": {
            "lst": ["LST_Day_1km", "Day_view_time", "QC_Day", "MODIS_Grid_Daily_1km_LST_eos_cf_projection"],
        },
        "MYD11A1.061": {
            "lst": ["LST_Day_1km", "Day_view_time", "QC_Day", "MODIS_Grid_Daily_1km_LST_eos_cf_projection"],
        },
        "MCD43A3.061": {
            "r0": ["Albedo_WSA_shortwave", "Albedo_BSA_shortwave", "BRDF_Albedo_Band_Mandatory_Quality_shortwave", "MOD_Grid_BRDF_eos_cf_projection"],
        },
    }

    out = {val:variables[product_name][val] for sublist in map(req_dl_vars[product_name].get, req_vars) for val in sublist}
    
    return out

def default_post_processors(product_name, req_vars = None):
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
        "MOD13Q1.061": {
            "ndvi": [mask_qa]
            },
        "MYD13Q1.061": {
            "ndvi": [mask_qa]
            },
        "MOD11A1.061": {
            "lst": [mask_bitwise_qa, expand_time_dim]
            },
        "MYD11A1.061": {
            "lst": [mask_bitwise_qa, expand_time_dim]
            },
        "MCD43A3.061": {
            "r0": [
                    shortwave_r0, 
                    partial(mask_qa, masker = ("r0_qa", 1.)),
                    ]
            },
    }

    out = {k:v for k,v in post_processors[product_name].items() if k in req_vars}

    return out

def download(folder, latlim, lonlim, timelim, product_name, req_vars,
                variables = None, post_processors = None):
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

    folder = os.path.join(folder, "MODIS")

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

    spatial_buffer = {"MYD13Q1.061": False, "MYD11A1.061": True, "MOD11A1.061": True, "MCD43A3.061": True, "MOD13Q1.061": False}[product_name]
    if spatial_buffer:
        latlim = [latlim[0] - 0.01, latlim[1] + 0.01]
        lonlim = [lonlim[0] - 0.01, lonlim[1] + 0.01]

    if product_name == "MOD13Q1.061" or product_name == "MYD13Q1.061":
        timedelta = np.timedelta64(8, "D")
        timelim[0] = timelim[0] - pd.Timedelta(timedelta)
    elif product_name == "MCD43A3.061":
        timedelta = np.timedelta64(12, "h")
        timelim[0] = timelim[0] - pd.Timedelta(timedelta)
    else:
        timedelta = None

    tiles = tiles_intersect(latlim, lonlim)
    coords = {"x": ["XDim", lonlim], "y": ["YDim", latlim], "t": ["time",timelim]}
    
    if isinstance(variables, type(None)):
        variables = default_vars(product_name, req_vars)

    if isinstance(post_processors, type(None)):
        post_processors = default_post_processors(product_name, req_vars)
    else:
        default_processors = default_post_processors(product_name, req_vars)
        post_processors = {k: {True: default_processors[k], False: v}[v == "default"] for k,v in post_processors.items() if k in req_vars}

    data_source_crs = get_crss("MODIS")
    parallel = False
    spatial_tiles = True
    un_pw = accounts.get("NASA")
    request_dims = True
    ds = opendap.download(fn, product_name, coords, 
                variables, post_processors, fn_func, url_func, un_pw = un_pw, 
                tiles = tiles, data_source_crs = data_source_crs, parallel = parallel, 
                spatial_tiles = spatial_tiles, request_dims = request_dims,
                timedelta = timedelta)

    return ds[req_vars_orig]

if __name__ == "__main__":

    products = [
        # ('MCD43A3.061', ["r0"]),
        ('MOD11A1.061', ["lst"]),
        # ('MYD11A1.061', ["lst"]),
        # ('MOD13Q1.061', ["ndvi"]),
        # ('MYD13Q1.061', ["ndvi"]),
    ]

    folder = r"/Users/hmcoerver/Downloads/pywapor_test"
    # latlim = [26.9, 33.7]
    # lonlim = [25.2, 37.2]
    latlim = [28.9, 29.7]
    lonlim = [30.2, 31.2]
    timelim = [datetime.date(2020, 7, 1), datetime.date(2020, 7, 11)]

    variables = None
    post_processors = None

    for product_name, req_vars in products:
        ds = download(folder, latlim, lonlim, timelim, product_name, req_vars)
        print(ds.rio.crs, ds.rio.grid_mapping)