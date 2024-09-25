import os
import datetime
import copy
import pandas as pd
import numpy as np
from osgeo import gdal
import pywapor.collect.protocol.cog as cog
from pywapor.collect.protocol.projections import get_crss
from pywapor.enhancers.apply_enhancers import apply_enhancers
from pywapor.general.processing_functions import open_ds, process_ds, save_ds, remove_ds

def default_vars(product_name, req_vars = ["p"]):
    """Given a `product_name` and a list of requested variables, returns a dictionary
    with metadata on which exact layers need to be requested from the server, how they should
    be renamed, and how their dimensions are defined.

    Parameters
    ----------
    product_name : str
        Name of the product.
    req_vars : list, optional
        List of variables to be collected, by default ["p"].

    Returns
    -------
    dict
        Metadata on which exact layers need to be requested from the server.
    """
    
    variables =  {
        "P05": {
            "precip": [("time", "lat", "lon"), "p"],
            "crs": [(), "spatial_ref"],
        }
    }

    req_dl_vars = {
        "P05": {
            "p": ["precip", "crs"],
        }
    }

    out = {val:variables[product_name][val] for sublist in map(req_dl_vars[product_name].get, req_vars) for val in sublist}
    
    return out

def default_post_processors(product_name, req_vars = ["p"]):
    """Given a `product_name` and a list of requested variables, returns a dictionary with a 
    list of functions per variable that should be applied after having collected the data
    from a server.

    Parameters
    ----------
    product_name : str
        Name of the product.
    req_vars : list, optional
        List of variables to be collected, by default ["p"].

    Returns
    -------
    dict
        Functions per variable that should be applied to the variable.
    """

    post_processors = {
        "P05": {
            "p": [],
        }
    }

    out = {k:v for k,v in post_processors[product_name].items() if k in req_vars}

    return out

def download(folder, latlim, lonlim, timelim, product_name = "P05", req_vars = ["p"],
                variables = None, post_processors = None, precision = 8):
    """Download CHIRPS data and store it in a single netCDF file.

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
        Name of the product to download, by default "P05".
    req_vars : list, optional
        Which variables to download for the selected product, by default ["p"].
    variables : dict, optional
        Metadata on which exact layers need to be requested from the server, by default None.
    post_processors : dict, optional
        Functions per variable that should be applied to the variable, by default None.

    Returns
    -------
    xr.Dataset
        Downloaded data.
    """
    folder = os.path.join(folder, "CHIRPS")

    if not os.path.isdir(folder):
        os.makedirs(folder)

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
        latlim = [latlim[0] - 0.05, latlim[1] + 0.05]
        lonlim = [lonlim[0] - 0.05, lonlim[1] + 0.05]

    coords = {"x": ["lon", lonlim], "y": ["lat", latlim], "t": ["time", timelim]}
    if isinstance(variables, type(None)):
        variables = default_vars(product_name, req_vars)

    if isinstance(post_processors, type(None)):
        post_processors = default_post_processors(product_name, req_vars)
    else:
        default_processors = default_post_processors(product_name, req_vars)
        post_processors = {k: {True: default_processors[k], False: v}[v == "default"] for k,v in post_processors.items() if k in req_vars}

    timedelta = pd.Timedelta(hours=12)
    data_source_crs = get_crss("WGS84")


    chirps_domain = [
        -180.0000000000000000,
        -50.0000014901161194, 
        180.0000053644180298,
        50.0000000000000000
        ]
    chirps_res = [7200, 2000]
    chirps_px_size = [(chirps_domain[2] - chirps_domain[0])/chirps_res[0],
                      (chirps_domain[3] - chirps_domain[1])/chirps_res[1]]
    
    def _snap_point(x0, xres, X):
        n = np.round((X - x0) / xres)
        Xsnap = n*xres + x0
        return Xsnap, n
    
    latlim_ = [
        _snap_point(chirps_domain[0], chirps_px_size[0], latlim[0])[0],
        _snap_point(chirps_domain[0], chirps_px_size[0], latlim[1])[0],
              ]
    lonlim_ = [
        _snap_point(chirps_domain[1], chirps_px_size[1], lonlim[0])[0],
        _snap_point(chirps_domain[1], chirps_px_size[1], lonlim[1])[0],
              ]

    warp_kwargs = dict()
    warp_kwargs["outputBounds"] = [lonlim_[0], latlim_[0], lonlim_[1], latlim_[1]]
    warp_kwargs["outputBoundsSRS"] = "epsg:4326"

    dates = pd.date_range(timelim[0], timelim[1], freq = "D")
    urls = [f"/vsicurl/https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/cogs/p05/{x.year}/chirps-v2.0.{x.year}.{x.month:>02}.{x.day:>02}.cog" for x in dates]
    temp_fn = fn.replace(".nc", "_temp.nc")
    temp_vrt = temp_fn.replace(".nc", ".vrt")
    
    gdal_config_options = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    }
    try:
        for k, v in gdal_config_options.items():
            gdal.SetConfigOption(k, v)
        x = cog.cog_dl(
                        urls, 
                        temp_fn, 
                       warp_kwargs=warp_kwargs
                       )
    except Exception as e:
        raise e
    finally:
        for k, v in gdal_config_options.items():
            gdal.SetConfigOption(k, None)
    
    ds_ = open_ds(x)
    ds = ds_.to_array("time").to_dataset(name = "precip")
    ds = ds.assign_coords({"time": [np.datetime64(x + timedelta, "ns") for x in dates]})
    ds = process_ds(ds, coords, variables, crs = data_source_crs)

    ds = apply_enhancers(post_processors, ds)

    ds = save_ds(ds, fn, precision=precision, label = f"Merging files.")

    remove_ds(temp_fn)
    if os.path.isfile(temp_vrt):
        remove_ds(temp_vrt)

    return ds[req_vars_orig]

if __name__ == "__main__":

    folder = r"/Users/hmcoerver/Local/pywapor_test_chirps"
    # # latlim = [26.9, 33.7]
    # # lonlim = [25.2, 37.2]
    bb = [2.1737, 36.2283, 2.1916, 36.2390] # [xmin, ymin, xmax, ymax] #Wad_Helal
    timelim = [np.datetime64("2022-02-26"), np.datetime64("2022-02-26")] 
    latlim = bb[1::2]
    lonlim = bb[0::2]

    product_name = "P05"
    req_vars = ["p"]
    # CHIRPS.
    ds = download(folder, latlim, lonlim, timelim)
