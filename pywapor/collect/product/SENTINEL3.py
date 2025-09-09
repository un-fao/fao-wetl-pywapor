import copy
import glob
import os
from datetime import datetime as dt

import numpy as np
import xarray as xr

import pywapor.collect.protocol.copernicus_odata as copernicus_odata
from pywapor.general.bitmasks import SENTINEL3_qa_translator, get_mask
from pywapor.general.curvilinear import create_grid, curvi_to_recto
from pywapor.general.logger import adjust_logger
from pywapor.general.processing_functions import (
    adjust_timelim_dtype,
    open_ds,
    remove_ds,
)


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
        "SL_2_LST___": {
            "LST_in.nc": [(), "lst", []],
            "geodetic_in.nc": [(), "coords", []],
            "flags_in.nc": [(), "qa", []],
        }
    }

    req_dl_vars = {
        "SL_2_LST___": {"lst": ["LST_in.nc", "geodetic_in.nc", "flags_in.nc"]},
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
        "SL_2_LST___": {"lst": []},
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
    start_dtime = np.datetime64(dt.strptime(fn.split("_")[7], "%Y%m%dT%H%M%S"), "ns")
    end_dtime = np.datetime64(dt.strptime(fn.split("_")[8], "%Y%m%dT%H%M%S"), "ns")
    dtime = start_dtime + (end_dtime - start_dtime) / 2
    return dtime


def s3_processor(scene_folder, variables, bb=None, **kwargs):
    ncs = [
        glob.glob(os.path.join(scene_folder, "**", "*" + k), recursive=True)[0]
        for k in variables.keys()
    ]

    ds_ = xr.open_mfdataset(ncs)

    ds = ds_.set_coords(("longitude_in", "latitude_in"))
    ds = ds.rename_vars({"longitude_in": "x", "latitude_in": "y"})
    ds = ds.rename_dims({"rows": "ny", "columns": "nx"})
    ds = ds[["LST", "LST_uncertainty", "confidence_in"]]

    flags = ["summary_cloud", "summary_pointing", "snow"]
    flag_bits = SENTINEL3_qa_translator()
    mask = get_mask(ds["confidence_in"], flags, flag_bits)

    ds = ds.where((ds["LST_uncertainty"] < 2.5) & (~mask))
    ds = ds.drop_vars(["LST_uncertainty", "confidence_in"])
    ds = ds.rename_vars({"LST": "lst"})
    ds["x"].attrs = {}
    ds["y"].attrs = {}

    combined_fn = os.path.join(scene_folder, "combined.nc")
    _ = ds.to_netcdf(
        combined_fn,
        encoding={
            "lst": {"_FillValue": -9999},
            "x": {"_FillValue": -9999, "scale_factor": 1},
            "y": {"_FillValue": -9999, "scale_factor": 1},
        },
    )

    bb_, nx, ny = create_grid(bb[1::2], bb[0::2], dx_dy=(0.01, 0.01))
    warp_kwargs = {"outputBounds": bb_, "width": nx, "height": ny}
    lats = f'NETCDF:"{combined_fn}":y'
    lons = f'NETCDF:"{combined_fn}":x'
    data = {"lst": f'NETCDF:"{combined_fn}":lst'}
    out_fn = os.path.join(scene_folder, "warped.nc")
    _ = curvi_to_recto(lats, lons, data, out_fn, warp_kwargs=warp_kwargs)

    ds_.close()
    ds.close()
    remove_ds(combined_fn)

    ds_ = open_ds(out_fn)
    ds = ds_.rename({"lat": "y", "lon": "x", "Band1": "lst"})
    ds = ds.drop_vars(["crs"])
    ds.attrs = {}
    for var in ds.data_vars:
        ds[var].attrs = {}
    ds = ds.rio.write_grid_mapping("spatial_ref")
    ds = ds.rio.write_crs("epsg:4326")
    ds = ds.sortby("y", ascending=False)
    ds = ds.rio.write_transform(ds.rio.transform(recalc=True))

    return ds, [ds_] + ncs


def most_recent(product_name, latlim, lonlim):
    return copernicus_odata.most_recent(product_name, latlim, lonlim, "SENTINEL-3")


def download(
    folder,
    latlim,
    lonlim,
    timelim,
    product_name,
    req_vars,
    variables=None,
    post_processors=None,
    extra_search_kwargs={},
):
    """Download SENTINEL3 data and store it in a single netCDF file.

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
        Extra search kwargs passed to SentinelAPI, by default {}.

    Returns
    -------
    xr.Dataset
        Downloaded data.
    """
    product_folder = os.path.join(folder, "SENTINEL3")

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

    spatial_buffer = True
    if spatial_buffer:
        dx = dy = 0.01
        latlim = [latlim[0] - dy, latlim[1] + dy]
        lonlim = [lonlim[0] - dx, lonlim[1] + dx]

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

    scenes = copernicus_odata.download(
        product_folder,
        latlim,
        lonlim,
        timelim,
        "SENTINEL3",
        product_name,
        node_filter=node_filter,
    )
    ds = copernicus_odata.process_sentinel(
        scenes,
        variables,
        time_func,
        os.path.split(fn)[-1],
        post_processors,
        s3_processor,
        bb=bb,
    )

    return ds[req_vars_orig]


if __name__ == "__main__":
    bb = [
        107.7646933699809324,
        12.7301600204295262,
        108.0391593389445148,
        12.9241818414943772,
    ]

    folder = r"/Users/hmcoerver/Local/s3_new"
    adjust_logger(True, folder, "INFO")
    timelim = ["2023-08-29", "2023-09-02"]
    latlim = [29.4, 29.7]
    lonlim = [30.7, 31.0]

    latlim = [bb[1], bb[3]]
    lonlim = [bb[0], bb[2]]

    product_name = "SL_2_LST___"

    req_vars = ["lst"]
    post_processors = None
    variables = None
    variables = default_vars(product_name, req_vars)
    extra_search_kwargs = {}

    product_folder = os.path.join(folder, "SENTINEL3")
    node_filter = None
    product_name = "SENTINEL3"
    product_type = "SL_2_LST___"

    timelim = adjust_timelim_dtype(timelim)

    scene_folder = "/Users/hmcoerver/Local/s3_new/S3B_SL_2_LST____20230901T024030_20230901T024330_20230901T043910_0180_083_260_2700_PS2_O_NR_004.SEN3"

    # source_name = "SENTINEL3"
    # scene_folder = "/Users/hmcoerver/Local/s3_new/SENTINEL3/S3A_SL_2_LST____20230829T075655_20230829T075955_20230829T100445_0179_102_363_2520_PS1_O_NR_004.SEN3"
    # ds = download(folder, latlim, lonlim, timelim, product_name,
    #             req_vars, variables = variables,  post_processors = post_processors)
