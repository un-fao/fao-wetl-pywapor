"""
Generates input data for `pywapor.et_look`.
"""
from pywapor.collect import downloader
from pywapor.general.logger import log, adjust_logger
from pywapor.general import compositer
import pywapor.general.levels as levels
import datetime
import numpy as np
import os
import pandas as pd
import xarray as xr
from functools import partial
import pywapor.general.pre_defaults as defaults
from pywapor.general.variables import fill_attrs
from pywapor.enhancers.temperature import lapse_rate_to_all

def rename_vars(ds, *args):
    """Rename some variables in a dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset whose variables will be renamed.

    Returns
    -------
    xr.Dataset
        Dataset with renamed variables.
    """
    varis = ["p", "ra_flat", "t_air", "t_air_min", "t_air_max", "u", "vp",
            "u2m", "v2m", "qv", "p_air", "p_air_0", "wv", "t_dew"]
    present_vars = [x for x in varis if x in ds.variables]
    ds = ds.rename({k: k + "_24" for k in present_vars})
    return ds

def calc_doys(ds, *args, bins = None):
    """Calculate the day-of-the-year (doy) in the middle of a timebin and assign the results to a new
    variable `doy`.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    bins : list, optional
        List of boundaries of the timebins, by default None

    Returns
    -------
    xr.Dataset
        Dataset with a new variable containing the `doy` per timebin.
    """
    bin_doys = [int(pd.Timestamp(x).strftime("%j")) for x in bins]
    doy = np.mean([bin_doys[:-1], bin_doys[1:]], axis=0, dtype = int)
    if "time_bins" in list(ds.variables):
        ds["doy"] = xr.DataArray(doy, coords = ds["time_bins"].coords).chunk("auto")
    return ds

def add_constants(ds, *args):
    """Adds default dimensionless constants to a dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to which the constants will be added.

    Returns
    -------
    xr.Dataset
        Dataset with extra variables.
    """
    ds = ds.assign(defaults.constants_defaults())
    return ds

def remove_empty_statics(ds, *args):
    for var in ds.data_vars:
        if "time" not in ds[var].coords and "time_bins" not in ds[var].coords:
            if ds[var].isnull().all():
                ds = ds.drop(var)
                log.info(f"--> Removing `{var}` from dataset since it is empty.")
    return ds

def add_constants_new(ds, *args): # TODO make sure this only adds constants for et_look, not for se_root.
    c = defaults.constants_defaults()
    for var, value in c.items():
        if var not in ds.data_vars:
            ds[var] = value
    return ds

def main(folder, latlim, lonlim, timelim, sources = "level_2_v3", bin_length = 1, enhancers = [lapse_rate_to_all]):
    """Prepare input data for `et_look`.

    Parameters
    ----------
    folder : str
        Path to folder in which to store (intermediate) data.
    latlim : list
        Latitude limits of area of interest.
    lonlim : list
        Longitude limits of area of interest.
    timelim : list
        Period for which to prepare data.
    sources : "level_1" | "level_2" | "level_2_v3" | dict, optional
        Configuration for each variable and source, by default `"level_1"`.
    bin_length : int | "DEKAD", optional
        Composite length, by_default 1.
    enhancers : list, optional
        Functions to apply to the xr.Dataset before creating the final
        output, by default `[lapse_rate]`.

    Returns
    -------
    xr.Dataset
        Dataset with input data for `pywapor.et_look`.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    _ = adjust_logger(True, folder, "INFO")

    log.info(f"--> lonlim: {lonlim}")
    log.info(f"--> latlim: {latlim}")
    log.info(f"--> timelim: {timelim}")

    t1 = datetime.datetime.now()
    log.info("> PRE_ET_LOOK").add()

    if isinstance(sources, str):
        sources = levels.pre_et_look_levels(sources, bin_length = bin_length)

    bins = compositer.time_bins(timelim, bin_length)

    general_enhancers = enhancers + [rename_vars, fill_attrs, partial(calc_doys, bins = bins), remove_empty_statics, add_constants_new]

    adjusted_timelim = [bins[0], bins[-1]]
    buffered_timelim = [adjusted_timelim[0] - np.timedelta64(3, "D"), 
                        adjusted_timelim[1] + np.timedelta64(3, "D")]

    dss, sources = downloader.collect_sources(folder, sources, latlim, lonlim, buffered_timelim)

    try:
        ds = compositer.main(dss, sources, folder, general_enhancers, bins)
    except Exception as e:
        log.exception("")
        raise e

    t2 = datetime.datetime.now()
    log.sub().info(f"< PRE_ET_LOOK ({str(t2 - t1)})")

    return ds

if __name__ == "__main__":

#     enhancers = "default"
#     diagnostics = None
#     example_source = None
#     bin_length = "DEKAD"
    enhancers = [lapse_rate_to_all]

#     import pywapor

#     folder = r"/Users/hmcoerver/pywapor_notebooks_b"
#     latlim = [28.9, 29.7]
#     lonlim = [30.2, 31.2]
#     timelim = ["2021-07-01", "2021-07-11"]
#     composite_length = "DEKAD"

#     et_look_version = "v2"
#     export_vars = "default"

#     level = "level_1"

#     et_look_sources = pywapor.general.levels.pre_et_look_levels(level)

#     et_look_sources = {k: v for k, v in et_look_sources.items() if k in ["ndvi", "z"]}

    # et_look_sources["ndvi"]["products"] = [
    #     {'source': 'MODIS',
    #         'product_name': 'MOD13Q1.061',
    #         'enhancers': 'default'},
    #     {'source': 'MODIS', 
    #         'product_name': 'MYD13Q1.061', 
    #         'enhancers': 'default'},
    #     {'source': 'PROBAV',
    #         'product_name': 'S5_TOC_100_m_C1',
    #         'enhancers': 'default',
    #         'is_example': True}
    # ]

    # et_look_sources["r0"]["products"] = [
    #     {'source': 'MODIS',
    #         'product_name': 'MCD43A3.061',
    #         'enhancers': 'default'},
    #     {'source': 'PROBAV',
    #         'product_name': 'S5_TOC_100_m_C1',
    #         'enhancers': 'default'}
    # ]

    # se_root_sources = pywapor.general.levels.pre_se_root_levels(level)
    # se_root_sources["ndvi"]["products"] = et_look_sources["ndvi"]["products"]

    # from functools import partial
    # et_look_sources["se_root"]["products"] = [
    #     {'source': partial(pywapor.se_root.se_root, sources = se_root_sources),
    #         'product_name': 'v2',
    #         'enhancers': 'default'},]

    # ds = pywapor.pre_et_look.main(folder, latlim, lonlim, timelim, 
    #                                 sources = et_look_sources,
    #                                 bin_length = composite_length)



