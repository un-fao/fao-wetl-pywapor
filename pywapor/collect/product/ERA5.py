import datetime
import os
from functools import partial

import numpy as np
import requests

from pywapor.collect.protocol import cds
from pywapor.enhancers.pressure import pa_to_kpa
from pywapor.enhancers.temperature import kelvin_to_celsius
from pywapor.enhancers.wind import adjust_wind_height, windspeed
from pywapor.general.logger import adjust_logger
from pywapor.general.processing_functions import open_ds, remove_ds


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
        "sis-agrometeorological-indicators": {
            "2m_temperature": [
                {"statistic": "24_hour_mean", "format": "zip", "version": "1_1"},
                "t_air",
            ],
            "2m_dewpoint_temperature": [
                {"statistic": "24_hour_mean", "format": "zip", "version": "1_1"},
                "t_dew",
            ],
            "2m_relative_humidity": [
                {
                    "time": [f"{x:02d}_00" for x in [6, 9, 12, 15, 18]],
                    "format": "zip",
                    "version": "1_1",
                },
                "rh",
            ],
            "10m_wind_speed": [
                {"statistic": "24_hour_mean", "format": "zip", "version": "1_1"},
                "u",
            ],
            "vapour_pressure": [
                {"statistic": "24_hour_mean", "format": "zip", "version": "1_1"},
                "vp",
            ],
            "solar_radiation_flux": [{"format": "zip", "version": "1_1"}, "ra_flat"],
        },
        "reanalysis-era5-single-levels": {
            "total_column_water_vapour": [
                {
                    "time": [f"{x:02d}:00" for x in range(24)],
                    "format": "netcdf",
                    "product_type": "reanalysis",
                },
                "wv",
            ],
            "10m_u_component_of_wind": [
                {
                    "time": [f"{x:02d}:00" for x in range(24)],
                    "format": "netcdf",
                    "product_type": "reanalysis",
                },
                "u10m",
            ],
            "10m_v_component_of_wind": [
                {
                    "time": [f"{x:02d}:00" for x in range(24)],
                    "format": "netcdf",
                    "product_type": "reanalysis",
                },
                "v10m",
            ],
            "2m_dewpoint_temperature": [
                {
                    "time": [f"{x:02d}:00" for x in range(24)],
                    "format": "netcdf",
                    "product_type": "reanalysis",
                },
                "t_dew",
            ],
            "mean_sea_level_pressure": [
                {
                    "time": [f"{x:02d}:00" for x in range(24)],
                    "format": "netcdf",
                    "product_type": "reanalysis",
                },
                "p_air_0",
            ],
            "surface_pressure": [
                {
                    "time": [f"{x:02d}:00" for x in range(24)],
                    "format": "netcdf",
                    "product_type": "reanalysis",
                },
                "p_air",
            ],
            "2m_temperature": [
                {
                    "time": [f"{x:02d}:00" for x in range(24)],
                    "format": "netcdf",
                    "product_type": "reanalysis",
                },
                "t_air",
            ],
        },
    }

    req_dl_vars = {
        "sis-agrometeorological-indicators": {
            "t_air": ["2m_temperature"],
            "t_air_max": ["2m_temperature"],
            "t_air_min": ["2m_temperature"],
            "t_dew": ["2m_dewpoint_temperature"],
            "rh": ["2m_relative_humidity"],
            "u": ["10m_wind_speed"],
            "vp": ["vapour_pressure"],
            "ra_flat": [
                "solar_radiation_flux"
            ],  # [J m-2d-1], Downward solar radiation ('Solar_Radiation_Flux')
        },
        "reanalysis-era5-single-levels": {
            "wv": ["total_column_water_vapour"],
            "u10m": ["10m_u_component_of_wind"],
            "v10m": ["10m_v_component_of_wind"],
            "u2m": ["10m_u_component_of_wind"],
            "v2m": ["10m_v_component_of_wind"],
            "u": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
            "t_dew": ["2m_dewpoint_temperature"],
            "p_air_0": ["mean_sea_level_pressure"],
            "p_air": ["surface_pressure"],
            "t_air": ["2m_temperature"],
            "t_air_min": ["2m_temperature"],
            "t_air_max": ["2m_temperature"],
        },
    }

    out = {
        val: variables[product_name][val]
        for sublist in map(req_dl_vars[product_name].get, req_vars)
        for val in sublist
    }

    return out


def jouleperday_to_watt(ds, var):
    ds[var] = ds[var] / 86400
    return ds


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
        "sis-agrometeorological-indicators": {
            "t_air": [kelvin_to_celsius],
            "t_air_max": [
                partial(kelvin_to_celsius, in_var="t_air", out_var="t_air_max")
            ],
            "t_air_min": [
                partial(kelvin_to_celsius, in_var="t_air", out_var="t_air_min")
            ],
            "t_dew": [kelvin_to_celsius],
            "rh": [],
            "u": [adjust_wind_height],
            "vp": [],  # is already mbar
            "ra_flat": [jouleperday_to_watt],
        },
        "reanalysis-era5-single-levels": {
            "wv": [],  # is already kg/m2
            "u10m": [],
            "v10m": [],
            "u2m": [adjust_wind_height],
            "v2m": [adjust_wind_height],
            "u": [windspeed, adjust_wind_height],
            "t_dew": [kelvin_to_celsius],
            "p_air_0": [pa_to_kpa],
            "p_air": [pa_to_kpa],
            "t_air": [kelvin_to_celsius],
            "t_air_max": [
                partial(kelvin_to_celsius, in_var="t_air", out_var="t_air_max")
            ],
            "t_air_min": [
                partial(kelvin_to_celsius, in_var="t_air", out_var="t_air_min")
            ],
        },
    }

    out = {k: v for k, v in post_processors[product_name].items() if k in req_vars}

    return out


def most_recent(product_name, *args):
    url = f"https://cds.climate.copernicus.eu/api/catalogue/v1/collections/{product_name}"
    r = requests.get(url)
    time_interval = r.json()["extent"]["temporal"]["interval"][0]
    end_time = time_interval[1]
    return datetime.datetime.strptime(end_time, "%Y-%m-%dT%H:%M:%SZ")


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
    """Download ERA5 data and store it in a single netCDF file.

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
    product_folder = os.path.join(folder, "ERA5")

    if not os.path.exists(product_folder):
        os.makedirs(product_folder)

    fn_final = os.path.join(product_folder, f"{product_name}.nc")
    if os.path.isfile(fn_final):
        ds = open_ds(fn_final)
        if np.all([x in ds.data_vars for x in req_vars]):
            return ds[req_vars]
        else:
            remove_ds(ds)

    spatial_buffer = True
    if spatial_buffer:
        buffer = {
            "sis-agrometeorological-indicators": 0.1,
            "reanalysis-era5-single-levels": 0.25,
        }[product_name]
        latlim = [latlim[0] - buffer, latlim[1] + buffer]
        lonlim = [lonlim[0] - buffer, lonlim[1] + buffer]

    if isinstance(variables, type(None)):
        variables = default_vars(product_name, req_vars)

    if isinstance(post_processors, type(None)):
        post_processors = default_post_processors(product_name, req_vars)
    else:
        default_processors = default_post_processors(product_name, req_vars)
        post_processors = {
            k: {True: default_processors[k], False: v}[v == "default"]
            for k, v in post_processors.items()
        }

    ds = cds.download(
        product_folder,
        product_name,
        latlim,
        lonlim,
        timelim,
        variables,
        post_processors,
    )

    return ds


if __name__ == "__main__":
    folder = r"/Users/hmcoerver/Local/era_test"
    latlim = [28.9, 29.7]
    lonlim = [30.2, 31.2]
    timelim = ["2021-06-26", "2021-07-05"]

    # product_name = "sis-agrometeorological-indicators"
    product_name = "reanalysis-era5-single-levels"
    req_vars = ["t_air", "t_dew"]  # , "vp", "ra"]
    # req_vars = ["u"]

    variables = None
    post_processors = None

    adjust_logger(True, folder, "INFO")

    ds = download(
        folder,
        latlim,
        lonlim,
        timelim,
        product_name=product_name,
        req_vars=req_vars,
        variables=variables,
        post_processors=post_processors,
    )
