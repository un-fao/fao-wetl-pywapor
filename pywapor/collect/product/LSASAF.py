"""
https://nextcloud.lsasvcs.ipma.pt/s/QSABgnG4dZGBo5W?dir=undefined&path=%2FPUM-Product_User_Manual&openfile=27089
https://nextcloud.lsasvcs.ipma.pt/s/bA9gYoa5mQX2yJw?dir=undefined&path=%2FPUM-Product_User_Manual&openfile=27251
https://lsa-saf.eumetsat.int/en/data/products/radiation/
"""

import copy
import datetime
import multiprocessing
import os
import posixpath
import re

import numpy as np
import pandas as pd
import requests
import tqdm
import xarray as xr
from joblib import Memory, Parallel, delayed

import pywapor
import pywapor.collect.protocol.crawler
import pywapor.collect.protocol.opendap
from pywapor.collect import accounts
from pywapor.collect.protocol.projections import get_crss
from pywapor.enhancers.apply_enhancers import apply_enhancers
from pywapor.general import bitmasks
from pywapor.general.logger import log
from pywapor.general.processing_functions import (
    is_corrupt_or_empty,
    open_ds,
    remove_ds,
    save_ds,
)


# NOTE https://stackoverflow.com/questions/75202475/joblib-persistence-across-sessions-machines
def cache(mem, module, **mem_kwargs):
    def cache_(f):
        f.__module__ = module
        f.__qualname__ = f.__name__
        return mem.cache(f, **mem_kwargs)

    return cache_


def apply_qa(
    ds,
    var,
    masker="qa",
    product_name="MSG_MDSSFTD",
    flags=["clear_ok", "cloudy_sky_method_ok"],
):
    flag_bits = bitmasks.LSASAF_qa_translator(product_name)
    mask = bitmasks.get_mask(ds[masker].astype("uint8"), flags, flag_bits)
    new_data = ds[var].where(mask, np.nan)
    ds[var] = new_data
    return ds


def night_is_zero(ds, var, masker="qa", product_name="MSG_MDSSFTD"):
    flag_bits = bitmasks.LSASAF_qa_translator(product_name)
    mask_day_night = bitmasks.get_mask(
        ds[masker].astype("uint8"), ["clear_night"], flag_bits
    )
    mask_land_cont_water = bitmasks.get_mask(
        ds[masker].astype("uint8"), ["land", "continental_water"], flag_bits
    )
    mask = mask_land_cont_water & mask_day_night
    new_data = ds[var].where(~mask, 0)
    ds[var] = new_data
    return ds


def jouleperday_to_watt(ds, var):
    ds[var] = ds[var] / 86400
    return ds


def celsius_to_kelvin(ds, var):
    ds[var] = ds[var] + 273.15
    return ds


def fn_func(product_name, tile):
    fn = os.path.split(tile)[-1]
    return fn


def url_func(product_name, tile):
    return tile


def default_vars(product_name, req_vars):
    variables = {
        "MSG_MDSSFTD": {  # MSG Total and Diffuse DSSF, https://nextcloud.lsasvcs.ipma.pt/s/bA9gYoa5mQX2yJw
            "DSSF_TOT": [("time", "lat", "lon"), "ra_flat"],
            "FRACTION_DIFFUSE": [("time", "lat", "lon"), "diffuse_fraction"],
            "quality_flag": [("time", "lat", "lon"), "qa"],
        },
        "MSG_MDIDSSF": {  # MSG Daily DSSF, https://nextcloud.lsasvcs.ipma.pt/s/QSABgnG4dZGBo5W
            "DSSF": [("time", "lat", "lon"), "ra_flat"],
            "max_nslots_missing": [("time", "lat", "lon"), "max_nslots_missing"],
            "missing_values_percent": [
                ("time", "lat", "lon"),
                "missing_values_percent",
            ],
            "weight_missing_values_percent": [
                ("time", "lat", "lon"),
                "weight_missing_values_percent",
            ],
        },
        #########
        "MSG_MLST": {
            "LST": [("time", "lat", "lon"), "lst"],  # inst. lst, Celsius
            "quality_flag": [("time", "lat", "lon"), "qa"],
            "standard_error": [("time", "lat", "lon"), "error"],
        },
        "MSG_MLST-AS": {
            "LST": [
                ("time", "lat", "lon"),
                "lst",
            ],  # inst. lst (all-sky, i.e. clear AND cloudy), Celsius
            "quality_flag": [("time", "lat", "lon"), "qa"],
        },
        "MSG_MLST-ASv2": {
            "LST": [
                ("time", "lat", "lon"),
                "lst",
            ],  # inst. lst (all-sky, i.e. clear AND cloudy), Celsius
            "quality_flag": [("time", "lat", "lon"), "qa"],
        },
        "MSG_METREF": {
            "METREF": [("time", "lat", "lon"), "et_ref_24_mm"],  # mm/day
            "quality_flag": [("time", "lat", "lon"), "qa"],
        },
        "MSG_MH": {
            "MH": [("time", "lat", "lon"), "h_i"],  # sensible heat flux, W/m2
            "quality_flag": [("time", "lat", "lon"), "qa"],
        },
        "MSG_MLE": {
            "MLE": [("time", "lat", "lon"), "lh_i"],  # latent heat flux, W/m2
            "quality_flag": [("time", "lat", "lon"), "qa"],
        },
    }

    req_dl_vars = {
        "MSG_MDSSFTD": {
            "ra_flat": ["DSSF_TOT"],
            "diffuse_fraction": ["FRACTION_DIFFUSE"],
            "qa": ["quality_flag"],
        },
        "MSG_MDIDSSF": {
            "ra_flat": ["DSSF"],
            "max_nslots_missing": ["max_nslots_missing"],
            "missing_values_percent": ["missing_values_percent"],
            "weight_missing_values_percent": ["weight_missing_values_percent"],
        },
        "MSG_MLST": {
            "lst": ["LST"],
            "qa": ["quality_flag"],
            "error": ["standard_error"],
        },
        "MSG_MLST-AS": {
            "lst": ["LST"],
            "qa": ["quality_flag"],
        },
        "MSG_MLST-ASv2": {
            "lst": ["LST"],
            "qa": ["quality_flag"],
        },
        "MSG_METREF": {"et_ref_24_mm": ["METREF"], "qa": ["quality_flag"]},
        "MSG_MH": {"h_i": ["MH"], "qa": ["quality_flag"]},
        "MSG_MLE": {"lh_i": ["MLE"], "qa": ["quality_flag"]},
    }

    out = {
        val: variables[product_name][val]
        for sublist in map(req_dl_vars[product_name].get, req_vars)
        for val in sublist
    }

    return out


def default_post_processors(product_name, req_vars):
    post_processors = {
        "MSG_MDSSFTD": {"ra_flat": [], "diffuse_fraction": [], "qa": []},
        "MSG_MDIDSSF": {
            "ra_flat": [jouleperday_to_watt],
            "max_nslots_missing": [],
            "missing_values_percent": [],
            "weight_missing_values_percent": [],
        },
        "MSG_MLST": {"lst": [celsius_to_kelvin], "qa": [], "error": []},
        "MSG_MLST-AS": {"lst": [celsius_to_kelvin], "qa": []},
        "MSG_MLST-ASv2": {"lst": [celsius_to_kelvin], "qa": []},
        "MSG_METREF": {"et_ref_24_mm": [], "qa": []},
        "MSG_MH": {"h_i": [], "qa": []},
        "MSG_MLE": {"lh_i": [], "qa": []},
    }

    out = {k: v for k, v in post_processors[product_name].items() if k in req_vars}

    return out


def most_recent(product_name, *args):
    product_name = "MSG_MDSSFTD"

    un, pw = accounts.get("LSASAF")
    session = requests.Session()
    session.auth = (un, pw)

    satellite, product_name_ = product_name.split("_")
    x = product_name_.split("-")
    if len(x) == 1:
        product_name__ = x[0]
    elif len(x) == 2:
        product_name__ = x[0]
    else:
        raise ValueError
    format = "NETCDF"

    regex = r'nc"><code>(.*?)<\/code>'

    def scrape(base_url):
        catalog_url = posixpath.join(base_url, "catalog.html")
        resp = session.get(catalog_url)
        resp.raise_for_status()
        return resp.content

    date = pd.Timestamp.today() + pd.Timedelta(days=1)
    fns = []
    while len(fns) == 0:
        date = date - pd.Timedelta(days=1)
        base_url = f"https://thredds.lsasvcs.ipma.pt/thredds/catalog/{satellite}/{product_name__}/{format}/{date.year}/{date.month:>02}/{date.day:>02}"
        try:
            content = scrape(base_url)
        except requests.exceptions.HTTPError:
            continue
        fns = re.findall(regex, str(content))

    return pd.Timestamp.now().to_pydatetime()


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
    folder = os.path.join(folder, "LSASAF")

    satellite, product_name_ = product_name.split("_")
    product_name__ = product_name_.split("-")[0]
    format = "NETCDF"

    if not os.path.exists(folder):
        os.makedirs(folder)

    # NOTE paths on windows have a max length, this extends the max length, see
    # here for more info https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=registry
    if os.name == "nt":
        cachedir = "\\\\?\\" + os.path.join(os.path.abspath(folder), "cache")
    else:
        cachedir = os.path.join(folder, "cache")

    memory = Memory(cachedir, verbose=0)

    fp = os.path.join(folder, f"{product_name}.nc")
    req_vars_orig = copy.deepcopy(req_vars)
    if os.path.isfile(fp):
        existing_ds = open_ds(fp)
        req_vars_new = list(set(req_vars).difference(set(existing_ds.data_vars)))
        if len(req_vars_new) > 0:
            req_vars = req_vars_new
            existing_ds = existing_ds.close()
        else:
            return existing_ds[req_vars_orig]

    spatial_buffer = True
    if spatial_buffer:
        res = {"MSG": 0.05}.get(satellite, "define_the_sat_res_please")
        latlim = [latlim[0] - res, latlim[1] + res]
        lonlim = [lonlim[0] - res, lonlim[1] + res]

    coords = {"x": ["lon", lonlim], "y": ["lat", latlim[::-1]], "t": ["time", timelim]}

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

    data_source_crs = get_crss("WGS84")

    timedelta = {
        "MSG_MDIDSSF": np.timedelta64(12, "h"),
        "MSG_METREF": np.timedelta64(12, "h"),
    }.get(product_name, None)

    un, pw = accounts.get("LSASAF")
    session = requests.Session()
    session.auth = (un, pw)

    @cache(memory, "LSASAF")
    def scrape(base_url):
        catalog_url = posixpath.join(base_url, "catalog.html")
        resp = session.get(catalog_url)
        resp.raise_for_status()
        return resp.content

    regex = r'nc"><code>(.*?)<\/code>'
    dates = pd.date_range(timelim[0], timelim[1], freq="D")
    urls = list()
    for date in list(dates[:-1]):
        base_url = f"https://thredds.lsasvcs.ipma.pt/thredds/catalog/{satellite}/{product_name__}/{format}/{date.year}/{date.month:>02}/{date.day:>02}"
        content = scrape(base_url)
        fns = re.findall(regex, str(content))
        # if not isinstance(extra_filter, type(None)):
        #     fns = [x for x in fns if f"-{extra_filter}_" in x]
        urls_ = [
            posixpath.join(base_url, fn).replace("/catalog/", "/dodsC/") for fn in fns
        ]
        urls += urls_

    log.info(f"--> Found {len(urls)} relevant tiles.")

    urls_ = [
        url
        for url in urls
        if not os.path.isfile(os.path.join(folder, os.path.split(url)[-1]))
    ]
    urls__ = [url for url in urls if url not in urls_]

    log.info(f"--> Downloading remaining {len(urls_)} tiles.").add()

    def download_tile(url, folder):
        fn = os.path.split(url)[-1]
        fp_ = os.path.join(folder, fn)
        corrupt = False
        if os.path.isfile(fp_):
            corrupt = is_corrupt_or_empty(fp_)
            if corrupt:
                log.info(
                    f"--> Removing corrupt or empty file `{os.path.split(fp_)[-1]}`."
                )
                remove_ds(fp_)
            else:
                log.info(f"--> Opening existing file `{os.path.split(fp_)[-1]}`.")
                ds_ = open_ds(fp_)
        if corrupt or not os.path.isfile(fp_):
            url_ = url.replace("https://", f"https://{un}:{pw}@")
            ds_ = pywapor.collect.protocol.opendap.download_xarray(
                url_,
                fp_,
                coords,
                variables,
                {},
                data_source_crs=data_source_crs,
                timedelta=timedelta,
                parallel=False,
            )
        log.sub()
        return ds_

    n_jobs = max(1, multiprocessing.cpu_count() - 1)
    parallel = True
    if parallel:
        dss = Parallel(n_jobs=n_jobs)(
            delayed(download_tile)(url, folder) for url in tqdm.tqdm(urls_, leave=False)
        )
    else:
        dss = list()
        for i, url in enumerate(urls_):
            fn = os.path.split(url)[-1]
            log.info(f"--> ({i + 1}/{len(urls)}) Collecting `{fn}`.").add()
            dss.append(download_tile(url, folder))
            log.sub()

    for url in urls__:
        fp_ = os.path.join(folder, os.path.split(url)[-1])
        dss.append(open_ds(fp_))

    log.sub()

    # Combine all data.
    ds = xr.concat(dss, dim="time").sortby("time")

    ds = ds.rio.write_crs("epsg:4326")

    # Apply product specific functions.
    ds = apply_enhancers(post_processors, ds)

    # Save final output.
    out_ = save_ds(ds, fp, encoding="initiate", label="Saving netCDF.")

    for x in dss:
        remove_ds(x)

    return out_[req_vars_orig]


if __name__ == "__main__":
    from pywapor.general.logger import adjust_logger

    tests = [
        # ("MSG_MDSSFTD", ["ra_flat", "diffuse_fraction", "qa"]),
        # (
        #     "MSG_MDIDSSF",
        #     [
        #         "ra_flat",
        #         "max_nslots_missing",
        #         "missing_values_percent",
        #         "weight_missing_values_percent",
        #     ],
        # ),
        # ("MSG_MLST", ["lst", "qa", "error"]),
        ("MSG_MLST-AS", ["lst", "qa"]),
        ("MSG_MLST-ASv2", ["lst", "qa"]),
        # ("MSG_METREF", ["et_ref_24_mm", "qa"]),
        # ("MSG_MH", ["h_i", "qa"]),
        # ("MSG_MLE", ["lh_i", "qa"]),
    ]

    folder = r"/Users/hmcoerver/Local/ra_test"

    adjust_logger(True, folder, "INFO")

    latlim = [30.168923, 36.520673]
    lonlim = [29.787384, 44.625092]
    timelim = [datetime.date(2023, 2, 5), datetime.date(2023, 2, 6)]
    # timelim = [datetime.date(2023,2,1), datetime.date(2023,2,13)]

    variables = None
    post_processors = None

    dss = list()
    for product_name, req_vars in tests:
        # break
        log.info(f"{product_name}, {req_vars}")
        ds_x = download(folder, latlim, lonlim, timelim, product_name, req_vars)
        dss.append(ds_x)

    # product_name = "MSG_MDIDSSF"
    # req_vars = ["ra_flat", "max_nslots_missing", "missing_values_percent", "weight_missing_values_percent"]
    # ds1 = download(folder, latlim, lonlim, timelim, product_name, req_vars)

    # product_name = "MSG_MDSSFTD"
    # req_vars = ["ra_flat", "qa"]
    # ds2 = download(folder, latlim, lonlim, timelim, product_name, req_vars)

    # ds3 = pywapor.collect.product.ERA5.download(
    #                                     folder,
    #                                     latlim,
    #                                     lonlim,
    #                                     timelim,
    #                                     product_name="sis-agrometeorological-indicators",
    #                                     req_vars=["ra_flat"]
    #                             )

    # ds4 = pywapor.collect.product.MERRA2.download(
    #                                     folder,
    #                                     latlim,
    #                                     lonlim,
    #                                     timelim,
    #                                     product_name="M2T1NXRAD.5.12.4",
    #                                     req_vars=["ra_flat"]
    #                             )

    # ds5 = pywapor.collect.product.GEOS5.download(
    #                                     folder,
    #                                     latlim,
    #                                     lonlim,
    #                                     timelim,
    #                                     product_name="tavg1_2d_rad_Nx",
    #                                     req_vars=["ra_flat"]
    #                             )

    # dss = [ds1, ds2, ds3, ds4, ds5]

    # fig, axs = plt.subplots(len(dss), 1, figsize = (6, 15), sharex=True, sharey=True, dpi = 300)

    # var = "ra_flat"

    # timeslice = [np.datetime64("2023-02-01 00:00:00"), np.datetime64("2023-02-03 00:00:00")]

    # for ax, ds in zip(axs, dss):
    #     source_name = os.path.split(os.path.split(ds.encoding["source"])[0])[-1]
    #     prod_name = os.path.splitext(os.path.split(ds.encoding["source"])[-1])[0]
    #     ds_ = ds[var].sel(time = slice(*timeslice))
    #     n = ds_.sizes["time"]
    #     ds__ = ds_.mean(dim = "time")
    #     ds__.attrs["units"] = "W m-2"
    #     ds__.plot(ax = ax, vmax = 220)
    #     ax.set_title(f"{source_name}.{prod_name}, n = {n}")
    #     xlabel = ax.get_xlabel()
    #     ax.set_xlabel("")
    #     ax.set_facecolor("lightgray")

    # plt.xlabel(xlabel)
    # plt.suptitle(f"{var}\n[{timeslice[0]} - {timeslice[1]}]")

    # plt.savefig(os.path.join(folder, "ra_flat_comparison.png"))
