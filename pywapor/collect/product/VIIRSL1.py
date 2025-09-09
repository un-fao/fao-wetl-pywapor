"""
https://github.com/nsidc/earthaccess/discussions/489
https://search.earthdata.nasa.gov/search/granules/collection-details?p=C2105091501-LAADS&pg[0][v]=f&pg[0][gsk]=-start_date&q=VNP02IMG&tl=1689845353!3!!
"""

import base64
import copy
import datetime
import multiprocessing
import os
import posixpath
import urllib
from functools import partial

import numpy as np
import requests
import tqdm
import xarray as xr
from joblib import Memory, Parallel, delayed
from osgeo import gdal

from pywapor.collect import accounts
from pywapor.collect.protocol.crawler import download_url, download_urls
from pywapor.collect.protocol.opendap import make_opendap_url
from pywapor.enhancers.apply_enhancers import apply_enhancers
from pywapor.enhancers.other import drop_empty_times
from pywapor.general.curvilinear import create_grid, curvi_to_recto
from pywapor.general.logger import adjust_logger, log
from pywapor.general.processing_functions import (
    adjust_timelim_dtype,
    is_corrupt_or_empty,
    open_ds,
    remove_ds,
    save_ds,
)

gdal.UseExceptions()


def get_token():
    s3_endpoint = "https://data.laadsdaac.earthdatacloud.nasa.gov/s3credentials"

    login_resp = requests.get(s3_endpoint, allow_redirects=False)
    login_resp.raise_for_status()

    un, pw = accounts.get("NASA")
    auth = f"{un}:{pw}"
    encoded_auth = base64.b64encode(auth.encode("ascii"))

    auth_redirect = requests.post(
        login_resp.headers["location"],
        data={"credentials": encoded_auth},
        headers={"Origin": s3_endpoint},
        allow_redirects=False,
    )
    auth_redirect.raise_for_status()
    if "currently unavailable" in auth_redirect.text:
        log.warning(
            f"The Earthdata Service is currently unavailable. Check {login_resp.headers['location']}."
        )
        raise ConnectionError(
            f"Earthdata Service is currently unavailable. Check {login_resp.headers['location']}."
        )

    final = requests.get(auth_redirect.headers["location"], allow_redirects=False)
    results = requests.get(
        s3_endpoint, cookies={"accessToken": final.cookies["accessToken"]}
    )
    results.raise_for_status()

    return results.json()


def adj(number, make):
    """
    Check if a number is odd or even and adjust it
    if necessary.
    """
    is_even = number % 2 == 0
    if is_even:
        if make == "odd_up":
            number += 1
        if make == "odd_down":
            number -= 1
    else:
        if make == "even_down":
            number -= 1
        if make == "even_up":
            number += 1
    return int(number)


def download_arrays(
    path,
    subdss,
    folder,
    path_appendix=".nc",
    creation_options=["ARRAY:COMPRESS=DEFLATE"],
    dtype_is_8bit=False,
):
    fn = os.path.splitext(os.path.split(path)[-1])[0]
    path_local = os.path.join(folder, fn + path_appendix)

    if os.path.isfile(path_local):
        remove_ds(path_local)

    # check if file is local or vsicurl or vsis3
    local = urllib.parse.urlparse(path).scheme not in (
        "http",
        "https",
    )

    if (not local) and ("vsis3" in path):
        creds = get_token()

        gdal_config_options = {
            "AWS_ACCESS_KEY_ID": creds["accessKeyId"],
            "AWS_SESSION_TOKEN": creds["sessionToken"],
            "AWS_SECRET_ACCESS_KEY": creds["secretAccessKey"],
            "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        }
    else:
        gdal_config_options = {}

    try:
        for k, v in gdal_config_options.items():
            gdal.SetConfigOption(k, v)

        md_options = gdal.MultiDimTranslateOptions(
            arraySpecs=subdss, creationOptions=creation_options
        )

        ds = gdal.MultiDimTranslate(path_local, path, options=md_options)
        ds = ds.FlushCache()
    except Exception as e:
        raise e
    finally:
        for k, v in gdal_config_options.items():
            gdal.SetConfigOption(k, None)

    return path_local


def search_stac(params, cachedir=None, extra_filters={"day_night_flag": "DAY"}):
    memory = Memory(cachedir, verbose=0)

    @memory.cache()
    def _post_search(search, params_):
        query = requests.post(search, json=params_)
        query.raise_for_status()
        out = query.json()
        return out

    @memory.cache()
    def _check(ftr, extra_filters):
        links = [x for x in ftr.get("links", []) if x["rel"] == "via"]
        link = links[0].get("href", None) if len(links) > 0 else None
        metadata_resp = requests.get(link)
        metadata_resp.raise_for_status()
        metadata = metadata_resp.json()
        check = [metadata[k] == v for k, v in extra_filters.items()]
        return all(check)

    search = "https://cmr.earthdata.nasa.gov/stac/LAADS/search"
    all_scenes = list()
    links = [{"href": params, "rel": "next"}]
    while "next" in [x.get("rel", None) for x in links]:
        params_ = [x.get("href") for x in links if x.get("rel") == "next"][0]
        if isinstance(params_, dict):
            out = _post_search(search, params_)
        elif isinstance(params_, str):
            out = _post_search(params_, params_={})
        all_scenes += out["features"]
        links = out["links"]
        if "datetime" not in params.keys():
            return all_scenes
    log.info(f"--> Found {len(all_scenes)} `{params['collections'][0]}` scenes.")

    if len(extra_filters) > 0:
        log.add().info(
            f"--> Filtering `{params['collections'][0]}` scenes with `{extra_filters}`."
        )
        final = [
            x
            for x in tqdm.tqdm(
                all_scenes, position=0, bar_format="{l_bar}{bar}|", delay=5, leave=False
            )
            if _check(x, extra_filters)
        ]
        log.info(
            f"--> Found {len(final)} relevant `{params['collections'][0]}` scenes."
        ).sub()
    else:
        final = all_scenes

    return final


def create_stac_summary(bb, timelim, cachedir=None, limit=1000):
    params = dict()

    if not isinstance(timelim, type(None)):
        sd = datetime.datetime.strftime(timelim[0], "%Y-%m-%dT00:00:00Z")
        ed = datetime.datetime.strftime(timelim[1], "%Y-%m-%dT23:59:59Z")
        search_dates = f"{sd}/{ed}"
        params["datetime"] = search_dates

    params["limit"] = limit
    params["bbox"] = bb
    params["sortby"] = [{"field": "properties.datetime", "direction": "desc"}]

    params["collections"] = ["VNP03IMG.v2"]
    results03 = search_stac(params, cachedir=cachedir, extra_filters={})
    params["collections"] = ["CLDMSK_L2_VIIRS_SNPP.v1"]
    resultsqa = search_stac(params, cachedir=cachedir, extra_filters={})
    params["collections"] = ["VNP02IMG.v2"]
    results02 = search_stac(params, cachedir=cachedir)

    def get_fns(results):
        return [
            os.path.split(x["assets"][list(x["assets"].keys())[0]]["href"])[-1]
            for x in results
        ]

    def get_dts(files):
        return {
            datetime.datetime.strptime(".".join(x.split(".")[1:3]), "A%Y%j.%H%M"): x
            for x in files
        }

    files02 = get_fns(results02)
    files03 = get_fns(results03)
    filesqa = get_fns(resultsqa)

    dates02 = get_dts(files02)
    dates03 = get_dts(files03)
    datesqa = get_dts(filesqa)

    unique_dates = (
        set(dates02.keys()).union(set(dates03.keys())).union(set(datesqa.keys()))
    )

    summary = {
        k: (dates02.get(k, None), dates03.get(k, None), datesqa.get(k, None))
        for k in unique_dates
    }
    filtered_summary = {k: v for k, v in summary.items() if None not in v}

    return filtered_summary


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
        "VNP02IMG": {"bt": {(), "bt"}},
    }

    req_dl_vars = {
        "VNP02IMG": {
            "bt": ["bt"],
        },
    }

    out = {
        val: variables[product_name][val]
        for sublist in map(req_dl_vars[product_name].get, req_vars)
        for val in sublist
    }

    return out


def default_post_processors(product_name, req_vars=None):
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
        "VNP02IMG": {"bt": [partial(drop_empty_times, drop_vars=["bt"])]},
    }

    out = {k: v for k, v in post_processors[product_name].items() if k in req_vars}

    return out


def preproc(ds):
    ds = ds.rename({"lat": "y", "lon": "x", "Band1": "bt"})
    ds = ds.drop("crs")
    ds.attrs = {}
    for var in ds.data_vars:
        ds[var].attrs = {}
    ds = ds.rio.write_grid_mapping("spatial_ref")
    ds = ds.rio.write_crs("epsg:4326")
    ds = ds.rio.write_transform(ds.rio.transform(recalc=True))
    fn = os.path.split(ds.encoding["source"])[-1]
    date = np.datetime64(fn[3:13] + " " + fn[14:22].replace("_", ":"), "ns")
    return ds.expand_dims("time").assign_coords({"time": [date]})


def combine_unprojected_data(nc02_file, ncqa_file, lut_file, unproj_fn):
    if isinstance(lut_file, type(None)):
        ds_ = xr.open_dataset(nc02_file, mask_and_scale=False, group="observation_data")
        ds1 = ds_[["I05", "I05_quality_flags"]]
        ds2 = ds_[["I05_brightness_temperature_lut"]]
        ds3 = xr.open_dataset(ncqa_file, chunks="auto", group="geophysical_data")
    else:
        ds1 = xr.open_dataset(nc02_file, mask_and_scale=False)
        ds2 = xr.open_dataset(lut_file, mask_and_scale=False)
        ds3 = xr.open_dataset(ncqa_file, chunks="auto")

    # Create cloud mask. (0=cloudy, 1=probably cloudy, 2=probably clear, 3=confident clear, -1=no result)
    cmask = (
        ds3["Integer_Cloud_Mask"]
        .interp(
            {
                "number_of_lines": np.linspace(
                    0 - 0.25,
                    (ds3["number_of_lines"].size - 1) + 0.25,
                    ds3["number_of_lines"].size * 2,
                ),
                "number_of_pixels": np.linspace(
                    0 - 0.25,
                    (ds3["number_of_pixels"].size - 1) + 0.25,
                    ds3["number_of_pixels"].size * 2,
                ),
            },
            kwargs={"fill_value": "extrapolate"},
        )
        .drop_vars(["number_of_lines", "number_of_pixels"])
    )

    # Convert DN to Brightness Temperature using the provided lookup table.
    bt_da = ds2["I05_brightness_temperature_lut"].isel(number_of_LUT_values=ds1["I05"])

    # Chunk and mask invalid pixels.
    bt = (
        bt_da.chunk("auto")
        .where(
            (bt_da >= bt_da.attrs["valid_min"])
            & (bt_da <= bt_da.attrs["valid_max"])
            & (bt_da != bt_da.attrs["_FillValue"])
            & (ds1["I05_quality_flags"] == 0)
            & (cmask == 3),
            bt_da.attrs["_FillValue"],
        )
        .rename("bt")
        .to_dataset()
    )

    _ = save_ds(bt, unproj_fn, encoding="initiate", label="Combining data.").close()


def most_recent(product_name, latlim, lonlim):
    bb = [lonlim[0], latlim[0], lonlim[1], latlim[1]]

    summary = create_stac_summary(bb, None, None, 20)

    return max(summary.keys())


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
    """Download VIIRSL1 data and store it in a single netCDF file.

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
    folder = os.path.join(folder, "VIIRSL1")

    # NOTE paths on windows have a max length, this extends the max length, see
    # here for more info https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=registry
    if os.name == "nt":
        cachedir = "\\\\?\\" + os.path.join(os.path.abspath(folder), "cache")
    else:
        cachedir = os.path.join(folder, "cache")

    if not os.path.exists(folder):
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

    timelim = adjust_timelim_dtype(timelim)

    spatial_buffer = True
    if spatial_buffer:
        dx = dy = 0.0033
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

    bb, nx, ny = create_grid(latlim, lonlim, dx_dy=(0.0033, 0.0033))
    filtered_summary = create_stac_summary(bb, timelim, cachedir=cachedir)

    log.info(f"--> Collecting {len(filtered_summary)} VIIRS scenes.")

    buckets = ("VNP02IMG", "VNP03IMG", "CLDMSK_L2_VIIRS_SNPP")
    all_urls = {
        k: tuple(f"/vsis3/prod-lads/{buckets[i]}/{v[i]}" for i in range(3))
        for k, v in filtered_summary.items()
    }
    all_proj_files = list()

    for i, (date, (nc02, nc03, nc_cloud)) in enumerate(copy.deepcopy(all_urls).items()):
        date_str = str(date).replace(" ", "_").replace(":", "_")
        proj_fn = os.path.join(folder, f"bt_{date_str}_projected.nc")
        if os.path.isfile(proj_fn):
            all_proj_files.append(proj_fn)
            _ = all_urls.pop(date)

    log.info(
        f"--> {len(all_proj_files)} scenes already downloaded, {len(all_urls)} remaining."
    ).add()

    dl_method = "opendap"
    if dl_method == "get_request":
        token = accounts.get("VIIRSL1")[1]

    # i = 0
    # date = list(all_urls.keys())[i]
    # nc02, nc03, nc_cloud = all_urls[date]

    for i, (date, (nc02, nc03, nc_cloud)) in enumerate(all_urls.items()):
        date_str = str(date).replace(" ", "_").replace(":", "_")
        unproj_fn = os.path.join(folder, f"bt_{date_str}_unprojected.nc")
        proj_fn = os.path.join(folder, f"bt_{date_str}_projected.nc")

        log.info(
            f"--> ({i + 1}/{len(all_urls)}) Processing '{os.path.split(nc02)[-1]}'."
        ).add()

        cleanup = []

        if (
            dl_method == "s3_bucket"
        ):  # NOTE only works in-region, see https://github.com/nsidc/earthaccess/discussions/489
            log.warning(
                "--> The `s3_bucket` method only works if you are running this script from the correct region."
            )
            parallel = True
            if parallel:
                inputs = [
                    {
                        "path": nc03,
                        "subdss": ["/geolocation_data/latitude"],
                        "folder": folder,
                        "path_appendix": "_lat.nc",
                    },
                    {
                        "path": nc03,
                        "subdss": ["/geolocation_data/longitude"],
                        "folder": folder,
                        "path_appendix": "_lon.nc",
                    },
                    {
                        "path": nc02,
                        "subdss": [
                            "/observation_data/I05_quality_flags",
                            "/observation_data/I05",
                        ],
                        "folder": folder,
                    },
                    {
                        "path": nc02,
                        "subdss": ["/observation_data/I05_brightness_temperature_lut"],
                        "folder": folder,
                        "path_appendix": "_lut.nc",
                    },
                    {
                        "path": nc_cloud,
                        "subdss": ["/geophysical_data/Integer_Cloud_Mask"],
                        "folder": folder,
                        "creation_options": ["COMPRESS=DEFLATE"],
                        "dtype_is_8bit": True,
                    },
                ]
                n_jobs = min(5, multiprocessing.cpu_count())
                lats_file, lons_file, nc02_file, lut_file, ncqa_file = Parallel(
                    n_jobs=n_jobs
                )(delayed(download_arrays)(**i) for i in inputs)
            else:
                lats_file = download_arrays(
                    nc03,
                    ["/geolocation_data/latitude"],
                    folder,
                    path_appendix="_lat.nc",
                )
                lons_file = download_arrays(
                    nc03,
                    ["/geolocation_data/longitude"],
                    folder,
                    path_appendix="_lon.nc",
                )
                nc02_file = download_arrays(
                    nc02,
                    ["/observation_data/I05_quality_flags", "/observation_data/I05"],
                    folder,
                )
                lut_file = download_arrays(
                    nc02,
                    ["/observation_data/I05_brightness_temperature_lut"],
                    folder,
                    path_appendix="_lut.nc",
                )
                ncqa_file = download_arrays(
                    nc_cloud,
                    ["/geophysical_data/Integer_Cloud_Mask"],
                    folder,
                    creation_options=["COMPRESS=DEFLATE"],
                    dtype_is_8bit=True,
                )

        elif dl_method == "get_request":
            base_url = "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/"
            nc02_parts = os.path.normpath(nc02).split(os.sep)
            nc03_parts = os.path.normpath(nc03).split(os.sep)
            nc_cloud_parts = os.path.normpath(nc_cloud).split(os.sep)
            year_doy = nc02_parts[-1].split(".")[1][1:]
            nc02_url = posixpath.join(
                base_url,
                "5200",
                nc02_parts[-2],
                year_doy[:4],
                year_doy[4:],
                nc02_parts[-1],
            )
            nc03_url = posixpath.join(
                base_url,
                "5200",
                nc03_parts[-2],
                year_doy[:4],
                year_doy[4:],
                nc03_parts[-1],
            )
            ncqa_url = posixpath.join(
                base_url,
                "5110",
                nc_cloud_parts[-2],
                year_doy[:4],
                year_doy[4:],
                nc_cloud_parts[-1],
            )
            urls = [nc02_url, nc03_url, ncqa_url]
            groups = ["observation_data", "geolocation_data", "geophysical_data"]
            for group, url in zip(groups, urls.copy()):
                fp_ = os.path.join(folder, os.path.split(url)[-1])
                if os.path.isfile(fp_):
                    corrupt = is_corrupt_or_empty(fp_, group=group)
                    if corrupt:
                        remove_ds(fp_)
                    else:
                        log.info(
                            f"--> Skipping downloading of `{os.path.split(url)[-1]}`."
                        )
                        urls.remove(url)
            headers = {"Authorization": "Bearer " + token}
            n_jobs = min(len(urls), multiprocessing.cpu_count())
            _ = download_urls(urls, folder, headers=headers, parallel=n_jobs)
            nc02_file_ = os.path.join(folder, os.path.split(nc02_url)[-1])
            nc03_file = os.path.join(folder, os.path.split(nc03_url)[-1])
            ncqa_file_ = os.path.join(folder, os.path.split(ncqa_url)[-1])
            # NOTE The data has already been downloaded, just running this to align the files
            # with the `method = s3_bucket` branch.
            lats_file = download_arrays(
                nc03_file,
                ["/geolocation_data/latitude"],
                folder,
                path_appendix="_lat.nc",
            )
            lons_file = download_arrays(
                nc03_file,
                ["/geolocation_data/longitude"],
                folder,
                path_appendix="_lon.nc",
            )
            lut_file = download_arrays(
                nc02_file_,
                ["/observation_data/I05_brightness_temperature_lut"],
                folder,
                path_appendix="_lut.nc",
            )
            nc02_file = download_arrays(
                nc02_file_,
                ["/observation_data/I05_quality_flags", "/observation_data/I05"],
                folder,
                path_appendix="_data.nc",
            )
            ncqa_file = download_arrays(
                ncqa_file_,
                ["/geophysical_data/Integer_Cloud_Mask"],
                folder,
                creation_options=["COMPRESS=DEFLATE"],
                path_appendix="_cloud.nc",
                dtype_is_8bit=True,
            )
            cleanup += [nc03_file, nc02_file_, ncqa_file_]

        elif dl_method == "opendap":
            base_url = "https://ladsweb.modaps.eosdis.nasa.gov/opendap/RemoteResources/laads/allData"
            nc02_parts = os.path.normpath(nc02).split(os.sep)
            nc03_parts = os.path.normpath(nc03).split(os.sep)
            nc_cloud_parts = os.path.normpath(nc_cloud).split(os.sep)
            year_doy = nc02_parts[-1].split(".")[1][1:]

            base_url02 = posixpath.join(
                base_url,
                "5200",
                nc02_parts[-2],
                year_doy[:4],
                year_doy[4:],
                nc02_parts[-1],
            )
            base_url03 = posixpath.join(
                base_url,
                "5200",
                nc03_parts[-2],
                year_doy[:4],
                year_doy[4:],
                nc03_parts[-1],
            )
            base_cloud = posixpath.join(
                base_url,
                "5110",
                nc_cloud_parts[-2],
                year_doy[:4],
                year_doy[4:],
                nc_cloud_parts[-1],
            )

            # Download geolocation data.
            order = {
                "/geolocation_data/longitude": {},
                "/geolocation_data/latitude": {},
            }
            url = make_opendap_url(base_url03, order)
            log.info(f"--> Downloading `{nc03_parts[-1]}`.")
            nc03_file = download_url(url, os.path.join(folder, nc03_parts[-1]))

            # Determine AOI inside scene.
            log.info("--> Determining indices of AOI inside scene.").add()
            geoloc_ = xr.open_dataset(nc03_file, group="geolocation_data")
            coords = {
                **{
                    f"{k}_750": (k, np.repeat(np.arange(0, v / 2), 2))
                    for k, v in geoloc_.sizes.items()
                },
                **{f"{k}_375": (k, np.arange(0, v)) for k, v in geoloc_.sizes.items()},
            }
            geoloc = geoloc_.assign_coords(coords)
            geoloc = geoloc.where(
                (geoloc.longitude < lonlim[1])
                & (geoloc.longitude > lonlim[0])
                & (geoloc.latitude > latlim[0])
                & (geoloc.latitude < latlim[1]),
                drop=True,
            )

            if np.prod(list(geoloc.sizes.values())) == 0:
                cleanup += [nc03_file]
                log.info("--> Found 0 pixels inside AOI, skipping scene.").sub()
                log.sub()
                continue

            domain_375 = {
                k.replace("_375", ""): [
                    adj(geoloc[k].min(), "even_down"),
                    adj(geoloc[k].max(), "odd_up"),
                ]
                for k in geoloc.coords
                if "_375" in k
            }
            domain_750 = {
                k.replace("_750", ""): [int(geoloc[k].min()), int(geoloc[k].max())]
                for k in geoloc.coords
                if "_750" in k
            }
            log.info(
                f"--> Found {np.prod([v[1] - v[0] + 1 for v in domain_375.values()]):,} pixels inside AOI."
            ).sub()

            geoloc_ds = geoloc_.isel(
                {k: slice(v[0], v[1] + 1) for k, v in domain_375.items()}
            )
            lats_file_ds = save_ds(
                geoloc_ds[["latitude"]],
                os.path.join(folder, nc03_parts[-1].replace(".nc", "_lat.nc")),
                label="Saving latitudes.",
            )
            lats_file = lats_file_ds.encoding["source"]
            lats_file_ds.close()

            lons_file_ds = save_ds(
                geoloc_ds[["longitude"]],
                os.path.join(folder, nc03_parts[-1].replace(".nc", "_lon.nc")),
                label="Saving longitudes.",
            )
            lons_file = lons_file_ds.encoding["source"]
            lons_file_ds.close()

            # # Check domains
            # for dim in list(domain_750.keys()):
            #     size_750 = domain_750[dim][1] - domain_750[dim][0] + 1
            #     size_375 = domain_375[dim][1] - domain_375[dim][0] + 1
            #     print(dim, size_750, size_375)
            #     print(size_750 == size_375/2)

            # Download observation data.
            log.info(f"--> Downloading `{nc02_parts[-1]}` for AOI.")
            order = {
                "/observation_data/I05_quality_flags": domain_375,
                "/observation_data/I05": domain_375,
                "/observation_data/I05_brightness_temperature_lut": {},
            }
            url = make_opendap_url(base_url02, order)
            nc02_file = download_url(
                url, os.path.join(folder, nc02_parts[-1].replace(".nc", "_data_lut.nc"))
            )

            # Download cloud mask.
            log.info(f"--> Downloading `{nc_cloud_parts[-1]}` for AOI.")
            order = {
                "/geophysical_data/Integer_Cloud_Mask": domain_750,
            }
            url = make_opendap_url(base_cloud, order)
            ncqa_file = download_url(
                url,
                os.path.join(folder, nc_cloud_parts[-1].replace(".nc", "_cloud.nc")),
            )

            lut_file = None
            geoloc_.close()
            cleanup += [nc03_file]
        else:
            raise ValueError

        if not os.path.isfile(unproj_fn):
            combine_unprojected_data(nc02_file, ncqa_file, lut_file, unproj_fn)

        warp_kwargs = {"outputBounds": bb, "width": nx, "height": ny}
        _ = curvi_to_recto(
            lats_file, lons_file, {"bt": unproj_fn}, proj_fn, warp_kwargs=warp_kwargs
        )
        all_proj_files.append(proj_fn)

        cleanup += [nc02_file, ncqa_file, lut_file, unproj_fn, lats_file, lons_file]
        for x in cleanup:
            if not isinstance(x, type(None)):
                remove_ds(x)

        log.sub()

    log.sub()

    ds = xr.open_mfdataset(all_proj_files, preprocess=preproc)

    # Apply product specific functions.
    ds = apply_enhancers(post_processors, ds)

    ds = save_ds(ds, fn, encoding="initiate", label="Merging files.")

    for x in all_proj_files:
        remove_ds(x)

    return ds[req_vars_orig]


if __name__ == "__main__":
    folder = r"/Users/hmcoerver/Local/viirs_solomon_prob"
    bb = [
        36.4219163206182230,
        32.2415207774500132,
        36.5695091730081572,
        32.3762104800108617,
    ]
    latlim = bb[1::2]
    lonlim = bb[0::2]

    timelim = [datetime.date(2023, 3, 1), datetime.date(2023, 9, 30)]
    product_name = "VNP02IMG"

    variables = None
    post_processors = None

    req_vars = ["bt"]

    adjust_logger(True, folder, "INFO")

    # ds = download(
    #     folder,
    #     latlim,
    #     lonlim,
    #     timelim,
    #     product_name,
    #     req_vars,
    #     variables=variables,
    #     post_processors=post_processors,
    # )

    # timelim = [np.datetime64 ('2023-07-03T00:00:00.000000000'),
    #            np.datetime64('2023-07-10T00:00:00.000000000')]
    # latlim = [29.4, 29.6]
    # lonlim = [30.7, 30.9]

    # timelim = [datetime.datetime(2021, 3, 4), datetime.datetime(2021, 5, 5)]
    # lonlim = [22.78125, 32.48438]
    # latlim = [23.72777, 32.1612]
    # # folder = workdir = r"/Users/hmcoerver/Local/viirs"
    # product_name = "VNP02IMG"
    # req_vars = ["bt"]
    # variables = None
    # post_processors = None
    # folder = r"/Users/hmcoerver/Local/viirs_test"

    # adjust_logger(True, folder, "INFO")

    # i = 2

    # folder = os.path.join(folder, "VIIRSL1")
    # date = datetime.datetime(2022, 4, 30, 7, 54)
    # nc02 = '/vsis3/prod-lads/VNP02IMG/VNP02IMG.A2022120.0754.002.2022270171852.nc'
    # nc03 = '/vsis3/prod-lads/VNP03IMG/VNP03IMG.A2022120.0754.002.2022120162348.nc'
    # nc_cloud = '/vsis3/prod-lads/CLDMSK_L2_VIIRS_SNPP/CLDMSK_L2_VIIRS_SNPP.A2022120.0754.001.2022120192728.nc'

    # path, subdss, folder = (nc03, ["/geolocation_data/latitude"], folder)
    # path_appendix="_lat.nc"

    # creation_options = ["ARRAY:COMPRESS=DEFLATE"]
    # dtype_is_8bit = False

    # bb, nx, ny = create_grid(latlim, lonlim, dx_dy = (0.0033, 0.0033))
    # timelim = adjust_timelim_dtype(timelim)

    # params = {'limit': 250,
    #             'bbox': [22.78125, 23.72777, 32.48655, 32.16257],
    #             'datetime': '2021-03-04T00:00:00Z/2021-05-05T23:59:59Z',
    #             'collections': ['VNP02IMG.v2']}

    # endpoint = 'https://cmr.earthdata.nasa.gov/stac/LAADS'
    # extra_filters = {"day_night_flag": "DAY"}

    # # cachedir = 'your_cache_location_directory'
    # cachedir = os.path.join(folder, "VIIRS_API_cache")

    # # memory = Memory(cachedir, verbose=0)
