import os
import requests
import tqdm
from pywapor.general.processing_functions import save_ds, open_ds, process_ds, remove_ds
from osgeo import gdal
import urllib
from pywapor.enhancers.apply_enhancers import apply_enhancers
from pywapor.general.logger import log

def download(fp, product_name, coords, variables, post_processors, url_func, 
                gdal_config_options = {}, waitbar = True, ndv = -9999):
    """Download data from a Cloud Optimized Geotiff hosted on a remote server.

    Parameters
    ----------
    fp : str
        Path to file in which to download.
    product_name : str
        Name of product.
    coords : dict
        Coordinate names and boundaries.
    variables : dict
        Keys are variable names, values are additional settings.
    post_processors : dict
        Processors to apply to specific variables.
    url_func : function
        Function that takes `product_name` as input and return a url.
    gdal_config_options : dict, optional
        Additional options passed to `gdal.SetConfigOption`, by default {}.
    waitbar : bool, optional
        Show a download progress bar or not, by default True.
    ndv : int, optional
        No data value to use in new file, by default -9999.

    Returns
    -------
    xr.Dataset
        Dataset with the downloaded data.
    """

    folder, fn = os.path.split(fp)

    # Create folder.
    if not os.path.isdir(folder):
        os.makedirs(folder)

    try:
        for k, v in gdal_config_options.items():
            gdal.SetConfigOption(k, v)

        bands = [int(k.replace("Band", "")) for k in variables.keys() if "Band" in k]

        # Define bounding-box.
        bb = [coords["x"][1][0], coords["y"][1][1], 
                coords["x"][1][1], coords["y"][1][0]]

        options_dict = {
                "projWin": bb,
                "format": "netCDF",
                "creationOptions": ["COMPRESS=DEFLATE", "FORMAT=NC4"],
                "noData": ndv,
                "bandList": bands,
        }

        if waitbar:
            # Create waitbar.
            waitbar = tqdm.tqdm(position = 0, total = 100, bar_format='{l_bar}{bar}|', delay = 30)

            # Define callback function for waitbar progress.
            def _callback_func(info, *args):
                waitbar.update(info * 100 - waitbar.n)

            # Add callback to gdal.Translate options.
            options_dict.update({"callback": _callback_func})

        # Set gdal.Translate options.
        options = gdal.TranslateOptions(**options_dict)

        # Check if url is local or online path.
        url = url_func(product_name)
        is_not_local = urllib.parse.urlparse(url).scheme in ('http', 'https',)
        if is_not_local and not "vsicurl" in url:
            url = f"/vsicurl/{url}"

        # Check if path is free.
        temp_path = fp.replace(".nc", "_temp.nc")
        while os.path.isfile(temp_path):
            temp_path = temp_path.replace(".nc", "_.nc")

        # Run gdal.Translate.
        ds_ = gdal.Translate(temp_path, url, options = options)
        ds_.FlushCache()
        ds_ = None

    except Exception as e:
        raise e
    finally:
        for k in gdal_config_options.keys():
            gdal.SetConfigOption(k, None)

    # Process the new netCDF.
    ds_ = open_ds(temp_path)

    ds = ds_.rename_vars({k: f"Band{v}" for k,v in zip(list(ds_.data_vars), bands)})

    ds = process_ds(ds, coords, variables)

    # Apply product specific functions.
    ds = apply_enhancers(post_processors, ds)

    # Save final output.
    out = save_ds(ds, temp_path.replace("_temp", ""), encoding = "initiate", label = f"Saving {fn}.")

    # Remove the temporary file.
    remove_ds(ds_)

    return out

def cog_dl(urls, out_fn, overview = "NONE", warp_kwargs = {}, vrt_options = {"separate": True}):

    out_ext = os.path.splitext(out_fn)[-1]
    valid_ext = {".nc": "netCDF", ".tif": "GTiff"}
    valid_cos = {".nc": ["COMPRESS=DEFLATE", "FORMAT=NC4C"], ".tif": ["COMPRESS=LZW"]}
    vrt_fn = out_fn.replace(out_ext, ".vrt")

    log.info("--> Building `.vrt` file.")
    ## Build VRT with all the required data.
    vrt_options_ = gdal.BuildVRTOptions(
        **vrt_options
    )
    vrt = gdal.BuildVRT(vrt_fn, urls, options = vrt_options_)
    vrt.FlushCache()

    ## Download the data.
    warp_options = gdal.WarpOptions(
        format = valid_ext[out_ext],
        cropToCutline = True,
        overviewLevel = overview,
        multithread = True,
        creationOptions = valid_cos[out_ext],
        **warp_kwargs,
    )
    log.info(f"--> Downloading {len(urls)} bands.")
    warp = gdal.Warp(out_fn, vrt_fn, options = warp_options)
    warp.SetMetadataItem('pyWaPOR_bb', os.environ.get("pyWaPOR_bb", "unknown"))
    warp.SetMetadataItem('pyWaPOR_period', os.environ.get("pyWaPOR_period", "unknown"))
    warp.FlushCache() # NOTE do not remove this.

    if os.path.isfile(vrt_fn):
        remove_ds(vrt_fn)

    return out_fn

def ping_urls(urls):

    faulty_urls = list()
    for url in urls:
        resp = requests.get(
            url.replace("/vsicurl/", ""), 
            stream=True
            )
        if not resp.ok:
            faulty_urls.append(url)

    if len(faulty_urls) > 0:
        print_urls = "\n".join(faulty_urls)
        log.warning(f"The following URLs are invalid: {print_urls}")

    return faulty_urls