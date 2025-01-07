import os
import tqdm
import tempfile
import xarray as xr
from osgeo import gdal, osr
from pywapor.general.logger import log
from pywapor.general.performance import format_bytes
from pywapor.general.processing_functions import save_ds, remove_ds
gdal.UseExceptions()

def calc_slope_or_aspect(ds, var, write_init = True, max_cache_size = 4e9):

    log.add()

    # Get filepath from xr.Dataset.
    fp = ds.encoding.get("source", None)

    if not var in ["slope", "aspect"]:
        raise ValueError("Please set `var` to either `'slope'` or `'aspect'`.")

    # Write data to file if necessary.
    if write_init or isinstance(fp, type(None)):
        fp = os.path.join(tempfile.gettempdir(), f"temp_{var}.nc")
        if os.path.isfile(fp):
            remove_ds(fp)
        while os.path.isfile(fp):
            fp = fp.replace(".nc", "_.nc")
        ds = save_ds(ds, fp, label = f"Generating {var} input file.")

    fp_out = fp.replace(".nc", f"_{var}.tif")

    ds_ = gdal.Open(fp)
    subdss_ = ds_.GetSubDatasets()

    if len(subdss_) > 0:
        fp = [x[0] for x in subdss_ if x[0].split(":")[-1] == "z"][0]
        ds_ = gdal.Open(fp)

    if ds_.RasterCount > 1:
        var_names = [ds_.GetRasterBand(i+1).GetMetadata().get("NETCDF_VARNAME", "unknown") for i in range(ds_.RasterCount)]
        band_select = var_names.find("z") if "z" in var_names else None
        if isinstance(band_select, type(None)):
            log.warning(f"--> Multiple bands found in `{fp}` and none is named `z`, using first band.")
            band_select = 1
    elif ds_.RasterCount == 1:
        band_select = 1
    else:
        raise ValueError(f"No bands found in `{fp}`.")

    band = ds_.GetRasterBand(band_select)
    wkt = ds_.GetProjection()
    src = osr.SpatialReference()
    src.ImportFromWkt(wkt)
    auth = src.GetAuthorityName(None)
    code = src.GetAuthorityCode(None)

    if f"{auth}:{code}" == "EPSG:4326":
        scale = band.GetScale()
        scale_ = {True: 1, False: scale}[isinstance(scale, type(None))] # NOTE in case there is no scale-factor, we set it to 1.
        scale_factor = int(111120 / scale_)
    else:
        scale_factor = 1
        log.warning("--> Dataset has different projection than `EPSG:4326`,\
                    slope scaling might be incorrect if CRS unit is not meters.")

    # Create waitbar.
    waitbar = tqdm.tqdm(position = 0, total = 100, bar_format='{l_bar}{bar}|', delay = 5)

    # Define callback function for waitbar progress.
    def _callback_func(info, *args):
        waitbar.update(info * 100 - waitbar.n)

    options_ = {
        "slopeFormat": "degree",
        "scale": scale_factor,
        "computeEdges": True,
        "zeroForFlat": True,
        "callback": _callback_func,
    }

    _ = options_.pop({"slope": "zeroForFlat", "aspect": "scale"}[var])

    options = gdal.DEMProcessingOptions(**options_)

    f_size = lambda nbytes: "{0:.2f}{1}".format(*format_bytes(nbytes))

    current_cache_size = gdal.GetCacheMax()
    if "NETCDF:" in fp:
        fp_ = fp.split(":")[1].replace('"', '')
    else:
        fp_ = fp
    
    if os.path.isfile(fp_):
        filesize = os.path.getsize(fp_)
    else:
        filesize = 0.5 * current_cache_size

    if current_cache_size < 0.95 * filesize:
        # NOTE The maximum practical value on 32 bit OS is between 2 and 4 GB. It is the responsibility 
        # of the user to set a consistent value. See: https://gdal.org/user/configoptions.html
        new_cache_size = min(max_cache_size, 2.0 * filesize)
        log.warning(f"--> Input file ({f_size(filesize)}) is larger than current maximum allowed GDAL cache-size ({f_size(current_cache_size)}), will try increasing it to {f_size(new_cache_size)}.")
        gdal.SetCacheMax(int(new_cache_size))
        log.info(f"--> Maximum cache size is {f_size(gdal.GetCacheMax())}.")
    else:
        new_cache_size = None

    ds__ = gdal.DEMProcessing(fp_out, fp, var, options = options)
    ds__.FlushCache()
    ds__ = None

    if not isinstance(new_cache_size, type(None)):
        gdal.SetCacheMax(int(current_cache_size))
        log.info(f"--> Maximum cache size is reset to {f_size(gdal.GetCacheMax())}.")

    ds_new = xr.open_dataset(fp_out).isel(band=0, drop=True).drop_vars(["x", "y"])
    ds_new = ds_new.rename({"band_data": var})
    ds_new.attrs = {}
    for var in ds_new.data_vars:
        ds_new[var].attrs = {}
    ds[var] = ds_new[var]

    log.sub()

    return ds

if __name__ == "__main__":

    import xarray as xr

    # ds = xr.open_dataset(r"/Users/hmcoerver/Local/aspect_slope_test/GLO30_stitched.nc", decode_coords = "all")

    var = "slope"
    write_init = False
    max_cache_size = 4e9