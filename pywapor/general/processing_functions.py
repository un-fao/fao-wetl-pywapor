import os
from dask.diagnostics import ProgressBar
import numpy as np
from pywapor.general.logger import log
import xarray as xr
import numpy as np
import shutil
import glob
from osgeo import gdal
from datetime import datetime as dt
import warnings
import rasterio.warp
import pandas as pd
from pywapor.general.variables import get_var_definitions
from pywapor.general.performance import performance_check
import glob
import os
import re

def func_from_string(string):
    parts = string.split(".")
    mod_str = parts.pop(0)
    func = __import__(mod_str)
    while parts:
        sub = parts.pop(0)
        func = getattr(func, sub)
    return func

def remove_temp_files(folder):

    log_files = glob.glob(os.path.join(folder, "log.txt"))

    if not len(log_files) == 1:
        return None
    else:
        log_file = log_files[0]

    with open(log_file, "r", encoding='utf8') as f:
        lines = f.readlines()

    regex_pattern = r"--> Unable to delete temporary file `(.*)`"

    files = list()
    for line in lines:
        out = re.findall(regex_pattern,line)
        if len(out) == 1:
            file = out[0]
            if os.path.isfile(file):
                files.append(out[0])
    
    for fh in files:
        if os.path.isfile(fh):
            try:
                os.remove(fh)
            except PermissionError:
                ...
                
    return files

def log_example_ds(example_ds):
    """Writes some metadata about a `example_ds` to the logger.

    Parameters
    ----------
    example_ds : xr.Dataset
        Dataset for which to log information.
    """
    if "source" in example_ds.encoding.keys():
        log.info(f"--> Using `{os.path.split(example_ds.encoding['source'])[-1]}` as reprojecting example.").add()
    else:
        log.info(f"--> Using variable `{list(example_ds.data_vars)[0]}` as reprojecting example.").add()
    shape = example_ds.y.size, example_ds.x.size
    res = example_ds.rio.resolution()
    log.info(f"> shape: {shape}, res: {abs(res[0]):.4f}° x {abs(res[1]):.4f}°.").sub()

def adjust_timelim_dtype(timelim):
    """Convert different time formats to `datetime.datetime`.

    Parameters
    ----------
    timelim : list
        List defining a period.

    Returns
    -------
    list
        List defining a period using `datetime.datetime` objects.
    """
    if isinstance(timelim[0], str):
        timelim[0] = dt.strptime(timelim[0], "%Y-%m-%d")
        timelim[1] = dt.strptime(timelim[1], "%Y-%m-%d")
    elif isinstance(timelim[0], np.datetime64):
        timelim[0] = dt.utcfromtimestamp(timelim[0].tolist()/1e9).date()
        timelim[1] = dt.utcfromtimestamp(timelim[1].tolist()/1e9).date()
    return timelim

def remove_ds(ds):
    """Delete a dataset-file from disk, assuring it's closed properly before doing so.

    Parameters
    ----------
    ds : xr.Dataset | str
        Either a `xr.Dataset` (in which case its `source` as defined in the `encoding` attribute will be used)
        or a `str` in which case it must be a path to a file.
    """
    rmve = {"NO": False, "YES": True}.get(os.environ.get("PYWAPOR_REMOVE_TEMP_FILES", "YES"), True)
    if not rmve:
        return

    fp = None
    if isinstance(ds, xr.Dataset):
        if "source" in ds.encoding.keys():
            fp = ds.encoding["source"]
        ds = ds.close()
    elif isinstance(ds, str):
        if os.path.isfile(ds):
            fp = ds

    if not isinstance(fp, type(None)):
        try:
            ds = xr.open_dataset(fp, chunks = "auto")
            ds = ds.close()
        except OSError:
            ... # file is corrupt/incomplete

        try:
            os.remove(fp)
        except PermissionError:
            log.info(f"--> Unable to delete temporary file `{fp}`.")

def is_corrupt_or_empty(fh, group = None):
    try:
        ds = xr.open_dataset(fh, chunks = "auto", group = group)
        if ds.sizes == {}:
            info = gdal.Info(fh, format = "json")
            subdss = info["metadata"].get("SUBDATASETS", False)
            if subdss:
                group_names = set([os.path.split(v.split(":")[-1])[0] for k, v in subdss.items() if "_NAME" in k])
                corrupt = any([is_corrupt_or_empty(fh, group = group) for group in group_names])
            else:
                corrupt = True
        else:
            corrupt = False
    except OSError:
        corrupt = True
    else:
        ds.close()
    return corrupt

def has_wrong_bb_or_period(fh, ref_bb, ref_period):

    info = gdal.Info(fh, format = "json")
    md = info["metadata"].get("", {})
    bb_key = [x for x in md.keys() if re.search("pyWaPOR_bb", x)]
    period_key = [x for x in md.keys() if re.search("pyWaPOR_period", x)]

    if bb_key:
        bb = md[bb_key[0]]
    else:
        bb = "unknown"
    
    if period_key:
        period = md[period_key[0]]
    else:
        period = "unknown"

    if period == "unknown" or bb == "unknown":
        wrong = None
    elif bb == str(ref_bb) and period == str(ref_period):
        wrong = False
    else:
        wrong = True

    return wrong

def process_ds(ds, coords, variables, crs = None):
    """Apply some rioxarray related transformations to a dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to be processed.
    coords : dict
        Dictionary describing the names of the spatial coordinates.
    variables : list
        List of variables to keep.
    crs : rasterio.CRS.crs, optional
        Coordinate reference system to assign (no reprojection is done), by default None.

    Returns
    -------
    xr.Dataset
        Dataset with some attributes corrected.
    """

    ds = ds[list(variables.keys())]

    ds = ds.rename({v[0]:k for k,v in coords.items() if k in ["x", "y"]})
    ds = ds.rename({k: v[1] for k, v in variables.items()})

    if not isinstance(crs, type(None)):
        ds = ds.rio.write_crs(crs)

    ds = ds.rio.write_grid_mapping("spatial_ref")

    for var in [x for x in list(ds.variables) if x not in ds.coords]:
        if "grid_mapping" in ds[var].attrs.keys():
            del ds[var].attrs["grid_mapping"]

    ds = ds.sortby("y", ascending = False)
    ds = ds.sortby("x")

    ds.attrs = {}

    return ds

def make_example_ds(ds, folder, target_crs, bb = None, example_ds_fp = None):
    """Make an dataset suitable to use as an example for matching with other datasets.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    folder : str
        Path to folder in which to save the `example_ds`.
    target_crs : rasterio.CRS.crs
        Coordinate reference system of the `example_ds`.
    bb : list, optional
        Boundingbox of the `example_ds` ([xmin, ymin, xmax, ymax]), by default None.

    Returns
    -------
    xr.Dataset
        Example dataset.
    """
    if isinstance(example_ds_fp, type(None)):
        example_ds_fp = os.path.join(folder, "example_ds.nc")
    if os.path.isfile(example_ds_fp):
        example_ds = open_ds(example_ds_fp)
    else:
        if not isinstance(bb, type(None)):
            if ds.rio.crs != target_crs:
                loc_bb = transform_bb(target_crs, ds.rio.crs, bb)
            else:
                loc_bb = bb
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category = FutureWarning)
                ds = ds.rio.clip_box(*loc_bb)
            ds = ds.rio.pad_box(*loc_bb)
        ds = ds.rio.reproject(target_crs)
        if not isinstance(bb, type(None)):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category = UserWarning)
                ds = ds.rio.clip_box(*bb)
        ds = ds.drop_vars(list(ds.data_vars))
        example_ds = save_ds(ds, example_ds_fp, encoding = "initiate", label = f"Creating example dataset.") # NOTE saving because otherwise rio.reproject bugs.
    return example_ds

@performance_check
def save_ds(ds, fp, 
            decode_coords = "all", 
            encoding = None, 
            chunks = "auto",

            precision = "auto", 
            default_precision = 8, 
            update_precision = True

            ) -> xr.Dataset:
    """Save a `xr.Dataset` as netcdf.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to save.
    fp : str
        Path to file to create.
    decode_coords : str, optional
        Controls which variables are set as coordinate variables when
        reopening the dataset, by default `"all"`.
    encoding : "initiate" | dict | None, optional
        Apply an encoding to the saved dataset. "initiate" will create a encoding on-the-fly, by default None.
    chunks : "auto" | dict
        Define the chunks with which to perform any pending calculations, by default "auto".
    precision : int | dict, optional
        How many decimals to store for each variable, only used when `encoding` is `"initiate"`, by default 8.

    Returns
    -------
    xr.Dataset
        The newly created dataset.
    """
    # Check if file already exists.
    if os.path.isfile(fp):
        log.info("--> Appending data to an existing file.")
        appending = True
        temp_fp = fp
    else:
        appending = False
        temp_fp = fp.replace(".nc", "_temp.xx")

    # Make folder.
    folder = os.path.split(fp)[0]
    if not os.path.isdir(folder):
        os.makedirs(folder)

    # Filter unwanted coordinates.
    valid_coords = ["x", "y", "spatial_ref", "time", "time_bins", "lmbda"]
    for coord in ds.coords.values():
        if coord.name not in valid_coords:
            ds = ds.drop_vars([coord.name])

    # Set dask chunks.
    if isinstance(chunks, dict):
        chunks = {dim: v for dim, v in chunks.items() if dim in ds.dims}
    ds = ds.chunk(chunks)

    # Make sure y is decreasing.
    if "y" in ds.coords:
        if len(ds.y.dims) == 1:
            ds = ds.sortby("y", ascending = False)
        ds = ds.rio.write_transform(ds.rio.transform(recalc=True))

    chunksizes = {var: tuple([v[0] for _, v in ds[var].chunksizes.items()]) for var in ds.data_vars if np.all([spat in ds[var].coords for spat in ["x", "y"]])}

    if encoding == "initiate":

        # Determine required precision per variable.
        if precision == "auto":
            precision = {var: ds[var].attrs.get("req_precision", default_precision) for var in ds.data_vars}            
        elif isinstance(precision, dict):
            precision = {var: precision.get(var, default_precision) for var in ds.data_vars}
        elif isinstance(precision, int):
            precision = {var: precision for var in ds.data_vars}
        elif precision == "default":
            precision = {var: default_precision for var in ds.data_vars}
        else:
            raise ValueError

        dtypes = {var: determine_dtype(ds[var], precision[var], update_precision = update_precision) for var in ds.data_vars}

        log.debug(f"{dtypes}")

        # Define encoding per variable.
        encoding = {
                    var: {
                            "zlib": True,
                            "_FillValue": dtypes[var][1],
                            "chunksizes": chunksizes[var],
                            "dtype": dtypes[var][0],
                            "scale_factor": 10.**-int(dtypes[var][2]), 
                        } for var in ds.data_vars if np.all([spat in ds[var].coords for spat in ["x", "y"]])
                    }

        # Make sure spatial_ref is correctly set.
        if "spatial_ref" in ds.coords:
            for var in ds.data_vars:
                if np.all([spat in ds[var].coords for spat in ["x", "y"]]):
                    ds[var].attrs.update({"grid_mapping": "spatial_ref"})

        # Always use float64 for coordinates.
        for var in ds.coords:
            if var in ds.dims:
                encoding[var] = {"dtype": "float64"}

    # Remove _FillValue if already in attrs, can conflict with value set in encoding.
    if isinstance(encoding, dict):
        for var in ds.data_vars:
            if "_FillValue" in ds[var].attrs.keys() and "_FillValue" in encoding[var].keys():
                _ = ds[var].attrs.pop("_FillValue")
            if "scale_factor" in ds[var].attrs.keys() and "scale_factor" in encoding[var].keys():
                _ = ds[var].attrs.pop("scale_factor")
            
    bb_ = os.environ.get("pyWaPOR_bb", "unknown")
    period_ = os.environ.get("pyWaPOR_period", "unknown")
    ds.attrs.update({"pyWaPOR_bb": bb_, "pyWaPOR_period": period_})

    with ProgressBar(minimum = 900, dt = 2.0):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            warnings.filterwarnings("ignore", message="invalid value encountered in power")
            warnings.filterwarnings("ignore", message="invalid value encountered in log")
            ds.to_netcdf(temp_fp, encoding = encoding, mode = {True: "a", False: "w"}[appending])

    ds = ds.close()

    if not appending:
        os.rename(temp_fp, fp)

    ds = open_ds(fp, decode_coords = decode_coords, chunks = {})

    return ds

def open_ds(fp, decode_coords = "all", chunks = "auto", **kwargs):
    """Open a file using xarray.

    Parameters
    ----------
    fp : str
        Path to file.
    decode_coords : str, optional
        Whether or not to decode coordinates, by default "all".
    chunks : str | dict, optional
        Chunks to use, by default "auto".

    Returns
    -------
    xr.Dataset
        Opened file.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        ds = xr.open_dataset(fp, decode_coords = decode_coords, chunks = chunks, **kwargs)
    return ds

def create_dummy_mask(x, y, sign = None, slope = None, xshift_fact = None, yshift_fact = None):
    if isinstance(sign, type(None)):
        sign = np.sign(np.random.random() - 0.5)
    if isinstance(slope, type(None)):
        slope = np.random.random()
    if isinstance(xshift_fact, type(None)):
        xshift_fact = np.random.random()
    if isinstance(yshift_fact, type(None)):
        yshift_fact = np.random.random()
    slope = sign*(slope * np.ptp(y) / np.ptp(x))
    yshift = {-1: y.max(), 1: y.min()}[np.sign(slope)]
    xshift = x.min()
    mask = (x - (xshift + xshift_fact * np.ptp(x))) * slope + (yshift + sign * yshift_fact * np.ptp(y))
    return y < mask

def create_dummy_ds(varis, fp = None, shape = (10, 1000, 1000), chunks = (-1, 500, 500), 
                    sdate = "2022-02-01", edate = "2022-02-11", precision = 2, min_max = [-1, 1],
                    latlim = [20,30], lonlim = [40, 50], data_generator = "random", mask_data = False):
    check = False
    if not isinstance(fp, type(None)):
        if os.path.isfile(fp):
            try:
                os.remove(fp)
            except PermissionError:
                ...
    if not check:
        nt, ny, nx = shape
        dates = pd.date_range(sdate, edate, periods = nt)
        y,t,x = np.meshgrid(np.linspace(latlim[0], latlim[1], shape[1]), np.linspace(0, len(dates) + 1, len(dates)), np.linspace(lonlim[0],lonlim[1],shape[2]))
        if data_generator == "random":
            data = np.random.uniform(size = np.prod(shape), low = min_max[0], high=min_max[1]).reshape(shape)
        elif data_generator == "uniform":
            data = np.sqrt(x**2 + y**2)
            data = (data - data.min()) * ((min_max[1] - min_max[0]) / (data.max() - data.min())) + min_max[0]
        if mask_data:
            for i in range(nt):
                mask = create_dummy_mask(x[0,...], y[0,...])
                data[i,mask] = np.nan
        ds = xr.Dataset({k: (["time", "y", "x"], data) for k in varis}, coords = {"time": dates, "y": np.linspace(latlim[0], latlim[1], ny), "x": np.linspace(lonlim[0], lonlim[1], nx)})
        ds = ds.rio.write_crs(4326)
        if isinstance(chunks, tuple):
            chunks = {name: size for name, size in zip(["time", "y", "x"], chunks)}
        if not isinstance(fp, type(None)):
            ds = save_ds(ds, fp, chunks = chunks, encoding = "initiate", precision = precision, label = "Creating dummy dataset.")
    return ds

def determine_dtype(da, min_precision, update_precision = True):
    
    if isinstance(da, xr.DataArray):
        if not isinstance(da.attrs.get("var_range", None), type(None)):
            low_range = np.floor(da.attrs["var_range"][0] * 10**int(min_precision))
            high_range = np.ceil(da.attrs["var_range"][1] * 10**int(min_precision))
        else:
            log.warning(f"--> No `var_range` specified for {da.name}, will compute minimum and maximum.")
            low_range = np.floor(da.min().values * 10**int(min_precision))
            high_range = np.ceil(da.max().values * 10**int(min_precision))
    elif np.all([isinstance(da, list), da[0]<=da[1], len(da) == 2]):
        low_range = np.floor(da[0] * 10**int(min_precision))
        high_range = np.floor(da[1] * 10**int(min_precision))
    else:
        raise ValueError

    # If da doesn't contain a single valid value.
    if (np.isnan(low_range) & np.isnan(high_range)) or (low_range == high_range == 0):
        dtype = np.int8
        info = np.iinfo(dtype)
        ndv = info.min
        var_precision_ = 0
        dtype_name = np.dtype(dtype).name
        return dtype_name, ndv, var_precision_

    dtypes = [np.int8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64]

    dtype_name = ndv = None
    for dtype in dtypes:
        info = np.iinfo(dtype)
        if np.all([
                    info.min < low_range <= info.max, 
                    info.min < high_range <= info.max
                    ], axis = 0):
            ndv = info.min
            dtype_name = np.dtype(dtype).name
            break

    out_of_range_msg = "Precision too high."
    if isinstance(dtype_name, type(None)):
        raise ValueError(out_of_range_msg)

    x = x_ = (dtype_name, ndv)
    var_precision_ = min_precision
    
    if update_precision:
        while x == x_:
            var_precision_ += 1
            try:
                x_ = determine_dtype(da, var_precision_, update_precision = False)[:2]
            except ValueError as e:
                if str(e) == out_of_range_msg:
                    x_ = None
                else:
                    raise e

        var_precision_ -= 1

    return x[0], ndv, var_precision_

def create_wkt(latlim, lonlim):
    left = lonlim[0]
    bottom = latlim[0]
    right = lonlim[1]
    top = latlim[1]
    x = f"{left} {bottom},{right} {bottom},{right} {top},{right} {bottom},{left} {bottom}"
    return "GEOMETRYCOLLECTION(POLYGON((" + x + ")))"

def unpack(file, folder):
    fn = os.path.splitext(file)[0]
    shutil.unpack_archive(os.path.join(folder, file), folder)
    folder = [x for x in glob.glob(os.path.join(folder, fn + "*")) if os.path.isdir(x)][0]
    return folder

def transform_bb(src_crs, dst_crs, bb):
    """Transforms coordinates from one CRS to another.

    Parameters
    ----------
    src_crs : rasterio.CRS.crs
        Source CRS.
    dst_crs : rasterio.CRS.crs
        Target CRS.
    bb : list
        Coordinates to be transformed.

    Returns
    -------
    list
        Transformed coordinates.
    """
    bb =rasterio.warp.transform_bounds(src_crs, dst_crs, *bb, densify_pts=21)
    return bb

if __name__ == "__main__":

    ...

    decode_coords = "all"
    encoding = "initiate"
    chunks = "auto"
    precision = "auto"
    default_precision = 8
    # folder = r"/Users/hmcoerver/Local/dummy_ds_test"

    # varis = ["my_var"]
    # shape = (10, 1000, 1000)
    # sdate = "2022-02-02"
    # edate = "2022-02-13" 
    # fp = os.path.join(folder, "dummy_test.nc")
    # precision = 2
    # min_max = [-1, 1]

    # ds = create_dummy_ds(varis, 
    #                 shape = shape, 
    #                 sdate = sdate, 
    #                 edate = edate, 
    #                 fp = fp,
    #                 precision = precision,
    #                 min_max = min_max,
    #                 )