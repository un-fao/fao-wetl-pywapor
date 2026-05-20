import xarray as xr
import numpy as np
from pywapor.general.processing_functions import save_ds, open_ds
from pywapor.enhancers.smooth.plotters import make_overview
from pywapor.general.logger import log

def xr_dist_to_finite(y, dim = "time"):

    from pywapor.enhancers.smooth.core import dist_to_finite

    if not dim in y.dims:
        raise ValueError

    out = xr.apply_ufunc(
        dist_to_finite, y, y[dim],
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[dim]],
        vectorize=False,
        dask="parallelized",
    )

    return out

def xr_choose_func(y, lmbdas, dim):

    from pywapor.enhancers.smooth.core import _wt1, _wt2

    funcs = [_wt1, _wt2]
    y_dims = getattr(y, "ndim", 0)
    lmbdas_dims = getattr(lmbdas, "ndim", 0)
    if y_dims in [2,3] and lmbdas_dims in [1]:
        wt_func = funcs[1]
        icd = [[dim],[],["lmbda"],[],[],[],[],[]]
        ocd = [["lmbda", dim]]
    elif y_dims in [2] and lmbdas_dims in [2]:
        raise ValueError
    else:
        wt_func = funcs[0]
        icd = [[dim],[],[],[],[],[],[],[]]
        ocd = [[dim]]

    return wt_func, icd, ocd

def xr_cve(y, x, lmbdas, u):

    from pywapor.enhancers.smooth.core import second_order_diff_matrix, cve1

    # Check dimension and dtypes are valid.
    x, y, lmbdas, dim = assert_dtypes(x, y, lmbdas)

    # Normalize x-coordinates
    x = (x - x.min()) / (x.max() - x.min()) * x.size

    # Create x-aware delta matrix.
    A = second_order_diff_matrix(x)

    # Make default u weights if necessary.
    if isinstance(u, type(None)):
        u = np.ones(x.shape)

    # Apply whittaker smoothing along axis.
    cves = xr.apply_ufunc(
        cve1, lmbdas, y, A, u,
        input_core_dims = [["lmbda"],[dim],[],[]],
        output_core_dims = [["lmbda"]],
        dask = "allowed",
        )
    
    return cves

def xr_wt(y, x, lmbdas, u = None, a = 0.5, min_drange = -np.inf, 
          max_drange = np.inf, max_iter = 10):
    
    from pywapor.enhancers.smooth.core import second_order_diff_matrix

    # Check dimension and dtypes are valid.
    x, y, lmbdas, dim = assert_dtypes(x, y, lmbdas)

    # Normalize x-coordinates
    x = (x - x.min()) / (x.max() - x.min()) * x.size

    # Create x-aware delta matrix.
    A = second_order_diff_matrix(x)

    # Make default u weights if necessary.
    if isinstance(u, type(None)):
        u = np.ones(x.shape)

    # Choose which vectorized function to use.
    _wt, icd, ocd = xr_choose_func(y, lmbdas, dim)

    # Make sure lmbdas is chunked similar to y.
    if not isinstance(y.chunk, type(None)):
        lmbdas = lmbdas.chunk({k: v for k, v in y.unify_chunks().chunksizes.items() if k in lmbdas.dims})

    # Apply whittaker smoothing along axis.
    z = xr.apply_ufunc(
        _wt, y, A, lmbdas, u, a, min_drange, max_drange, max_iter,
        input_core_dims = icd,
        output_core_dims = ocd,
        dask = "allowed",
        )

    # Add some metadata.
    z.attrs = {"a": f"{a:.2f}", 
               "min_drange": str(min_drange), 
               "max_drange": str(max_drange)}

    return z

def make_weights(sensor_da, weights):
    weights_dict = {{v: k for k, v in sensor_da.attrs.items()}[sensor]: weight for sensor, weight in weights.items() if sensor in sensor_da.attrs.values()}
    values = np.array(list(weights_dict.values()))
    coords = np.array(list(weights_dict.keys()), dtype = int)
    transformer = xr.DataArray(values, dims=["sensor"], coords = {"sensor": coords})
    u = transformer.sel(sensor = sensor_da).values
    return u

def assert_dtypes(x, y, lmbdas):

    # Check x and y.
    assert x.ndim == 1
    if isinstance(x, xr.DataArray) and isinstance(y, xr.DataArray):
        dim_name = x.dims[0]
        assert dim_name in y.dims
    elif isinstance(x, np.ndarray) and isinstance(y, xr.DataArray):
        dim_names = [k for k,v in y.sizes.items() if v == x.size]
        if len(dim_names) != 1:
            raise ValueError
        else:
            dim_name = dim_names[0]
            x = xr.DataArray(x, dims = [dim_name], coords = y.dim_name)
    elif isinstance(x, xr.DataArray) and isinstance(y, np.ndarray):
        x = x.values
        dim_name = None
        assert x.size == y.shape[-1]
    elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        assert x.size == y.shape[-1]
        dim_name = None
    else:
        raise TypeError

    # Check lmbdas.
    if isinstance(lmbdas, float) or isinstance(lmbdas, int) or isinstance(lmbdas, list):
        lmbdas = np.array(lmbdas)
    assert lmbdas.ndim <= 2
    if isinstance(x, xr.DataArray) and (isinstance(lmbdas, np.ndarray) or np.isscalar(lmbdas)):
        if not np.isscalar(lmbdas):
            assert lmbdas.ndim <= 1
            if lmbdas.ndim == 0:
                lmbdas = float(lmbdas)
            else:
                lmbdas = xr.DataArray(lmbdas, dims = ["lmbda"], coords = {"lmbda": lmbdas})
        # else:
        lmbdas = xr.DataArray(lmbdas)
    elif isinstance(x, xr.DataArray) and isinstance(lmbdas, xr.DataArray):
        ...
    elif isinstance(x, np.ndarray) and (isinstance(lmbdas, np.ndarray) or np.isscalar(lmbdas)):
        if lmbdas.ndim == 0:
            lmbdas = float(lmbdas)
        elif lmbdas.ndim == 2 and y.ndim == 3:
            assert y.shape[:-1] == lmbdas.shape
    elif isinstance(x, np.ndarray) and isinstance(lmbdas, xr.DataArray):
        lmbdas = lmbdas.values
        if lmbdas.ndim == 0:
            lmbdas = float(lmbdas)
    else:
        raise TypeError
    
    return x, y, lmbdas, dim_name

def whittaker_smoothing(ds, var, lmbdas = 100., weights = None, a = 0.5, 
                        max_iter = 10, out_fh = None, xdim = "time", max_dist = None,
                        new_x = None, export_all = False, chunks = {}, make_plots = None,
                        valid_drange = [-np.inf, np.inf], **kwargs):
    """Apply Whittaker smoothing to a variable in a dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing a variable `var` which has at least the dimension called `xdim`.
    var : str
        Name of the variable in `ds` to smooth. The variable can have a shape
        (nt), (nx,ny,nt) or (n,nt) with nt the size of the dimension `xdim`.
    lmbdas : int | float | np.array | xr.DataArray, optional
        Lambda value to use for smoothing, shape should be (), (m) or (nx, ny), by default 100.
    a : float, optional
        Apply quantile smoothing, value can be between 0.01 and 0.99. When 0.5,
        no iterations (limited by `max_iter`) are done, by default 0.5
    max_iter : int, optional
        Maximum number of iterations to perform when applying quantile smoothing, by default 10
    out_fh : _type_, optional
        Path to store results, when None the output is saved in the same folder as the input, by default None.
    xdim : str, optional
        The dimension describing the x values of the variable to be smoothed, by default "time"
    new_x : _type_, optional
        Extra values to add to the x dimension (used for interpolation), by default None
    export_all : bool, optional
        Whether to save only the smoothed data or also diagnostics information, by default False

    Returns
    -------
    xr.Dataset
        Dataset containing at least (depending on `export_all`) a variable called `{var}_smoothed`.
    """
    crs = ds.rio.crs
    
    # Check if the chunk overwriter is correct (i.e. doesnt chunk along core-dims)
    if not getattr(chunks, "get", lambda x,y: False)(xdim, False) == -1 and (not chunks == {}):
        log.warning(f"--> Adjusting defined chunks (`{chunks}`), to avoid chunks along core dimension ({xdim}).")
        chunks = {dim: {True: -1, False: getattr(chunks, "get", lambda x,y: chunks)(dim, "auto")}[dim == xdim] for dim in ds.dims}

    # Check if the ds is chunked correctly in case `chunks` is not set.
    is_chunked = ds.chunksizes.get(xdim, False)
    if is_chunked and (getattr(is_chunked, "__len__", lambda: 1)() != 1) and (chunks == {}):
        chunks = {dim: {True: -1, False: "auto"}[dim == xdim] for dim in ds.dims}
        log.warning(f"--> Adjusting chunks to (`{chunks}`), to avoid chunks along core dimension ({xdim}).")

    # Chunk dataset.
    ds = ds.chunk(chunks)

    # Add new x values.
    if not isinstance(new_x, type(None)) and getattr(xdim, '__len__', lambda: 0)() > 0:
        ds = xr.merge([ds, xr.Dataset({xdim: new_x})], compat='no_conflicts', join='outer').drop_duplicates(xdim).chunk(chunks).sortby(xdim).chunk(chunks)
        if "sensor" in ds.data_vars:
            sensor_id = np.nanmax(np.unique(ds["sensor"])) + 1
            ds["sensor"] = ds["sensor"].fillna(sensor_id).assign_attrs({str(int(sensor_id)): "Interp."})
            if not isinstance(weights, type(None)):
                weights["Interp."] = 0.0

    # Add lmbdas as coordinate.
    ds = ds.assign_coords({"lmbda": {True: np.array([lmbdas]), False: lmbdas}[np.isscalar(lmbdas)]})

    # Create weights.
    if not isinstance(weights, type(None)) and ("sensor" in ds.data_vars):
        u = make_weights(ds["sensor"], weights)
        source_legend = {i: f"{x} [{weights.get(x, np.nan):.2f}]" for i, x in ds["sensor"].attrs.items()}
        ds["sensor"].attrs = source_legend
    else:
        u = None

    # Only do this when more then one lmbda is provided.
    if np.any([getattr(lmbdas, 'size', 1) > 1, export_all, "plot_folder" in kwargs.keys()]):
        if a != 0.5 and getattr(lmbdas, 'size', 1) > 1:
            log.warning(f"--> Picking lambda when `a` != 0.5 (`a` = {a}) is unsupported and can result in unexpected behaviour.")
        if not np.isinf(valid_drange[0]) and getattr(lmbdas, 'size', 1) > 1:
            log.warning(f"--> Picking lambda with forces bounds (min = {valid_drange[0]}) is unsupported and can result in unexpected behaviour.")
        if not np.isinf(valid_drange[1]) and getattr(lmbdas, 'size', 1) > 1:
            log.warning(f"--> Picking lambda with forces bounds (max = {valid_drange[1]}) is unsupported and can result in unexpected behaviour.")
        ds["cves"] = xr_cve(ds[var], ds[xdim], ds["lmbda"], u)
        ds["lmbda_sel"] = ds["cves"].idxmin("lmbda")
        lmbdas = ds["lmbda_sel"]

    # Smooth the data.
    ds[f"{var}_smoothed"] = xr_wt(ds[var], ds[xdim], lmbdas = lmbdas, u = u, a = a,
                                      min_drange = valid_drange[0], max_drange = valid_drange[1],
                                      max_iter = max_iter)

    # Interpolate any data that was skipped because of stability issues.
    ds[f"{var}_smoothed"] = ds[f"{var}_smoothed"].interpolate_na(dim = xdim)

    # Mask values that are too far away from any measurement.
    if not isinstance(max_dist, type(None)):
        xdist = xr_dist_to_finite(ds[var], dim = xdim)
        ds[f"{var}_smoothed"] = ds[f"{var}_smoothed"].where(xdist <= max_dist, drop = False)

    # Create a plot
    if "plot_folder" in kwargs.keys():
        ds = ds.compute()
        # NOTE kwargs are passed on to make_overview.
        # kwargs = {"point_method": "equally_spaced", "n": 3, "offset": 0.1, "folder": r""}
        # kwargs = {"point_method": "worst", "n": 5, "xdim": "time", "folder": r""}
        plot_folder = kwargs.pop("plot_folder")
        make_overview(ds, var, plot_folder, **kwargs)

    # Drop irrelevant data.
    if not export_all:
        ds = ds[[f"{var}_smoothed"]]
        if not isinstance(new_x, type(None)) and getattr(xdim, '__len__', lambda: 0)() > 0:
            ds = ds.sel({xdim: new_x})

    # Make sure the dimensions are in the correct order. TODO Maybe move this inside `save_ds`.
    ds = ds.transpose("time", "y", "x", ...)

    if not isinstance(crs, type(None)):
        ds = ds.rio.write_crs(crs)

    if not isinstance(out_fh, type(None)):
        ds = save_ds(ds, out_fh, encoding = "initiate", label = f"Applying whittaker smoothing ({var}).")

    # Give warnings if data is outside of defined range.
    if not np.isinf(valid_drange[0]):
        min_bound = float(ds[f"{var}_smoothed"].min().values)
        if min_bound < valid_drange[0]:
            log.warning(f"--> Minimum of `{var}_smoothed` is smaller than `valid_drange` min ({min_bound:.2f} < {valid_drange[0]}).")
            ds[f"{var}_smoothed"] = ds[f"{var}_smoothed"].clip(valid_drange[0], np.inf)

    if not np.isinf(valid_drange[1]):
        max_bound = float(ds[f"{var}_smoothed"].max().values)
        if max_bound > valid_drange[1]:
            log.warning(f"--> Maximum of `{var}_smoothed` is larger than `valid_drange` max ({max_bound:.2f} > {valid_drange[1]}).")
            ds[f"{var}_smoothed"] = ds[f"{var}_smoothed"].clip(-np.inf, valid_drange[1])
            
    return ds

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    ds = open_ds(r"/Users/hmcoerver/Local/poster_whit_data/consolidated_data.nc")

    ds.sensor.attrs["3"] = 'MOD13Q1.061'
    ds.sensor.attrs["4"] = 'MYD13Q1.061'

    _new_x = ds["new_x"].values
    ds = ds.drop_vars("new_x").unify_chunks()

    var = "ndvi"
    lmbdas = 100.
    weights = None
    a = 0.5
    max_iter = 10
    out_fh = r"/Users/hmcoerver/Local/poster_whit_data/out_data.nc"
    xdim = "time"
    max_dist = None
    new_x = None
    export_all = True
    chunks = {}
    valid_drange = [-1.0, 1.0]
    kwargs = {
        "plot_folder": r"/Users/hmcoerver/Local/poster_whit_data/graphs",
    }

    out = whittaker_smoothing(ds, var, lmbdas = lmbdas, weights = weights, a = a, 
                        max_iter = max_iter, out_fh = out_fh, xdim = xdim, max_dist = max_dist,
                        new_x = new_x, export_all = export_all, chunks = chunks, make_plots = None,
                        valid_drange = valid_drange, **kwargs)

    import matplotlib.pyplot as plt
    from pywapor.enhancers.smooth.plotters import plot_point

    fig = plt.figure(figsize = (10, 5))
    ax = fig.gca()
    point_ds = out.sel(x = 30.725, y = 29.410, method = "nearest")
    plot_point(ax, point_ds.compute(), var, ylim = [-0.2, 1], t_idx = None, title = True, xtick = True)
