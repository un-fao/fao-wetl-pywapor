"""
Functions to prepare input for `pywapor.et_look`, more specifically to
group various parameters in time to create composites. 
"""
import os
import dask
import numpy as np
import xarray as xr
import pandas as pd
import pywapor.general.levels as levels
from itertools import chain
from pywapor.general.logger import log
from pywapor.general.processing_functions import save_ds, func_from_string, open_ds, remove_ds, adjust_timelim_dtype, log_example_ds
from pywapor.general.reproject import align_pixels
from pywapor.enhancers.apply_enhancers import apply_enhancer
from pywapor.enhancers.smooth.whittaker import whittaker_smoothing

dask.config.set(**{'array.slicing.split_large_chunks': True})

def add_times(ds, bins, composite_type):
    """Add times to the time coordinates, so that every bin has at least one
    datapoint.

    Parameters
    ----------
    ds : xr.Dataset
        Datasat for which to check empty bins.
    bins : list
        List of np.datetime64's which are the boundaries of the groups into
        which the variables will grouped.
    composite_type : {"min" | "max" | "mean"}
        Type of composites that will be created based on the data inside each bin.

    Returns
    -------
    xr.Dataset
        Dataset to which time coordinates have been added to assure no empty bins
        exist.
    """
    
    try:
        bin_count = ds.time.groupby_bins("time", bins).count()
        empty_bins = bin_count.sel({"time_bins": bin_count.isnull()}).time_bins
    except ValueError as e:
        if "None of the data falls within bins" in str(e):
            x = [pd.Interval(pd.Timestamp(x0), 
                            pd.Timestamp(x1)) for x0, x1 in zip(bins[:-1], bins[1:])]
            empty_bins = xr.DataArray(x, {"time_bins": x})
        else:
            raise e

    new_t = determine_new_x(empty_bins.values, composite_type, bins = None, dtype = None)

    if len(new_t) > 0:
        ds = xr.merge([ds, xr.Dataset({"time": new_t})], compat='no_conflicts').sortby("time")

    return ds

def determine_new_x(intervals, composite_type, bins = None, dtype = None):
    
    if not isinstance(bins, type(None)):
        intervals = [pd.Interval(left = pd.Timestamp(l), right = pd.Timestamp(r)) for l, r in zip(bins[:-1], bins[1:])]
    
    if composite_type == "mean":
        new_t = [x.mid for x in intervals]
    else:
        new_t1 = [x.left + pd.Timedelta(seconds=1) for x in intervals]
        new_t2 = [x.right - pd.Timedelta(seconds=1) for x in intervals]
        new_t = new_t1 + new_t2
    
    if not isinstance(dtype, type(None)):
        new_t = [dtype(x) for x in new_t]

    return new_t

def time_bins(timelim, bin_length):
    """Based on the time limits and the bin length, create the bin boundaries.

    Parameters
    ----------
    timelim : list
        Period for which to prepare data.
    bin_length : int | "DEKAD"
        Length of the bins in days or "DEKAD" for dekadal bins.

    Returns
    -------
    list
        List of np.datetime64's which are the boundaries of the groups into
        which the variables will grouped.
    """
    timelim = adjust_timelim_dtype(timelim)
    sdate = timelim[0]
    edate = timelim[1]
    if bin_length == "DEKAD":
        dekad1 = pd.date_range(sdate - pd.to_timedelta(35, unit='d'), edate + pd.to_timedelta(35, unit='d'), freq = "MS")
        dekad2 = dekad1 + pd.Timedelta(10, unit='D')
        dekad3 = dekad1 + pd.Timedelta(20, unit='D')
        dates = np.sort(np.array([dekad1, dekad2, dekad3]).flatten())
        big_interval = pd.Interval(pd.Timestamp(sdate), pd.Timestamp(edate))
        intervals = [pd.Interval(pd.Timestamp(x), pd.Timestamp(y)) for x,y in zip(dates[:-1], dates[1:])]
        out = np.unique([[np.datetime64(x.right, "ns"), np.datetime64(x.left, "ns")] for x in intervals if x.overlaps(big_interval)]).flatten()
    else:
        days = (edate - sdate).days
        no_periods = int(days // bin_length + 1)
        dates = pd.date_range(sdate, periods = no_periods + 1 , freq = f"{bin_length}D")
        out = dates.to_numpy()
    return out

def main(dss, sources, folder, general_enhancers, bins):
    """Create composites for variables contained in the 'xr.Dataset's in 'dss'.

    Parameters
    ----------
    dss : dict
        Keys are tuples of ('source', 'product_name'), values are xr.Dataset's 
        which will be aligned along the time dimensions.
    sources : dict
        Configuration for each variable and source.
    folder : str
        Path to folder in which to store (intermediate) data.
    general_enhancers : list
        Functions to apply to the xr.Dataset before creating the final
        output, by default "default".
    bins : list
        List of 'np.datetime64's which are the boundaries of the groups into
        which the variables will grouped.

    Returns
    -------
    xr.Dataset
        Dataset with variables grouped into composites.
    """
    chunks = {"time": -1, "y": "auto", "x": "auto"}

    # Open unopened netcdf files.
    dss = {**{k: open_ds(v) for k, v in dss.items() if isinstance(v, str)}, 
            **{k:v for k,v in dss.items() if not isinstance(v, str)}}

    final_path = os.path.join(folder, "et_look_in.nc")

    dss2 = dict()

    compositers = {
        "mean": xr.DataArray.mean,
        "min": xr.DataArray.min,
        "max": xr.DataArray.max,
    }

    cleanup = list()

    log.info(f"--> Compositing {len(sources)} variables.").add()

    for i, (var, config) in enumerate(sources.items()):

        dst_path = os.path.join(folder, "_composites", f"{var}_bin.nc")

        spatial_interp = config["spatial_interp"]
        temporal_interp = config["temporal_interp"]
        composite_type = config["composite_type"]

        log.info(f"--> ({i+1}/{len(sources)}) Compositing `{var}` ({composite_type}).").add()

        if os.path.isfile(dst_path):
            ds = open_ds(dst_path)
        else:

            if isinstance(temporal_interp, dict):
                kwargs = temporal_interp.copy()
                temporal_interp = kwargs.pop("method")
            else:
                kwargs = {}

            # Align pixels of different products for a single variable together.
            dss_part = [ds[[var]] for ds in dss.values() if var in ds.data_vars]
            dss1, temp_files1 = align_pixels(dss_part, folder, spatial_interp, fn_append = "_step1")
            cleanup.append(temp_files1)

            # Combine different source_products (along time dimension).
            if np.all(["time" in x[var].dims for x in dss1]):
                ds = xr.combine_nested(dss1, concat_dim = "time").chunk(chunks).sortby("time").squeeze()
            elif np.all(["time" not in x[var].dims for x in dss1]):
                ds = xr.concat(dss1, "stacked").median("stacked")
                if len(dss1) > 1:
                    log.warning(f"--> Multiple ({len(dss1)}) sources for time-invariant `{var}` found, reducing those with 'median'.")
            else:
                ds = xr.combine_nested([x for x in dss1 if "time" in x[var].dims], concat_dim = "time").chunk(chunks).sortby("time").squeeze()
                log.warning(f"--> Both time-dependent and time-invariant data found for `{var}`, dropping time-invariant data.")

            if "time" in ds.dims:

                if temporal_interp:

                    # When combining different products, it is possible to have images 
                    # on the exact same date & time. In that case, the median of those images
                    # is used. So gaps in one image are filled in with data from the other
                    # image(s).
                    if np.unique(ds.time).size != ds.time.size:
                        groups = ds.groupby(ds["time"])
                        ds = groups.median(dim = "time")
                        ds = ds.chunk(chunks)
                        log.warning(f"--> Multiple `{var}` images for an identical datetime found, reducing those with 'median'.")

                    if temporal_interp == "whittaker":
                        log.info("--> Applying whittaker smoothing.")
                        log.add().info(f"> shape: {ds[var].shape}, kwargs: {list(kwargs.keys())}.").sub()
                        if "weights" in kwargs.keys():
                            ds["sensor"] = xr.combine_nested([xr.ones_like(x.time.astype(int)) * i for i, x in enumerate(dss1)], concat_dim="time").sortby("time")
                            source_legend = {str(i): os.path.split(x.encoding["source"])[-1].replace(".nc", "") for i, x in enumerate(dss1)}
                            ds["sensor"] = ds["sensor"].assign_attrs(source_legend)
                        new_x = determine_new_x(None, composite_type, bins = bins, dtype = lambda x: np.datetime64(x, "ns"))
                        ds = whittaker_smoothing(ds, var, new_x = new_x, chunks = chunks, **kwargs)
                        ds = ds.rename_vars({f"{var}_smoothed": var})
                        temporal_interp = "linear"
                    else:
                        ds = add_times(ds, bins, composite_type = composite_type).chunk(chunks)
                        ds = ds.interpolate_na(dim = "time", method = temporal_interp, **kwargs)

                # Make composites.
                ds[var] = compositers[composite_type](ds[var].groupby_bins("time", bins, labels = bins[:-1]))

            # Drop time coordinates.
            if "time" in ds.coords:
                ds = ds.drop_vars(["time"])

            # Set dimension order.
            ds = ds.transpose(*sorted(list(ds.sizes.keys()), key = lambda e: ["time_bins", "y", "x"].index(e)))

            # Save output
            ds = save_ds(ds, dst_path, label = f"Saving `{var}` composites.")
        log.sub()

        dss2[var] = ds
        # cleanup.append([ds])


    log.sub()

    # Apply variable specific enhancers
    variables = levels.find_setting(sources, "variable_enhancers")
    for var in variables:
        if var in dss2.keys():
            for func in sources[var]["variable_enhancers"]:
                if isinstance(func, str):
                    func = func_from_string(func)
                dss2[var] = func(dss2, var, folder)

    # Align all the variables together.
    example_source = levels.find_setting(sources, "is_example", min_length = 1)
    if len(example_source) == 1:
        example_ds = dss[example_source[0]]
        log_example_ds(example_ds)
    elif len(example_source) == 0:
        example_ds = None
    else:
        log.warning(f"--> Multiple example datasets found, selecting finest resolution.")
        example_ds = None

    spatial_interps = [sources[list(x.data_vars)[0]]["spatial_interp"] for x in dss2.values()]
    dss3, temp_files3 = align_pixels(dss2.values(), folder, spatial_interps, example_ds, stack_dim = "time_bins", fn_append = "_step2")
    cleanup.append(temp_files3)
    
    # Merge everything
    ds = xr.merge(dss3, compat='no_conflicts')

    # Apply general enhancers.
    for func in general_enhancers:
        ds, label = apply_enhancer(ds, None, func)

    while os.path.isfile(final_path):
        final_path = final_path.replace(".nc", "_.nc")

    ds = save_ds(ds, final_path, encoding = "initiate", 
                    label = f"Creating merged file `{os.path.split(final_path)[-1]}`.")
    
    for nc in list(chain.from_iterable(cleanup)):
        remove_ds(nc)

    return ds

# if __name__ == "__main__":

#     cleanup = False

#     dss = {
#         ("LANDSAT", "LC08_SR"): r"/Users/hmcoerver/Local/custom_levels/LANDSAT/LC08_SR.nc",
#         ("MODIS", "MYD13Q1.061"): r"/Users/hmcoerver/Local/custom_levels/MODIS/MYD13Q1.061.nc",
#     }

#     folder = r"/Users/hmcoerver/Local/custom_levels"
#     timelim = ["2021-08-11", "2021-08-21"]
#     lonlim = [68.5, 69.0]
#     latlim = [25.5, 26.0]

#     import datetime

#     sources = {
#         "ndvi":         [("MODIS", "MOD13Q1.061"), ("MODIS", "MYD13Q1.061")],
#         "r0":           [("MODIS", "MCD43A3.061")],
#         "lst":          [("MODIS", "MOD11A1.061"), ("MODIS", "MYD11A1.061")],
#         "z":            [("SRTM", "30M")],
#         "p":            [("CHIRPS", "P05")],
#         "ra":           [("MERRA2", "M2T1NXRAD.5.12.4")],
#         "t_air":        [("MERRA2", "M2I1NXASM.5.12.4")],
#         # "t_air_max":    [("MERRA2", "M2I1NXASM.5.12.4")],
#     }

#     diagnostics = { # label          # lat      # lon
#                     "water":	    (29.44977,	30.58215),
#                     "desert":	    (29.12343,	30.51222),
#                     "agriculture":	(29.32301,	30.77599),
#                     "urban":	    (29.30962,	30.84109),
#                     }

#     folder = r"/Users/hmcoerver/Downloads/pywapor_test"
#     latlim = [28.9, 29.7]
#     lonlim = [30.2, 31.2]
#     timelim = [datetime.date(2020, 6, 25), datetime.date(2020, 7, 30)]
#     example_source = ("MODIS", "MOD13Q1.061")
#     bin_length = 4

#     bins = time_bins(timelim, bin_length)
#     dss = collect_sources(folder, sources, latlim, lonlim, [bins[0], bins[-1]])

#     ds = main(dss, sources, example_source, bins, folder, 
#                 diagnostics = None)

#     if diagnostics:
#         _ = main(dss, sources, example_source, bins, folder, diagnostics = diagnostics)