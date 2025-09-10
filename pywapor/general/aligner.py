"""
Functions to prepare input for `pywapor.se_root`, more specifically to
interpolate various parameters in time to match with land-surface-temperature
times. 
"""
from pywapor.general.processing_functions import save_ds, open_ds, remove_ds, log_example_ds, func_from_string
from pywapor.general.reproject import align_pixels
from pywapor.enhancers.apply_enhancers import apply_enhancer
from pywapor.general.logger import log
import os
import numpy as np
import xarray as xr
from itertools import chain
import pywapor.general.levels as levels
from pywapor.enhancers.smooth.whittaker import whittaker_smoothing

def main(dss, sources, folder, general_enhancers, example_t_vars = ["lst"]):
    """Aligns the datetimes in de `dss` xr.Datasets with the datetimes of the 
    dataset with variable `example_t_var`.

    Parameters
    ----------
    dss : dict
        Keys are tuples of (`source`, `product_name`), values are xr.Dataset's 
        or paths (str) to netcdf files, which will be aligned along the time dimensions.
    sources : dict
        Configuration for each variable and source.
    folder : str
        Path to folder in which to store (intermediate) data.
    general_enhancers : list
        Functions to apply to the xr.Dataset before creating the final
        output, by default "default".
    example_t_var : list, optional
        Which variables to align the other datasets to in the time dimension, by default ["lst"].

    Returns
    -------
    xr.Dataset
        Dataset in which all variables have been interpolated to the same times.
    """
    chunks = {"time": -1, "y": 500, "x": 500}

    # Open unopened netcdf files.
    dss = {**{k: open_ds(v) for k, v in dss.items() if isinstance(v, str)}, 
            **{k:v for k,v in dss.items() if not isinstance(v, str)}}

    # Determine final output path.
    final_path = os.path.join(folder, "se_root_in.nc")

    # Make list to store intermediate datasets.
    dss2 = dict()

    # Make inventory of all variables.
    variables = [x for x in np.unique(list(chain.from_iterable([ds.data_vars for ds in dss.values()]))).tolist() if x in sources.keys()]
    variables = example_t_vars + [x for x in variables if x not in example_t_vars]

    # Create variable to store times to interpolate to.
    example_time = None

    cleanup = list()

    # Loop over the variables
    for var in variables:

        dst_path = os.path.join(folder, "_instantaneous",f"{var}_i.nc")

        if os.path.isfile(dst_path):
            ds = open_ds(dst_path, chunks = chunks)
            if var in example_t_vars:
                if isinstance(example_time, type(None)):
                    example_time = ds["time"]
        else:
            config = sources[var]
            spatial_interp = config["spatial_interp"]
            temporal_interp = config["temporal_interp"]

            if isinstance(temporal_interp, dict):
                kwargs = temporal_interp.copy()
                temporal_interp = kwargs.pop("method")
            else:
                kwargs = {}

            # Align pixels of different products for a single variable together.
            dss_part = [ds[[var]] for ds in dss.values() if var in ds.data_vars]
            if len(dss_part) > 1:
                log.info(f"--> Spatially aligning {len(dss_part)} `{var}` products together.").add()
            dss1, temp_files1 = align_pixels(dss_part, folder, spatial_interp, fn_append = "_step1")
            if len(dss_part) > 1:
                log.sub()
            cleanup.append(temp_files1)

            # Combine different source_products (along time dimension).
            ds = xr.combine_nested(dss1, concat_dim = "time").chunk(chunks).sortby("time").squeeze()

            if var in example_t_vars:
                if "time" not in ds[var].dims:
                    ds[var] = ds[var].expand_dims("time").transpose("time", "y", "x")
                
                if isinstance(example_time, type(None)):
                    example_time = ds["time"]
                else:
                    example_time = xr.concat([example_time, ds["time"]], dim = "time").drop_duplicates("time").sortby("time")
            else:
                ...

            if "time" in ds[var].dims and not isinstance(temporal_interp, type(None)):
                orig_time_size = ds["time"].size

                if temporal_interp == "whittaker":
                    if (not ds.time.equals(example_time)) and (var not in example_t_vars):
                        new_x = example_time.values
                    else:
                        new_x = None
                        
                    if "weights" in kwargs.keys():
                        ds["sensor"] = xr.combine_nested([xr.ones_like(x.time.astype(int)) * i for i, x in enumerate(dss1)], concat_dim="time").sortby("time")
                        source_legend = {str(i): os.path.split(x.encoding["source"])[-1].replace(".nc", "") for i, x in enumerate(dss1)}
                        ds["sensor"] = ds["sensor"].assign_attrs(source_legend)
                    ds = whittaker_smoothing(ds, var, chunks = chunks, new_x = new_x, **kwargs)
                    ds = ds.rename_vars({f"{var}_smoothed": var})
                else:
                    if not ds.time.equals(example_time):
                        new_coords = xr.concat([ds.time, example_time], dim = "time").drop_duplicates("time").sortby("time")
                        ds = ds.reindex_like(new_coords).chunk(chunks)
                    ds = ds.interpolate_na(dim = "time", method = temporal_interp, **kwargs).sel(time = example_time)

                lbl = f"Aligning times in `{var}` ({orig_time_size}) with `{'` and `'.join(example_t_vars)}` ({example_time.time.size}, {temporal_interp})."
                ds = save_ds(ds, dst_path, chunks = chunks, encoding = "initiate", label = lbl)
                log.add().info(f"> shape: {ds[var].shape}, kwargs: {list(kwargs.keys())}.").sub()
            else:
                lbl = f"Saving spatially aligned `{var}` data without temporal interpolation."
                ds = save_ds(ds, dst_path, chunks = chunks, encoding = "initiate", label = lbl)

        # Set dimension order.
        ds = ds.transpose(*sorted(list(ds.sizes.keys()), key = lambda e: ["time", "y", "x"].index(e)))

        dss2[var] = ds

    # Apply variable specific enhancers
    variables = levels.find_setting(sources, "variable_enhancers")
    for var in variables:
        for func in sources[var]["variable_enhancers"]:
            if isinstance(func, str):
                func = func_from_string(func)
            dss2 = func(dss2, var, folder)

    # TODO this info needs to be moved to `sources`.
    vars_for_sharpening = ['nmdi', 'bsi', 'mndwi', 'cos_solar_zangle', 'vari_red_edge', 'psri', 'nir', 'green']
    for x in vars_for_sharpening:
        if x in dss2.keys():
            _ = dss2.pop(x)

    # Align all the variables together.
    example_source = levels.find_setting(sources, "is_example", min_length = 1)
    if len(example_source) == 1:
        example_ds = dss[example_source[0]]
        log_example_ds(example_ds)
    elif len(example_source) == 0:
        example_ds = None
    else:
        log.warning("--> Multiple example datasets found, selecting lowest resolution.")
        example_ds = None

    # Open unopened netcdf files.
    dss2 = {**{k: open_ds(v) for k, v in dss2.items() if isinstance(v, str)}, 
            **{k:v for k,v in dss2.items() if not isinstance(v, str)}}

    spatial_interps = [sources[list(x.data_vars)[0]]["spatial_interp"] for x in dss2.values()]
    dss3, temp_files3 = align_pixels(dss2.values(), folder, spatial_interps, example_ds, fn_append = "_step2")
    cleanup.append(temp_files3)

    # Merge everything.
    ds = xr.merge(dss3, compat='no_conflicts')

    # Apply general enhancers.
    for func in general_enhancers:
        ds, label = apply_enhancer(ds, None, func)

    while os.path.isfile(final_path):
        final_path = final_path.replace(".nc", "_.nc")

    if ds.time.size == 0:
        log.warning("--> No valid data created (ds.time.size == 0).")
        return ds

    ds = save_ds(ds, final_path, encoding = "initiate",
                    label = f"Creating merged file `{os.path.split(final_path)[-1]}`.")

    for nc in list(chain.from_iterable(cleanup)):
        remove_ds(nc)

    return ds

# if __name__ == "__main__":

#     import numpy as np
#     import glob

#     dss = {
#         ("SENTINEL2", "S2MSI2A"): r"/Users/hmcoerver/Local/test_data/SENTINEL2/S2MSI2A.nc",
#         ("SENTINEL3", "SL_2_LST___"): r"/Users/hmcoerver/Local/test_data/SENTINEL3/SL_2_LST___.nc",
#         # ("VIIRSL1", "VNP02IMG"): r"/Users/hmcoerver/Local/test_data/VIIRSL1/VNP02IMG.nc",
#     }
#     example_source = ("SENTINEL2", "S2MSI2A")

#     sources = {
#             "ndvi": {"spatial_interp": "nearest", "temporal_interp": "linear"},
#             "lst": {"spatial_interp": "nearest", "temporal_interp": "linear"},
#             "bt": {"spatial_interp": "nearest", "temporal_interp": "linear"},
#                 }

#     # example_source = ("source1", "productX")
#     folder = r"/Users/hmcoerver/Local/test_data"
#     enhancers = list()
#     example_t_vars = ["lst"]

#     chunks = (1, 500, 500)

#     for fp in glob.glob(os.path.join(folder, "*.nc")):
#         os.remove(fp)

#     from pywapor.general.logger import log, adjust_logger
#     adjust_logger(True, folder, "INFO")

#     ds = main(dss, sources, example_source, folder, enhancers, example_t_vars = ["lst"])