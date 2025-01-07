import os
import xarray as xr
from osgeo import gdal
import datetime
from pywapor.general.processing_functions import save_ds, remove_ds, open_ds
from pywapor.enhancers.dms.pyDMS import DecisionTreeSharpener
import xml.etree.ElementTree as ET
import pandas as pd
from pywapor.general.logger import log, adjust_logger
import matplotlib.pyplot as plt
import numpy as np
import pywapor
import re
from pywapor.general.performance import performance_check
from mpl_toolkits.axes_grid1 import make_axes_locatable

def gdal_stop_warnings_and_raise_errors():
    # https://gis.stackexchange.com/questions/43404/how-to-detect-a-gdal-ogr-warning/68042
    class GdalErrorHandler(object):
        def __init__(self):
            self.err_level=gdal.CE_None
            self.err_no=0
            self.err_msg=''
        def handler(self, err_level, err_no, err_msg):
            self.err_level=err_level
            self.err_no=err_no
            self.err_msg=err_msg
    err=GdalErrorHandler()
    handler=err.handler
    gdal.PushErrorHandler(handler)
    gdal.UseExceptions()

gdal_stop_warnings_and_raise_errors()

def highres_inputs(workdir, temporal_hr_input_data, static_hr_input_data, filter_times = None):
    """Create VRT files with all highres inputs per datetime.

    Parameters
    ----------
    workdir : str
        Folder in which to store the VRT files.
    temporal_hr_input_data : list
        Paths to the input highres data.
    static_hr_input_data : list
        Paths to static highres data.

    Returns
    -------
    list
        Paths to VRT files.
    """

    if not os.path.exists(workdir):
        os.makedirs(workdir)

    # Find file with smallest pixels
    all_files = temporal_hr_input_data + static_hr_input_data
    template = all_files[np.argmin([abs(geot[1]) * abs(geot[5]) for fn, geot in [(x, gdal.Open(x).GetGeoTransform()) for x in all_files]])]

    # Determine bounds and resolution
    template_ds = gdal.Open(template)
    geot = template_ds.GetGeoTransform()
    xres = abs(geot[1])
    yres = abs(geot[5])
    bounds = (geot[0], geot[3] + template_ds.RasterYSize * geot[5], geot[0] + template_ds.RasterXSize * geot[1], geot[3])
    template_ds = None

    buildvrt_options = gdal.BuildVRTOptions(outputBounds = bounds, xRes = xres, yRes = yres, separate = True, bandList=[1])
    output_vrt = os.path.join(workdir, "highres_input_template.vrt")
    if os.path.isfile(output_vrt):
        try:
            os.remove(output_vrt)
        except PermissionError:
            log.info(f"--> Unable to delete temporary file `{output_vrt}`.")

    ds = gdal.BuildVRT(output_vrt, temporal_hr_input_data + static_hr_input_data, options = buildvrt_options)
    ds.FlushCache()
    ds = None

    ds = xr.open_dataset(re.findall(r"NETCDF:(.*):.*", temporal_hr_input_data[0])[0])
    ds = ds.sortby("time")
    if not isinstance(filter_times, type(None)):
        times = ds["time"].sel(time = ds["time"].isin(filter_times))
    else:
        times = ds["time"]
    ds = ds.close()

    with open(output_vrt, 'r') as f:
        data = f.read()

    fps = list()
    for time_index, dt in enumerate(times.values):
        tree = ET.fromstring(data)
        for x in tree.findall("VRTRasterBand"):
            source_node = x.findall("ComplexSource")[0]
            fn_node = source_node.findall("SourceFilename")[0]
            if fn_node.text in temporal_hr_input_data:
                band_node = source_node.findall("SourceBand")[0]
                band_node.text = str(time_index + 1)

        dt_str = pd.to_datetime(dt).strftime("%Y%m%d_%H%M%S_%f")
        fp_out = output_vrt.replace("_template.vrt", f"_{dt_str}.vrt")
        with open(fp_out, 'wb') as output:
            output.write(ET.tostring(tree))
        fps.append(fp_out)

    try:
        os.remove(output_vrt)
    except PermissionError:
        log.info(f"--> Unable to delete temporary file `{output_vrt}`.")

    return fps, times

def lowres_inputs(workdir, temporal_lr_input_data):
    """Create VRT files with all lowres inputs per datetime.

    Parameters
    ----------
    workdir : str
        Folder in which to store the VRT files.
    temporal_hr_input_data : list
        Paths to the input lowres data.

    Returns
    -------
    list
        Paths to VRT files.
    """

    if not os.path.exists(workdir):
        os.makedirs(workdir)

    ds = xr.open_dataset(re.findall(r"NETCDF:(.*):.*", temporal_lr_input_data)[0], chunks = "auto")
    ds = ds.sortby("time")
    time_da = ds["time"]
    times = [x.strftime("%Y%m%d_%H%M%S_%f") for x in pd.to_datetime(time_da.values)]
    ds = ds.close()

    fps = list()
    for time_index, dt in enumerate(times):
        buildvrt_options = gdal.BuildVRTOptions(resolution = "highest", bandList = [time_index + 1])
        output_vrt = os.path.join(workdir, f"lowres_input_{dt}.vrt")
        if os.path.isfile(output_vrt):
            try:
                os.remove(output_vrt)
            except PermissionError:
                log.info(f"--> Unable to delete temporary file `{output_vrt}`.")
        ds = gdal.BuildVRT(output_vrt, [temporal_lr_input_data], options = buildvrt_options)
        ds.FlushCache()
        ds = None
        fps.append(output_vrt)

    return fps, time_da

def preprocess(ds, dt):
    """Adds a time dimension to `ds`. Used internally by `sharpen` when calling `xr.open_mfdataset` to create
    a stacking dimension.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.

    Returns
    -------
    xr.Dataset
        Dataset with added time dimension.
    """
    ds["Band1"] = ds["Band1"].expand_dims({"time": 1}).assign_coords({"time": [dt]})
    return ds

def plot_sharpening(lr_fn, hr_fn, var, workdir):
    """Create a plot of a sharpened image.

    Parameters
    ----------
    lr_fn : str
        Path to lowres file.
    hr_fn : str
        Path to highres file.
    var : str
        Name of var.
    workdir : str
        Path to folder in which to save graphs.
    """

    if not os.path.exists(workdir):
        os.makedirs(workdir)

    fig, axs = plt.subplots(1,2, figsize=(10,6), dpi=300, sharex=True, sharey=True)
    
    # NOTE https://github.com/corteva/rioxarray/issues/580
    lr = xr.open_dataset(lr_fn).isel(band=0).band_data
    hr = xr.open_dataset(hr_fn).Band1
    lr_clipped = lr.rio.clip_box(*hr.rio.bounds()) * 0.0001

    lr_clipped.plot(ax = axs[0], add_colorbar=False, vmin = 280, vmax = 320)
    im1 = hr.plot(ax = axs[1], add_colorbar=False, vmin = 280, vmax = 320)

    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax, orientation='vertical', label = var)

    dt = datetime.datetime.strptime(os.path.split(hr.encoding["source"])[-1], "highres_output_%Y%m%d_%H%M%S.nc")
    fig.suptitle(dt.strftime("%Y-%m-%d %H:%M:%S"))
    axs[0].set_facecolor("lightgray")
    axs[1].set_facecolor("lightgray")
    axs[0].grid()
    axs[1].grid()
    axs[0].set_title(f"{lr.rio.resolution()[0]:.4f}° x {lr.rio.resolution()[1]:.4f}°")
    axs[1].set_title(f"{hr.rio.resolution()[0]:.4f}° x {hr.rio.resolution()[1]:.4f}°")
    axs[1].set_ylabel("")
    fig.savefig(os.path.join(workdir, dt.strftime("%Y%m%d_%H%M%S.jpg")))

    lr = lr.close()
    hr = hr.close()

def sharpen(dss, var, folder, *args, make_plots = False, req_vars = ['nmdi', 'bsi', 'mndwi',
                'vari_red_edge', 'psri', 'nir', 'green']):
    """Thermal sharpen datasets.

    Parameters
    ----------
    dss : dict
        Keys are variable names, values are xr.Datasets which contain the lowres and highres input data.
    var : str
        Variable name of the lowres input data (usually `lst`).
    folder : str
        Path to folder in which to store (intermediate) files.
    make_plots : bool, optional
        Whether or not to create plots, by default False.
    vars_for_sharpening : list, optional
        Variables to use as sharpening features, by default ['nmdi', 'bsi', 'mndwi', 'cos_solar_zangle', 'vari_red_edge', 'psri', 'nir', 'green', 'z', 'aspect', 'slope'].

    Returns
    -------
    dict
        Keys are variable names, values are (sharpened) datasets.
    """
    workdir = os.path.join(folder, "DMS")
    fp = os.path.join(workdir, f"{var}_i.nc")

    if not os.path.isfile(fp):
        # Open unopened netcdf files.
        dss = {**{k: open_ds(v) for k, v in dss.items() if isinstance(v, str)}, 
                **{k:v for k,v in dss.items() if not isinstance(v, str)}}

        if 'cos_solar_zangle' in req_vars and not 'cos_solar_zangle' in dss.keys():
            dss = get_cos_solar_zangle(dss, var, folder)
            remove_cos_solar_zangle = True
        else:
            remove_cos_solar_zangle = False

        out_fns = list()

        temporal_lr_input_data = f"NETCDF:{dss[var].encoding['source']}:{var}"
        temporal_hr_input_data = [f"NETCDF:{y.encoding['source']}:{x}" for x, y in dss.items() if "time" in y.dims and x in req_vars]
        static_hr_input_data = [f"NETCDF:{y.encoding['source']}:{x}" for x, y in dss.items() if "time" not in y.dims and x in req_vars]

        lowResFiles, lowResTimes = lowres_inputs(workdir, temporal_lr_input_data)
        highResFiles, highResTimes = highres_inputs(workdir, temporal_hr_input_data, static_hr_input_data, filter_times = lowResTimes)

        assert np.all([x == y for x,y in zip(lowResTimes, highResTimes)])

        missing_vars = [x for x in req_vars if x not in dss.keys()]
        if len(missing_vars) > 0:
            log.info(f"--> Sharpening {len(highResFiles)} `{var}` images, without `{'` and `'.join(missing_vars)}` features.").add()
        else:
            log.info(f"--> Sharpening {len(highResFiles)} `{var}` images.").add()

        for i, (highResFilename, lowResFilename) in enumerate(zip(highResFiles, lowResFiles)):
            fp_, fn = os.path.split(highResFilename)

            lbl = f"({i+1}/{len(highResFiles)}) Sharpening `{os.path.split(lowResFilename)[-1]}` with `{fn}`."

            _, correctedImage = thermal_sharpen(highResFilename, lowResFilename, label = lbl)

            fp_out = os.path.join(fp_, fn.replace("input", "output").replace(".vrt", ".nc"))
            ds = gdal.Translate(fp_out, correctedImage)
            ds.FlushCache()
            ds = None

            if make_plots:
                plot_sharpening(lowResFilename, fp_out, var, workdir)

            out_fns.append(fp_out)

            try:
                os.remove(highResFilename)
            except PermissionError:
                log.info(f"--> Unable to delete temporary file `{highResFilename}`.")

            try:
                os.remove(lowResFilename)
            except PermissionError:
                log.info(f"--> Unable to delete temporary file `{lowResFilename}`.")

        log.sub()

        dss_ = [preprocess(open_ds(x), dt) for x, dt in zip(out_fns, highResTimes.values)]
        ds_ = xr.concat(dss_, dim = "time").sortby("time")
        ds_ = ds_.rename_dims({"lat": "y", "lon": "x"})
        ds_ = ds_.transpose("time", "y", "x")
        ds_ = ds_.rename_vars({"crs": "spatial_ref", "Band1": var, "lat": "y", "lon": "x"})
        ds_ = ds_.rio.write_grid_mapping("spatial_ref")
        ds_ = ds_.sortby("y", ascending = False)
        ds_ = ds_.sortby("x")
        ds_.attrs = {}

        ds = save_ds(ds_, fp, encoding = "initiate", label = f"Merging sharpened `{var}` files.")

        ds_ = ds_.close()

        for x in dss_:
            remove_ds(x)
        
        if 'cos_solar_zangle' in dss.keys() and remove_cos_solar_zangle:
            remove_ds(dss['cos_solar_zangle'])
            _ = dss.pop("cos_solar_zangle")

    dss[var] = fp

    return dss

@performance_check
def thermal_sharpen(highres_fn, lowres_fn):
    """Sharpen a single lowres file using a single highres file (in which each band contains one feature).

    Parameters
    ----------
    highres_fn : str
        Path to highres input.
    lowres_fn : str
        Path to lowres input.

    Returns
    -------
    tuple
        `gdal.Dataset`s of the residual and the corrected image.
    """

    commonOpts = {
                "highResFiles":                     [highres_fn],
                "lowResFiles":                      [lowres_fn],
                "lowResQualityFiles":               [],
                "lowResGoodQualityFlags":           [],
                "cvHomogeneityThreshold":           0.0,
                "movingWindowSize":                 0,
                "disaggregatingTemperature":        True,
            }

    dtOpts = {
            "perLeafLinearRegression":              True,
            "linearRegressionExtrapolationRatio":   0.25
            }

    opts = commonOpts.copy()
    opts.update(dtOpts)
    disaggregator = DecisionTreeSharpener(**opts)

    disaggregator.trainSharpener()
    downscaledFile = disaggregator.applySharpener(highres_fn, lowres_fn)
    residualImage, correctedImage = disaggregator.residualAnalysis(downscaledFile, lowres_fn, lowResQualityFilename = True)

    return residualImage, correctedImage

def get_cos_solar_zangle(dss, var, folder):
    """Calculate the cosine solar zenith angle.

    Parameters
    ----------
    dss : dict
        Keys are variables, values are datasets.
    var : str
        Variable from which the time is used to calculate `cos_solar_zangle` at.
    folder : str
        Path to folder to store files.

    Returns
    -------
    dict
        Keys are variables, values are datasets.
    """
    reqs = ["slope", "aspect", var]
    checks = [(x in dss.keys()) and (x in dss[x].data_vars) for x in reqs]
    if np.all(checks):
        lat_deg = xr.DataArray(dss["slope"]["y"].values, coords = {"y":dss["slope"]["y"]}).rename("lat").chunk("auto")
        lat = pywapor.et_look_v2_v3.solar_radiation.latitude_rad(lat_deg)
        slope = dss["slope"]["slope"]
        aspect = dss["aspect"]["aspect"]
        doy = dss[var]["time"].dt.strftime("%j").astype(int).chunk("auto")
        dtime = (dss[var]["time"].dt.strftime("%H").astype(int) + dss[var]["time"].dt.strftime("%M").astype(int)/60.).chunk("auto")
        sc = pywapor.et_look_v2_v3.solar_radiation.seasonal_correction(doy)
        decl = pywapor.et_look_v2_v3.solar_radiation.declination(doy)
        ha = pywapor.et_look_v2_v3.solar_radiation.hour_angle(sc, dtime)
        ds = pywapor.et_look_v2_v3.solar_radiation.cosine_solar_zenith_angle(ha, decl, lat, slope, aspect).rename('cos_solar_zangle').to_dataset()
        ds = ds.rio.write_crs(aspect.rio.crs)
        ds = save_ds(ds, os.path.join(folder, "cos_solar_zangle_i.nc"), label = "Calculating `cos_solar_zangle`.")
        dss['cos_solar_zangle'] = ds
    else:
        log.warning(f"--> Couldn't calculate `cos_solar_zangle`, `{'` and `'.join([x for x, check in zip(reqs,checks) if not check])}` is missing.")

    return dss

if __name__ == "__main__":

    folder = r"/Users/hmcoerver/Local/Test_Rahma_error"

    adjust_logger(True, folder, "INFO")

    dss = {
        'lst':      open_ds(r"/Users/hmcoerver/Local/Test_Rahma_error/_instantaneous/lst_i.nc"),
        'ndvi':     open_ds(r"/Users/hmcoerver/Local/Test_Rahma_error/_instantaneous/ndvi_i.nc"),
        'bsi':      open_ds(r"/Users/hmcoerver/Local/Test_Rahma_error/_instantaneous/bsi_i.nc"),
        'mndwi':    open_ds(r"/Users/hmcoerver/Local/Test_Rahma_error/_instantaneous/mndwi_i.nc"),
        'green':    open_ds(r"/Users/hmcoerver/Local/Test_Rahma_error/_instantaneous/green_i.nc"),
        }
    
    var = "lst"
    args = None

    req_vars = ['ndvi', 'bsi', 'mndwi', 'green']
    make_plots = False

    # sharpen(dss, var, folder, *args, make_plots = False, 
    #         req_vars = ['ndvi', 'bsi', 'mndwi', 'green', 'z'])
    