## pyWaPOR

![downloads](https://img.shields.io/pypi/dw/pywapor) [![version](https://img.shields.io/pypi/v/pywapor)](https://pypi.org/project/pywapor/)

This repository contains a Python implementation of the algorithm used to generate the [WaPOR](http://www.fao.org/in-action/remote-sensing-for-water-productivity/en/) [datasets](https://wapor.apps.fao.org/home/WAPOR_2/1). It can be used to calculate evaporation, transpiration and biomass production maps.

### Installation

Its recommended to install in a new [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html).

```bash
conda create -n my_pywapor_env -c conda-forge pywapor
```

### Contact

For technical questions or bug-reports, please open an [Issue here](https://github.com/un-fao/fao-wetl-pywapor/issues). For more general questions please contact Bert Coerver at [bert.coerver@fao.org](mailto:bert.coerver@fao.org) or the WaPOR team at [wapor@fao.org](mailto:wapor@fao.org).

### Usage

Run the model for a selected bounding-box and period of time:

```python
import pywapor

project_folder = r"/my_first_ETLook_run/"
bb = [30.2, 28.9, 31.2, 29.7] # [xmin, ymin, xmax, ymax]
period = ["2021-07-01", "2021-07-03"]

# Set up a project.
project = pywapor.Project(project_folder, bb, period)

# Load a configuration.
project.load_configuration(name = "WaPOR3_level_2")

# Set up required accounts.
project.set_passwords()

# Download the input data.
datasets = project.download_data()

# Run the models.
se_root_in = project.run_pre_se_root()
se_root = project.run_se_root()

et_look_in = project.run_pre_et_look()
et_look = project.run_et_look()
```

To see a summary of your configuration, checkout the `summary` attribute of the configuration. For more detailed information, check the `full`, `se_root` and `et_look` attributes:

```python
print(project.configuration.summary)
print(project.configuration.full)
print(project.configuration.se_root)
print(project.configuration.et_look)
```

Making configurations yourself can be done by passing a name to `Project.load_configuration` (as done above), by passing a summary or by loading a configuration from a JSON-file:

```python
summary = {
    # Define which products to use.
    'elevation': {'COPERNICUS.GLO30'},
    'meteorological': {'GEOS5.tavg1_2d_slv_Nx'},
    'optical': {'SENTINEL2.S2MSI2A_R20m'},
    'precipitation': {'CHIRPS.P05'},
    'solar radiation': {'ERA5.sis-agrometeorological-indicators'},
    'statics': {'STATICS.WaPOR3'},
    'thermal': {'VIIRSL1.VNP02IMG'},
    # Use se_root output as soil moisture data.
    'soil moisture': {'FILE:{folder}{sep}se_root_out*.nc'},
    # Define which product to reproject the other products to.
    '_EXAMPLE_': 'SENTINEL2.S2MSI2A_R20m', 
    # Define any special functions to apply to a specific variable.
    '_ENHANCE_': {"bt": ["pywapor.enhancers.dms.thermal_sharpener.sharpen"],},
    # Choose which products should be gapfilled.
    '_WHITTAKER_': {
        'SENTINEL2.S2MSI2A_R20m': {'lmbdas': 1000.0, 'method': 'whittaker'}, 
        'VIIRSL1.VNP02IMG': {'a': 0.85, 'lmbdas': 1000.0, 'method': 'whittaker'}},
    }

project.load_configuration(summary = summary)
```

Save a configuration to a JSON-file like this:

```python
project.configuration.to_json("/path/to/my/configuration.json")
```

Load a configuration from a JSON-file:

```python
project.load_configuration(json = "/path/to/my/configuration.json")
```

Changing model parameters (or entire variables) can be done in between the `project.run_pre_se_root()` and `project.run_se_root()` steps (same goes for `et_look`) by making changes to the `xarray.Dataset` returned by `project.run_pre_se_root` (and stored at `project.se_root_in`):

```python
print(se_root_in["r0_bare"].values)
>>> 0.38

se_root_in["r0_bare"] = 0.32

print(se_root_in["r0_bare"].values)
print(project.se_root_in["r0_bare"].values)
>>> 0.32
>>> 0.32
```

### Documentation
Go [here](https://www.fao.org/aquastat/py-wapor/) for the full pyWaPOR documentation.


### WaPOR Documentation
###### WaPOR v3
<ul>
<li><a href="https://bitbucket.org/cioapps/wapor-et-look/wiki/Home">WaPOR-ETLook Online Manual (v3)</a></li>
</ul>

###### WaPOR v2

<ul>
<li><a href="https://storage.googleapis.com/fao-cog-data/WAPOR_DOCS/FRAME_ET_v2_data_manual_finaldraft_v2.2.pdf">WaPOR-ETLook Data Manual (v2)</a></li>

<li><a href="https://storage.googleapis.com/fao-cog-data/WAPOR_DOCS/FRAME_NPP_v2_data_manual_finaldraft_v2.2.pdf">WaPOR-Biomass Data Manual (v2)</a></li>
</ul>

###### WaPOR v1

<ul>
<li><a href="https://storage.googleapis.com/fao-cog-data/WAPOR_DOCS/20190522_V1_WaPOR_v_1_Data_Manual_Evapotranspiration.pdf">WaPOR-ETLook Data Manual (v1)</a></li>
</ul>

### Acknowledgments
The methodology for WaPOR was developed by the FRAME1 consortium, consisting of eLEAF (lead), VITO, ITC, University of Twente and Waterwatch foundation, commissioned by and in partnership with the Land and Water Division of FAO. The method for calculating evapotranspiration is based on the ETLook model developed by eLEAF in 2010. The method for calculating total biomass production is based on the C-Fix model. 

The code in the pywapor.et_look_v2_v3 module of this repository, containing all core physical functions used by ETLook, was written by Henk Pelgrum (eLEAF) and Rutger Kassies (eLEAF). The remaining modules have been developed by Bert Coerver (FAO), Tim Hessels (WaterSat), and, in the framework of the ESA-funded ET4FAO project, Radoslaw Guzinski (DHI-GRAS), Hector Nieto (Complutig) and Laust Faerch (DHI-GRAS).

### Release Notes

#### 3.5.0 (2024-03-19)
<br>
<ul>
<li> Restructured the overall workflow, there are now clear sequential phases instead of a nested process.</li>
<li> Only one configuration needs to be made, instead of two separate ones for et_look and se_root.</li>
<li> There is now much more recycling of previously created files and API responses are cached on disk wherever possible, making rerunning after a crash or configuration change much faster.</li>
<li> Quickly set up the passwords needed for the configuration you're running using the logged/printed instructions.</li>
<li> The process prior to the start of downloading VIIRS data is now roughly 3 times faster.</li>
<li> VIIRS data is now downloaded using OPeNDAP.</li>
<li> Downloading of GEOS5 data is now chunked, making it more stable when downloading long time-series.</li>
<li> Fixed a bug in the thermal sharpener that caused it to return only missing data in some cases.</li>
<li> The static variables are now available globally.</li>
</ul>

#### 3.4.0 (2023-09-15)
<br>
<ul>
    <li> Rewritten code to download and process VIIRSL1 data, which is now available from an S3 bucket.</li>
    <li> Rewritten code to download PROBA-V data, which now comes from the new Terrascope platform (the old portal was deprecated). </li>
    <li> Rewritten Sentinel downloader, which now uses the new Copernicus Data Space Ecosystem. Most importantly this means that all images are available instantly (so no more tedious requesting from the Long Term Archive for older scenes).
    <li> Projecting of curvilinear data (VIIRSL1 and Sentinel-3) is now done using gdal-warp. </li>
    <li> Calculation of terrain slope and aspect is now done using gdal-dem. </li>
    <li> It is now possible turn off SSL-verification when downloading certain products. Run 'import os; os.environ["PYWAPOR_VERIFY_SSL"] = "NO"' to do so. </li>
    <li> Simplified installation requirements. </li>
    <li> Available options in `pywapor.pre_et_look.main` for `sources` are now "level_1", "level_2", "level_3", "level_2_v3" (new default value) and "level_3_v3". The `bin_length' is now set to 1 by default (was `"DEKAD"`).
    <li> Various other smaller bug fixes.</li>
</ul>

#### 3.3.0 (2023-04-05)
<br>
<ul>
    <li> Option to smooth and interpolate data with a Whittaker smoother.</li>
    <li> Downloading of Sentinel-3 data is now faster.
    <li> Fixed an issue that could result in an incorrect scale-factor being applied to Sentinel-2 images.</li>
</ul>

#### 3.2.0 (2023-01-04)
<br>
<ul>
    <li> Full Landsat support, automatically download and process Landsat (5-9) data.</li>
    <li> Fixed a bug that caused some MODIS data to be missing inside the selected boundingbox. </li>
    <li> You can now re-enter a password in case you've provided an incorrect one. </li>
    <li> Bugfixes (including more Windows specific ones). </li>
    <li> Updated weights to calculate albedo for Landsat and Sentinel. </li>
</ul>

#### 3.1.0 (2022-09-22)
<br>
<ul>
    <li> Added a thermal sharpening algorithm (<a href = https://github.com/radosuav/pyDMS>pyDMS</a>) to increase LST resolution.</li>
    <li> Now, when after several failed attempts to download a variable, the code will continue processing other variables. </li>
    <li> Improved cloud-masking for VIIRSL1. </li>
    <li> Bugfixes (including several Windows specific ones). </li>
    <li> More information in the log. </li>
</ul>

#### 3.0.0 (2022-08-31)
<br>
<ul>
    <li> Bugfixes. Most noteably, server side errors when downloading data are now handeled better, i.e. collect tools will retry several times when a download fails, but many other smaller issues have been fixed too.</li>
    <li> Performance improvements, mostly due to fewer reprojections. </li>
    <li> Better logging. The logs from SENTINEL and ERA5 are now directed to seperate channels and logs now show peak-memory-usage for critical calculation steps.
    <li> `et_look` and `se_root` now accept a `chunks` keyword to adjust the chunksizes at which the calculations are done. Increase them if you have a lot of RAM available, decrease them for slower calculations but with less memory usage.</li>
    <li> Support for WaPOR v3 methodology. Choose `et_look_version = "v3"` and `se_root_version = "v3"` when running the respective models (`et_look` and `se_root`).</li>
    <li> Default configurations for WaPOR v3 input datasets, i.e. choose `sources = "level_2_v3"` when running `pre_et_look` or `pre_se_root`.
    <li> New collect functions for COPERNICUS DEM. </li>
    <li> The data structure for STATICS is now consistent with the other products. </li>
</ul>

#### 2.6.0 (2022-08-04)
<br>
<ul>
    <li> New collect functions for VIIRS (Level-1), SENTINEL-2, SENTINEL-3 and (ag)ERA5.</li>
    <li> pyWaPOR now works with Python versions greater than 3.8.
</ul>

#### 2.5.0 (2022-06-23)
<br>
<ul>
    <li> Rewritten collect tools.</li>
    <li> The entire workflow now works with netCDF.</li>
    <li> All the netCDF files are formatted to support the <a href = "https://corteva.github.io/rioxarray/stable/getting_started/getting_started.html">rio-acccessor</a>.</li>
</ul>

#### 2.4.2 (2022-04-26)
<br>
<ul>
    <li> New biomass module and NPP calculation.</li>
</ul>

#### 2.4.1 (2022-03-11)
<br>
<ul>
    <li> NetCDF files are now compressed when saved to disk.</li>
    <li> Calculation of Total Biomass Production is now turned on by default.</li>
    <li> It is no longer required to provide <b>all</b> input variables to et_look,
    the model will calculate as many variables as possible with the given data. For example,
    if you are only interested in acquiring interception rates, it would now suffice to only prepare ndvi and p (precipitation) data with pre_et_look.</li>
    <li> et_look now automatically generates an interactive network graph visualising the executed computation steps.</li>
</ul>

#### 2.4.0 (2022-02-03)
<br>
<ul>
    <li> Easily apply your own functions to data, i.e. use your own custom filters, gap-fillers etc.</li>
    <li> Side-load your own data, i.e. easily incorporate you own datasets.</li>
    <li> Added functions to process Landsat Level-2 to “ndvi”, “lst” and “r0”.</li>
    <li> Data is now stored and processed as netCDF (using xarray and dask).</li>
    <li> Calculations in et_look() and se_root() are now done in chunks, instead of using a for-loop.</li>
    <li> Some previously constant parameters now have spatial variability.</li>
    <li> Improved logging.</li>
    <li> Download functions now show progress and download-speed.</li>
    <li> MODIS datasets switched from v6.0 (decommissioned soon) to v6.1.</li>
    <li> The lapse-rate correction to temperature data is now more accurate and faster.</li>
    <li> VITO and WAPOR passwords are now checked when entered.</li>
    <li> Other bug-fixes and performance improvements.</li>
</ul>

#### 2.3.0 (2021-11-19)
<br>
<ul> 
    <li>Automatically create input composites before running ETLook.</li>
    <li>Choose composite lengths in number of days or dekads.</li>
    <li>Option to choose which products to use per variable.</li>
    <li>Calculate soil saturation separate from ETLook.</li>
    <li>PROBA-V support for NDVI and Albedo inputs.</li>
    <li>Define diagnostics pixels, for which extra outputs are created (e.g. charts, maps etc.).</li>
    <li>Bug-fixes and performance improvements.</li>
</ul>

### License
[APACHE](https://github.com/un-fao/fao-wetl-pywapor/blob/main/LICENSE)