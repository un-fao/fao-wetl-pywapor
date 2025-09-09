import warnings

import numpy as np
import datetime
import pytest
import rasterio
import os
import glob

from pywapor.general.logger import adjust_logger, log
import importlib



def has_geotransform(ds):
    varis = ds.data_vars
    for var in varis:
        with warnings.catch_warnings(record=True) as w:
            _ = rasterio.open(f"netcdf:{ds.encoding['source']}:{var}")
            if len(w) > 0:
                for warning in w:
                    no_geot = "Dataset has no geotransform, gcps, or rpcs." in str(
                        warning.message
                    )
                    if no_geot:
                        return False
    return True


def strictly_increasing(L):
    return all(x < y for x, y in zip(L, L[1:]))


def strictly_decreasing(L):
    return all(x > y for x, y in zip(L, L[1:]))


SOURCES = {
    "GEOS5.inst3_2d_asm_Nx": [
        "p_air",
        "p_air_0",
        "qv",
        "t_air",
        "t_air_max",
        "t_air_min",
        "u2m",
        "v2m",
        "wv",
    ],
    "GEOS5.tavg1_2d_slv_Nx": [
        "p_air",
        "p_air_0",
        "qv",
        "t_air",
        "t_air_max",
        "t_air_min",
        "to3",
        "u10m",
        "u2m",
        "v10m",
        "v2m",
        "wv",
    ],
    "GEOS5.tavg1_2d_rad_Nx": ["ra_flat"],
    "GEOS5.tavg1_2d_lnd_Nx": ["p"],
    "GEOS5.tavg3_2d_aer_Nx": ["totangstr", "totexttau"],
    "STATICS.WaPOR2": [
        "land_mask",
        "lw_offset",
        "lw_slope",
        "r0_bare",
        "r0_full",
        "rn_offset",
        "rn_slope",
        "rs_min",
        "t_amp",
        "t_opt",
        "vpd_slope",
        "z_obst_max",
        "z_oro",
    ],
    "STATICS.WaPOR3": [
        "lw_offset",
        "lw_slope",
        "rn_offset",
        "rn_slope",
        "t_amp",
        "t_opt",
        "vpd_slope",
    ],
    "MODIS.MOD13Q1.061": ["ndvi"],
    "MODIS.MYD13Q1.061": ["ndvi"],
    "MODIS.MOD11A1.061": ["lst"],
    "MODIS.MYD11A1.061": ["lst"],
    "MODIS.MCD43A3.061": ["r0"],
    "MERRA2.M2I1NXASM.5.12.4": [
        "p_air",
        "p_air_0",
        "qv",
        "t_air",
        "t_air_max",
        "t_air_min",
        "u2m",
        "v2m",
        "wv",
    ],
    "MERRA2.M2T1NXRAD.5.12.4": ["ra_flat"],
    "GLOBCOVER.2009_V2.3_Global": [
        "land_mask",
        "lue_max",
        "lulc",
        "rs_min",
        "z_obst_max",
    ],
    "CHIRPS.P05": ["p"],
    "SRTM.30M": ["aspect", "slope", "z"],
    "ERA5.sis-agrometeorological-indicators": [
        "ra_flat",
        "rh",
        "t_air",
        "t_air_max",
        "t_air_min",
        "t_dew",
        "u",
        "vp",
    ],
    "ERA5.reanalysis-era5-single-levels": [
        "p_air",
        "p_air_0",
        "t_air",
        "t_air_max",
        "t_air_min",
        "t_dew",
        "u",
        "u10m",
        "u2m",
        "v10m",
        "v2m",
        "wv",
    ],
    "SENTINEL2.S2MSI2A_R10m": ["blue", "green", "ndvi", "nir", "qa", "r0", "red"],
    "SENTINEL2.S2MSI2A_R20m": [
        "blue",
        "bsi",
        "coastal_aerosol",
        "green",
        "mndwi",
        "ndvi",
        "nir",
        "nmdi",
        "psri",
        "qa",
        "r0",
        "red",
        "red_edge_703",
        "red_edge_740",
        "red_edge_782",
        "swir1",
        "swir2",
        "vari_red_edge",
    ],
    "SENTINEL2.S2MSI2A_R60m": [
        "blue",
        "bsi",
        "coastal_aerosol",
        "green",
        "mndwi",
        "narrow_nir",
        "ndvi",
        "nir",
        "nmdi",
        "psri",
        "qa",
        "r0",
        "red",
        "red_edge_703",
        "red_edge_740",
        "red_edge_782",
        "swir1",
        "swir2",
        "vari_red_edge",
    ],
    "SENTINEL3.SL_2_LST___": ["lst"],
    "VIIRSL1.VNP02IMG": ["bt"],
    "COPERNICUS.GLO30": ["aspect", "slope", "z"],
    "COPERNICUS.GLO90": ["aspect", "slope", "z"],
    "TERRA.urn:eop:VITO:PROBAV_S5_TOC_100M_COG_V2": ["ndvi", "r0"],
    "LANDSAT.LT05_SR": [
        "blue",
        "green",
        "ndvi",
        "nir",
        "pixel_qa",
        "r0",
        "radsat_qa",
        "red",
        "swir1",
        "swir2",
    ],
    "LANDSAT.LT05_ST": ["lst", "lst_qa"],
    "LANDSAT.LE07_SR": [
        "blue",
        "green",
        "ndvi",
        "nir",
        "pixel_qa",
        "r0",
        "radsat_qa",
        "red",
        "swir1",
        "swir2",
    ],
    "LANDSAT.LE07_ST": ["lst", "lst_qa"],
    "LANDSAT.LC08_SR": [
        "blue",
        "coastal",
        "green",
        "ndvi",
        "nir",
        "pixel_qa",
        "r0",
        "radsat_qa",
        "red",
        "swir1",
        "swir2",
    ],
    "LANDSAT.LC08_ST": ["lst", "lst_qa"],
    "LANDSAT.LC09_SR": [
        "blue",
        "coastal",
        "green",
        "ndvi",
        "nir",
        "pixel_qa",
        "r0",
        "radsat_qa",
        "red",
        "swir1",
        "swir2",
    ],
    "LANDSAT.LC09_ST": ["lst", "lst_qa"],
    'LSASAF.MSG_MDSSFTD': ['diffuse_fraction', 'qa', 'ra_flat'],
    'LSASAF.MSG_MDIDSSF': ['max_nslots_missing',
    'missing_values_percent',
    'ra_flat',
    'weight_missing_values_percent'],

    'LSASAF.MSG_MLST': ['error', 'lst', 'qa'],
    'LSASAF.MSG_MLST-AS': ['lst', 'qa'],
    'LSASAF.MSG_MLST-ASv2': ['lst', 'qa'],
    'LSASAF.MSG_METREF': ['et_ref_24_mm', 'qa'],
    'LSASAF.MSG_MH': ['h_i', 'qa'],
    'LSASAF.MSG_MLE': ['lh_i', 'qa']
}

TIMELIM = {
    # cog
    "STATICS": None,
    "GLOBCOVER": None,
    "CHIRPS": ["2022-03-01", "2022-03-03"],
    "TERRA": ["2020-07-02", "2020-07-09"],
    # opendap
    "SRTM": None,
    "COPERNICUS": None,
    "MODIS": ["2019-03-01", "2019-04-01"],
    "MERRA2": ["2022-03-01", "2022-03-03"],
    "VIIRSL1": ["2022-03-01", "2022-03-01"],
    # opendap.xarray
    "GEOS5": ["2022-03-01", "2022-03-03"],
    # cds
    "ERA5": ["2022-03-01", "2022-03-03"],
    "SENTINEL2": ["2023-03-01", "2023-03-03"],
    "SENTINEL3": ["2023-03-01", "2023-03-03"],

    'LSASAF.MSG_MDIDSSF': ["2023-03-01", "2023-03-03"],
    'LSASAF.MSG_MDSSFTD': ["2023-03-01", "2023-03-03"],

    'LSASAF.MSG_MLST': ["2023-03-01", "2023-03-03"],
    'LSASAF.MSG_METREF': ["2023-03-01", "2023-03-03"],
    'LSASAF.MSG_MH': ["2023-03-01", "2023-03-03"],
    'LSASAF.MSG_MLE': ["2023-03-01", "2023-03-03"],

    'LSASAF.MSG_MLST-AS': ["2023-02-05", "2023-02-06"],
    'LSASAF.MSG_MLST-ASv2': ["2023-02-05", "2023-02-06"],

    "LANDSAT.LT05_SR": ["2005-03-01", "2005-03-12"],
    "LANDSAT.LT05_ST": ["2005-03-01", "2005-03-12"],

    "LANDSAT.LE07_SR": ["2022-03-01", "2022-03-12"],
    "LANDSAT.LE07_ST": ["2022-03-01", "2022-03-12"],
    "LANDSAT.LC08_SR": ["2022-03-01", "2022-03-12"],
    "LANDSAT.LC08_ST": ["2022-03-01", "2022-03-12"],
    "LANDSAT.LC09_SR": ["2022-03-01", "2022-03-12"],
    "LANDSAT.LC09_ST": ["2022-03-01", "2022-03-12"],

}

@pytest.mark.parametrize("source_product", sorted(SOURCES.keys()))
def test_small(source_product, tmp_path):
    test_base(source_product, tmp_path, slicer=slice(1))

@pytest.mark.parametrize("source_product", sorted(SOURCES.keys()))
def test_base(source_product, tmp_path, slicer = slice(None)):
    adjust_logger(True, tmp_path, "INFO")
    log.info(source_product)

    folder = tmp_path
    x = source_product.split(".")
    source = x[0]
    product_name = ".".join(x[1:])

    timelim = TIMELIM.get(source, TIMELIM.get(source_product, None))
    req_vars = SOURCES[source_product][slicer]
    latlim = [29.4, 29.5]
    lonlim = [30.7, 30.8]

    args = {
        "folder": folder,
        "latlim": latlim,
        "lonlim": lonlim,
        "timelim": timelim,
        "product_name": product_name,
        "req_vars": req_vars,
    }

    mod = importlib.import_module(f"pywapor.collect.product.{source}")

    ds = mod.download(**args)

    assert ds.rio.crs.to_epsg() == 4326
    assert "spatial_ref" in ds.coords
    assert strictly_increasing(ds["x"].values)
    assert strictly_decreasing(ds["y"].values)
    if "time" in ds.dims:
        assert strictly_increasing(ds["time"].values)
    assert has_geotransform(ds)
    assert np.all([int(ds[var].notnull().sum().values) > 0 for var in ds.data_vars])

    assert os.path.isdir(os.path.join(folder, source))
    x = product_name.replace(":","_")
    assert os.path.isfile(os.path.join(folder, source, f"{x}.nc"))
    fhs = glob.glob(os.path.join(folder, source, "**", "*.nc"), recursive=True) + \
        glob.glob(os.path.join(folder, source, "**", "*.tif"), recursive=True) + \
        glob.glob(os.path.join(folder, source, "**", "*.vrt"), recursive=True) + \
        glob.glob(os.path.join(folder, source, "**", "*.jp2"), recursive=True)
    assert len(fhs) == 1

@pytest.mark.parametrize("source_product", sorted(SOURCES.keys()))
def test_most_recent(source_product):

    x = source_product.split(".")
    source = x[0]
    product_name = ".".join(x[1:])
    latlim = [29.4, 29.5]
    lonlim = [30.7, 30.8]
    mod = importlib.import_module(f"pywapor.collect.product.{source}")

    x = mod.most_recent(product_name, latlim, lonlim)

    if isinstance(x, datetime.datetime):
        assert x.timestamp() < datetime.datetime.now().timestamp()
    else:
        assert isinstance(x, type(None))


if __name__ == "__main__":

    import os
    os.environ["PYWAPOR_REMOVE_TEMP_FILES"] = "NO"

    tmp_path = "/Users/hmcoerver/Local/test_dl"
    source_product = "SENTINEL2.S2MSI2A_R10m"
    slicer = slice(None)
