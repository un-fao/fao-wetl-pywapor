from pywapor.main import Configuration
import os
import copy
from pywapor.general.logger import log, adjust_logger

all_configs = {

    "WaPOR3_level_2": {
            '_ENHANCE_': {"bt": ["pywapor.enhancers.dms.thermal_sharpener.sharpen"],},
            '_EXAMPLE_': 'SENTINEL2.S2MSI2A_R60m',
            '_WHITTAKER_': {'SENTINEL2.S2MSI2A_R60m': {'lmbdas': 1000.0,
                                                        'method': 'whittaker'},
                            'VIIRSL1.VNP02IMG': {'a': 0.85,
                                                'lmbdas': 1000.0,
                                                'method': 'whittaker'}},
            'elevation': {'COPERNICUS.GLO90'},
            'meteorological': {'ERA5.reanalysis-era5-single-levels'},
            'optical': {'SENTINEL2.S2MSI2A_R60m'},
            'precipitation': {'CHIRPS.P05'},
            'soil moisture': {'FILE:{folder}{sep}se_root_out*.nc'},
            'solar radiation': {'ERA5.sis-agrometeorological-indicators'},
            'statics': {'STATICS.WaPOR3'},
            'thermal': {'VIIRSL1.VNP02IMG'}
            },

    "WaPOR3_level_3": {
            '_ENHANCE_': {"bt": ["pywapor.enhancers.dms.thermal_sharpener.sharpen"],},
            '_EXAMPLE_': 'SENTINEL2.S2MSI2A_R20m',
            '_WHITTAKER_': {'SENTINEL2.S2MSI2A_R20m': {'lmbdas': 1000.0,
                                                        'method': 'whittaker'},
                            'VIIRSL1.VNP02IMG': {'a': 0.85,
                                                'lmbdas': 1000.0,
                                                'method': 'whittaker'}},
            'elevation': {'COPERNICUS.GLO30'},
            'meteorological': {'ERA5.reanalysis-era5-single-levels'},
            'optical': {'SENTINEL2.S2MSI2A_R20m'},
            'precipitation': {'CHIRPS.P05'},
            'soil moisture': {'FILE:{folder}{sep}se_root_out*.nc'},
            'solar radiation': {'ERA5.sis-agrometeorological-indicators'},
            'statics': {'STATICS.WaPOR3'},
            'thermal': {'VIIRSL1.VNP02IMG'}
            },

    "WaPOR2_level_1": {
            '_ENHANCE_': {},
            '_EXAMPLE_': 'MODIS.MOD13Q1.061',
            '_WHITTAKER_': {},
            'elevation': {'SRTM.30M'},
            'meteorological': {'GEOS5.inst3_2d_asm_Nx'},
            'optical': {'MODIS.MYD13Q1.061', 'MODIS.MOD13Q1.061', 'MODIS.MCD43A3.061'},
            'precipitation': {'CHIRPS.P05'},
            'soil moisture': {'FILE:{folder}{sep}se_root_out*.nc'},
            'solar radiation': {'MERRA2.M2T1NXRAD.5.12.4'},
            'statics': {'STATICS.WaPOR2'},
            'thermal': {'MODIS.MOD11A1.061', 'MODIS.MYD11A1.061'}
            },

    "WaPOR2_level_2": {
            '_ENHANCE_': {},
            '_EXAMPLE_': 'TERRA.urn:eop:VITO:PROBAV_S5_TOC_100M_COG_V2',
            '_WHITTAKER_': {},
            'elevation': {'SRTM.30M'},
            'meteorological': {'GEOS5.inst3_2d_asm_Nx'},
            'optical': {'TERRA.urn:eop:VITO:PROBAV_S5_TOC_100M_COG_V2'},
            'precipitation': {'CHIRPS.P05'},
            'soil moisture': {'FILE:{folder}{sep}se_root_out*.nc'},
            'solar radiation': {'MERRA2.M2T1NXRAD.5.12.4'},
            'statics': {'STATICS.WaPOR2'},
            'thermal': {'MODIS.MOD11A1.061', 'MODIS.MYD11A1.061'}
            },

    "WaPOR2_level_3": {
            '_ENHANCE_': {},
            '_EXAMPLE_': 'LANDSAT.LC08_SR',
            '_WHITTAKER_': {},
            'elevation': {'SRTM.30M'},
            'meteorological': {'GEOS5.inst3_2d_asm_Nx'},
            'optical': {'LANDSAT.LC08_SR',
                        'LANDSAT.LC09_SR',
                        'LANDSAT.LE07_SR',
                        'LANDSAT.LT05_SR'},
            'precipitation': {'CHIRPS.P05'},
            'soil moisture': {'FILE:{folder}{sep}se_root_out*.nc'},
            'solar radiation': {'MERRA2.M2T1NXRAD.5.12.4'},
            'statics': {'STATICS.WaPOR2'},
            'thermal': {'LANDSAT.LC08_ST',
                        'LANDSAT.LC09_ST',
                        'LANDSAT.LE07_ST',
                        'LANDSAT.LT05_ST'}
            },

    "nrt": {
            '_ENHANCE_': {"bt": ["pywapor.enhancers.dms.thermal_sharpener.sharpen"],},
            '_EXAMPLE_': 'SENTINEL2.S2MSI2A_R20m',
            '_WHITTAKER_': {'SENTINEL2.S2MSI2A_R20m': {'lmbdas': 1000.0,
                                                        'method': 'whittaker'},
                            'VIIRSL1.VNP02IMG': {'a': 0.85,
                                                'lmbdas': 1000.0,
                                                'method': 'whittaker'}},
            'elevation': {'COPERNICUS.GLO30'},
            'meteorological': {'GEOS5.tavg1_2d_slv_Nx'},
            'optical': {'SENTINEL2.S2MSI2A_R20m'},
            'precipitation': {'CHIRPS.P05'},
            'soil moisture': {'FILE:{folder}{sep}se_root_out*.nc'},
            'solar radiation': {'ERA5.sis-agrometeorological-indicators'},
            'statics': {'STATICS.WaPOR3'},
            'thermal': {'VIIRSL1.VNP02IMG'}
            },

    "all_in": {
            '_ENHANCE_': {"bt": ["pywapor.enhancers.dms.thermal_sharpener.sharpen"],
                          "lst": ["pywapor.enhancers.dms.thermal_sharpener.sharpen"],},
            '_EXAMPLE_': 'SENTINEL2.S2MSI2A_R20m',
            '_WHITTAKER_': {'LANDSAT.LC08_SR': {'lmbdas': 1000.0, 'method': 'whittaker'},
                            'LANDSAT.LC08_ST': {'lmbdas': 1000.0, 'method': 'whittaker'},
                            'LANDSAT.LC09_SR': {'lmbdas': 1000.0, 'method': 'whittaker'},
                            'LANDSAT.LC09_ST': {'lmbdas': 1000.0, 'method': 'whittaker'},
                            'LANDSAT.LE07_SR': {'lmbdas': 1000.0, 'method': 'whittaker'},
                            'LANDSAT.LE07_ST': {'lmbdas': 1000.0, 'method': 'whittaker'},
                            'LANDSAT.LT05_SR': {'lmbdas': 1000.0, 'method': 'whittaker'},
                            'LANDSAT.LT05_ST': {'lmbdas': 1000.0, 'method': 'whittaker'},
                            'SENTINEL2.S2MSI2A_R20m': {'lmbdas': 1000.0, 'method': 'whittaker'},
                            'VIIRSL1.VNP02IMG': {'lmbdas': 1000.0, 'method': 'whittaker'}},
            'elevation': {'COPERNICUS.GLO30'},
            'meteorological': {'ERA5.reanalysis-era5-single-levels'},
            'optical': {
                        'LANDSAT.LC08_SR',
                        'LANDSAT.LC09_SR',
                        'LANDSAT.LE07_SR',
                        'LANDSAT.LT05_SR',
                        'SENTINEL2.S2MSI2A_R20m'},
            'precipitation': {'CHIRPS.P05'},
            'soil moisture': {'FILE:{folder}{sep}se_root_out*.nc'},
            'solar radiation': {'ERA5.sis-agrometeorological-indicators'},
            'statics': {'STATICS.WaPOR3'},
            'thermal': {
                    'VIIRSL1.VNP02IMG',
                    'LANDSAT.LC08_ST',
                    'LANDSAT.LC09_ST',
                    'LANDSAT.LE07_ST',
                    'LANDSAT.LT05_ST',
                    }
            },

}

if __name__ == "__main__":

    folder = "/Users/hmcoerver/Library/Mobile Documents/com~apple~CloudDocs/GitHub/pywapor/pywapor/configs"
    adjust_logger(True, folder, "INFO")

    for name, summar in all_configs.items():
        X = copy.deepcopy(summar)

        log.info(name)
        fh = os.path.join(folder, f"{name}.json")

        config = Configuration.from_summary(summar)
        config.to_json(fh)

        config_ref = Configuration.from_json(fh)

        log.info(f"{X == config.summary == config_ref.summary}")
        log.info(f"{config_ref.summary == config.summary}")
        log.info(f"{config_ref.full == config.full}")
        log.info(f"{config_ref.se_root == config.se_root}")
        log.info(f"{config_ref.et_look == config.et_look}")
