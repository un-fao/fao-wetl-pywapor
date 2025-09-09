"""This module defines the default products to use for each variable per level
and defines the variables and their units.
"""
import xarray as xr
import numpy as np
import copy

def fill_attrs(ds, *args):
    """Fills in the attributes of a xr.Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset in which to add metadata attributes.

    Returns
    -------
    xr.Dataset
        The dataset with extra metadata.
    """

    defs = get_var_definitions()

    for var in list(ds.variables):

        if var in defs.keys():

            attributes = defs[var]

            _ = attributes.pop("definition")

            for k, v in attributes.items():
                ds[var].attrs[k] = v

    ds = fill_source_data_attr(ds)

    return ds

def initiate_ds(ds = None, dummy_data = False):
    """Create a template input xr.Dataset for et_look.

    Parameters
    ----------
    ds : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    if isinstance(ds, type(None)):
        ds = xr.Dataset()

    defs = get_var_definitions()

    for var, attributes in defs.items():

        if not var in list(ds.variables):
            if dummy_data:
                ds[var] = 0.5
            else:
                ds[var] = None

        _ = attributes.pop("definition")

        ds[var].attrs = attributes

    return ds

def flatten(list_of_lists):
    flat_list = list()
    def _flatten(list_of_lists):
        for item in list_of_lists:
            if type(item) == list:
                _flatten(item)
            else:
                flat_list.append(item)
        return flat_list
    _flatten(list_of_lists)
    return flat_list

def iterate(depends_on):
    j = 0
    depends_on_n = copy.deepcopy(depends_on)
    depends_on_n1 = dict()
    while depends_on_n != depends_on_n1:
        j += 1
        # print(f"{j}")
        depends_on_n1 = copy.deepcopy(depends_on_n)
        for var in depends_on_n.keys():
            for i, dependant in enumerate(depends_on_n[var]):
                if dependant in depends_on_n.keys():
                    depends_on_n[var][i] = depends_on[dependant]
        for var in depends_on_n.keys():
            depends_on_n[var] = np.unique(flatten(depends_on_n[var])).tolist()
        if j > 100:
            break
    return depends_on_n

def fill_source_data_attr(ds):
    """Determines the input data sources each variable depends on, based
    on the "calculated_with" attribute in the `ds`.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset for which to determine the source data.

    Returns
    -------
    xr.Dataset
        Dataset with new attribute information.
    """
    depends_on = {y: flatten([ds[y].calculated_with]) for y in ds.variables if "calculated_with" in ds[y].attrs.keys()}
    depends_on_n = iterate(depends_on)
    for var, dependants in depends_on_n.items():
        if var in list(ds.variables):
            ds[var].attrs["source_data"] = dependants
    return ds

def get_var_definitions():
    """Gives he defintion and unit of all the variables used in `et_look` and
    `se_root`.

    Returns
    -------
    dict
        Keys are variable names as used in `et_look` and `se_root`, values are
        dictionaries with keys `long_name`, `units` and `definition`.
    """

    defs = {
        "IO": {
            "long_name": "Solar constant",
            "units": "W m-2",
            "definition": ""
        },
        "B0c": {
            "long_name": "Beam irradiance normal to the solar beam",
            "units": "W m-2",
            "definition": ""
        },
        "Bhc": {
            "long_name": "Beam irradiance at a horizontal surface",
            "units": "W m-2",
            "definition": ""
        },
        "Dhc": {
            "long_name": "Diffuse irradiance at a horizontal surface",
            "units": "W m-2",
            "definition": ""
        },
        "G0": {
            "long_name": "Ext rad normal to solar beam",
            "units": "W m-2",
            "definition": ""
        },
        "L_bare": {
            "long_name": "Monin obukhov length dry vegetation",
            "units": "m",
            "definition": ""
        },
        "L_full": {
            "long_name": "Monin obukhov length wet vegetation",
            "units": "m",
            "definition": ""
        },
        "Tl2": {
            "long_name": "Airmass 2 linke atmospheric turbidity factor",
            "units": "-",
            "definition": ""
        },
        "ad_24": {
            "long_name": "Daily air density",
            "units": "kg m-3",
            "definition": ""
        },
        "ad_dry_24": {
            "long_name": "Daily dry air density",
            "units": "kg m-3",
            "definition": ""
        },
        "ad_dry_i": {
            "long_name": "Instantaneous dry air density",
            "units": "kg m-3",
            "definition": ""
        },
        "ad_i": {
            "long_name": "Instantaneous air density",
            "units": "kg m-3",
            "definition": ""
        },
        "ad_moist_24": {
            "long_name": "Daily moist air density",
            "units": "kg m-3",
            "definition": ""
        },
        "ad_moist_i": {
            "long_name": "Instantaneous moist air density",
            "units": "kg m-3",
            "definition": ""
        },
        "angle": {
            "long_name": "",
            "units": "",
            "definition": ""
        },
        "aspect": {
            "long_name": "Aspect (0 is north; 180 is south)",
            "units": "degrees",
            "definition": ""
        },
        "aspect_rad": {
            "long_name": "Aspect (0 is north; pi is south)",
            "units": "radians",
            "definition": ""
        },
        "band": {
            "long_name": "",
            "units": "",
            "definition": ""
        },
        "day_angle": {
            "long_name": "Day angle",
            "units": "radians",
            "definition": ""
        },
        "dd": {
            "long_name": "Damping depth",
            "units": "m",
            "definition": ""
        },
        "decl": {
            "long_name": "Solar declination",
            "units": "radians",
            "definition": ""
        },
        "disp": {
            "long_name": "Displacement height",
            "units": "m",
            "definition": ""
        },
        "disp_bare": {
            "long_name": "Displacement height (bare)",
            "units": "m",
            "definition": ""
        },
        "disp_full": {
            "long_name": "Displacement height (fully vegetated)",
            "units": "m",
            "definition": ""
        },
        "doy": {
            "long_name": "Day of year",
            "units": "-",
            "definition": ""
        },
        "dtime": {
            "long_name": "Decimal time",
            "units": "hours",
            "definition": ""
        },
        "e_24": {
            "long_name": "Daily evaporation energy equivalent",
            "units": "W m-2",
            "definition": ""
        },
        "e_24_init": {
            "long_name": "Initial estimate radiation equivalent daily evaporation",
            "units": "W m-2",
            "definition": ""
        },
        "e_24_mm": {
            "long_name": "Daily evaporation in mm",
            "units": "mm day-1",
            "definition": ""
        },
        "emiss_atm_i": {
            "long_name": "Instantaneous atmospheric emissivity",
            "units": "-",
            "definition": ""
        },
        "et_24_mm": {
            "long_name": "Daily evapotranspiration in mm",
            "units": "mm day-1",
            "definition": ""
        },
        "aeti_24_mm": {
            "long_name": "Daily evapotranspiration and interception in mm",
            "units": "mm day-1",
            "definition": ""
        },
        "et_ref_24": {
            "long_name": "Daily reference evapotranspiration (well watered grass) energy equivalent",
            "units": "W m-2",
            "definition": ""
        },
        "et_ref_24_mm": {
            "long_name": "Reference evapotranspiration (well watered grass)",
            "units": "mm day-1",
            "definition": ""
        },
        "g0_24": {
            "long_name": "Daily soil heat flux",
            "units": "W m-2",
            "definition": ""
        },
        "g0_bs": {
            "long_name": "Bare soil heat flux",
            "units": "W m-2",
            "definition": ""
        },
        "h0": {
            "long_name": "Solar elevation angle",
            "units": "degrees",
            "definition": ""
        },
        "h0ref": {
            "long_name": "Solar elevation angle corrected for refraction",
            "units": "degrees",
            "definition": ""
        },
        "h_bare": {
            "long_name": "Sensible heat flux for dry bare soil",
            "units": "W m-2",
            "definition": ""
        },
        "h_canopy_24_init": {
            "long_name": "Initial estimate of the sensible heat flux",
            "units": "W m-2",
            "definition": ""
        },
        "h_full": {
            "long_name": "Sensible heat flux full vegetation",
            "units": "W m-2",
            "definition": ""
        },
        "h_soil_24_init": {
            "long_name": "Initial estimate of the sensible heat flux for soil",
            "units": "W m-2",
            "definition": ""
        },
        "ha": {
            "long_name": "Solar hour angle",
            "units": "radians",
            "definition": ""
        },
        "ied": { # TODO check this with iesd
            "long_name": "Inverse earth sun distance",
            "units": "AU",
            "definition": ""
        },
        "iesd": {
            "long_name": "Inverse earth sun distance",
            "units": "AU",
            "definition": ""
        },
        "int_mm": {
            "long_name": "Interception",
            "units": "mm day-1",
            "definition": ""
        },
        "int_wm2": {
            "long_name": "Interception",
            "units": "W m-2",
            "definition": ""
        },
        "l_net": {
            "long_name": "Daily net longwave radiation",
            "units": "W m-2",
            "definition": ""
        },
        "lai": {
            "long_name": "Leaf area index",
            "units": "-",
            "definition": ""
        },
        "lai_eff": {
            "long_name": "Effective leaf area index",
            "units": "-",
            "definition": ""
        },
        "land_mask": {
            "long_name": "Land use classification",
            "units": "-",
            "definition": ""
        },
        "lat": {
            "long_name": "Latitude",
            "units": "degrees",
            "definition": ""
        },
        "lat_deg": {
            "long_name": "Latitude in degrees",
            "units": "degrees",
            "definition": ""
        },
        "lat_rad": {
            "long_name": "",
            "units": "",
            "definition": ""
        },
        "lh_24": {
            "long_name": "Daily latent heat of evaporation",
            "units": "J kg-1",
            "definition": ""
        },
        "lon": {
            "long_name": "Longitude",
            "units": "degrees",
            "definition": ""
        },
        "lon_deg": {
            "long_name": "Longitude in degrees",
            "units": "degrees",
            "definition": ""
        },
        "lon_rad": {
            "long_name": "Longitude in radians",
            "units": "radians",
            "definition": ""
        },
        "lst": {
            "long_name": "Land surface temperature",
            "units": "K",
            "definition": ""
        },
        "lst_max": {
            "long_name": "Maximum temperature at dry conditions",
            "units": "K",
            "definition": ""
        },
        "lst_min": {
            "long_name": "Minimum temperature at wet conditions",
            "units": "K",
            "definition": ""
        },
        "biomass_prod": {
            "long_name": "",
            "units": "kgDM ha-1 day-1",
            "definition": ""
        },
        "npp": {
            "long_name": "Net primary production",
            "units": "gC m-2 day-1",
            "definition": ""
        },
        "apar": {
            "long_name": "Absorbed photosynthetically active radiation",
            "units": "MJ m-2 day-1",
            "definition": ""
        },    
        "fpar": {
            "long_name": "Fraction of photosynthetically active radiation",
            "units": "-",
            "definition": ""
        },
        "lue": {
            "long_name": "Light use efficiency",
            "units": "gDM MJ-1",
            "definition": ""
        },
        "lue_max": {
            "long_name": "Maximum light use efficiency",
            "units": "gDM MJ-1",
            "definition": ""
        },
        "lw_offset": {
            "long_name": "Offset of the tau-term in the FAO-56 longwave radiation relationship",
            "units": "-",
            "definition": ""
        },
        "lw_slope": {
            "long_name": "Slope of the tau-term in the FAO-56 longwave radiation relationship",
            "units": "-",
            "definition": ""
        },
        "m": {
            "long_name": "Relative optical airmass",
            "units": "-",
            "definition": ""
        },
        "ndvi": {
            "long_name": "Normalized difference vegetation index",
            "units": "-",
            "definition": ""
        },
        "p_24": {
            "long_name": "Daily precipitation",
            "units": "mm day-1",
            "definition": ""
        },
        "p_air_0_24": {
            "long_name": "Daily air pressure at sea level",
            "units": "kpa",
            "definition": ""
        },
        "p_air_0_24_mbar": {
            "long_name": "Daily air pressure at sea level",
            "units": "mbar",
            "definition": ""
        },
        "p_air_0_i": {
            "long_name": "Instantaneous air pressure at sea level",
            "units": "-",
            "definition": ""
        },
        "p_air_24": {
            "long_name": "Daily air pressure",
            "units": "mbar",
            "definition": ""
        },
        "p_air_i": {
            "long_name": "Instantaneous air pressure",
            "units": "mbar",
            "definition": ""
        },
        "psy_24": {
            "long_name": "Daily psychrometric constant",
            "units": "mbar K-1",
            "definition": ""
        },
        "quantile": {
            "long_name": "",
            "units": "",
            "definition": ""
        },
        "qv_24": {
            "long_name": "Daily specific humidity",
            "units": "kg kg-1",
            "definition": ""
        },
        "qv_i": {
            "long_name": "Instantaneous specific humidity",
            "units": "kg kg-1",
            "definition": ""
        },
        "r0": {
            "long_name": "Albedo",
            "units": "-",
            "definition": ""
        },
        "r0_bare": {
            "long_name": "Dry bare soil surface albedo",
            "units": "-",
            "definition": ""
        },
        "r0_full": {
            "long_name": "Surface albedo under full vegetation cover",
            "units": "-",
            "definition": ""
        },
        "r_canopy": {
            "long_name": "Canopy resistance",
            "units": "s m-1",
            "definition": ""
        },
        "r_canopy_unstressed": {
            'long_name': 'Canopy resistance for zero moisture stress', 
            'units': 's m-1', 
            'definition': ''},
        "r_canopy_0": {
            "long_name": "Atmospheric canopy resistance",
            "units": "s m-1",
            "definition": ""
        },
        "r_soil": {
            "long_name": "Soil resistance",
            "units": "s m-1",
            "definition": ""
        },
        "ra_24": {
            "long_name": "Daily solar radiation",
            "units": "W m-2",
            "definition": ""
        },
        "ra_24_toa_flat": {
            "long_name": "Daily solar radiation at the top of atmosphere for a flat surface",
            "units": "W m-2",
            "definition": ""
        },
        "ra_canopy_init": {
            "long_name": "Initial canopy aerodynamic resistance",
            "units": "s m-1",
            "definition": ""
        },
        "ra_hor_clear_i": {
            "long_name": "Total clear-sky irradiance on a horizontal surface",
            "units": "W m-2",
            "definition": ""
        },
        "ra_soil_init": {
            "long_name": "Initial soil aerodynamic resistance",
            "units": "s m-1",
            "definition": ""
        },
        "raa": {
            "long_name": "Aerodynamical resistance dry surface",
            "units": "s m-1",
            "definition": ""
        },
        "rac": {
            "long_name": "Aerodynamical resistance canopy",
            "units": "s m-1",
            "definition": ""
        },
        "ras": {
            "long_name": "Aerodynamical resistance",
            "units": "s m-1",
            "definition": ""
        },
        "rn_24": {
            "long_name": "Daily net radiation",
            "units": "W m-2",
            "definition": ""
        },
        "rn_24_canopy": {
            "long_name": "Daily net radiation for the canopy",
            "units": "W m-2",
            "definition": ""
        },
        "rn_24_grass": {
            "long_name": "Daily net radiation for reference grass",
            "units": "W m-2",
            "definition": ""
        },
        "rn_24_soil": {
            "long_name": "Daily net radiation for soil",
            "units": "W m-2",
            "definition": ""
        },
        "rn_bare": {
            "long_name": "Net radiation bare soil",
            "units": "W m-2",
            "definition": ""
        },
        "rn_full": {
            "long_name": "Net radiation full vegetation",
            "units": "W m-2",
            "definition": ""
        },
        "rn_offset": {
            "long_name": "Offset rn/g0-relation water",
            "units": "-",
            "definition": ""
        },
        "rn_slope": {
            "long_name": "Slope rn/g0-relation water",
            "units": "-",
            "definition": ""
        },
        "rotm": {
            "long_name": "Rayleigh optical thickness at airmass m",
            "units": "-",
            "definition": ""
        },
        "rs_min": {
            "long_name": "Minimal stomatal resistance",
            "units": "s m-1",
            "definition": ""
        },
        "sc": {
            "long_name": "Seasonal correction",
            "units": "hours",
            "definition": ""
        },
        "se_root": {
            "long_name": "Relative root zone soil moisture",
            "units": "-",
            "definition": ""
        },
        "sf_soil": {
            "long_name": "Soil fraction",
            "units": "-",
            "definition": ""
        },
        "slope": {
            "long_name": "Slope",
            "units": "degrees",
            "definition": ""
        },
        "ssvp_24": {
            "long_name": "Daily slope of saturated vapour pressure curve",
            "units": "mbar K-1",
            "definition": ""
        },
        "stc": {
            "long_name": "Soil thermal conductivity",
            "units": "W m-1 K-1",
            "definition": ""
        },
        "stress_moist": {
            "long_name": "Stress factor for root zone soil moisture",
            "units": "-",
            "definition": ""
        },
        "stress_rad": {
            "long_name": "Stress factor for radiation",
            "units": "-",
            "definition": ""
        },
        "stress_temp": {
            "long_name": "Stress factor for air temperature",
            "units": "-",
            "definition": ""
        },
        "stress_vpd": {
            "long_name": "Stress factor for vapour pressure deficit",
            "units": "-",
            "definition": ""
        },
        "svp_24": {
            "long_name": "Daily saturated vapour pressure",
            "units": "mbar",
            "definition": ""
        },
        "t_24": {
            "long_name": "Daily transpiration energy equivalent",
            "units": "W m-2",
            "definition": ""
        },
        "t_24_init": {
            "long_name": "Initial estimate radiation equivalent daily transpiration",
            "units": "W m-2",
            "definition": ""
        },
        "t_24_mm": {
            "long_name": "Daily transpiration in mm",
            "units": "mm day-1",
            "definition": ""
        },
        "t_air_24": {
            "long_name": "Daily air temperature",
            "units": "C",
            "definition": ""
        },
        "t_air_i": {
            "long_name": "Instantaneous air temperature",
            "units": "C",
            "definition": ""
        },
        "t_air_k_24": {
            "long_name": "Daily air temperature in kelvin",
            "units": "K",
            "definition": ""
        },
        "t_air_k_i": {
            "long_name": "Instantaneous air temperature in kelvin",
            "units": "K",
            "definition": ""
        },
        "t_air_max_24": {
            "long_name": "Daily maximum temperature",
            "units": "C",
            "definition": ""
        },
        "t_air_min_24": {
            "long_name": "Daily minimum temperature",
            "units": "C",
            "definition": ""
        },
        "t_amp": {
            "long_name": "Yearly air temperature amplitude",
            "units": "K",
            "definition": ""
        },
        "t_diff": {
            "long_name": "Lapse-rate temperature adjustment",
            "units": "C",
            "definition": ""
        },
        "t_max_bare": {
            "long_name": "Maximum temperature at bare soil",
            "units": "K",
            "definition": ""
        },
        "t_max_full": {
            "long_name": "Maximum temperature at full vegetation cover",
            "units": "K",
            "definition": ""
        },
        "t_opt": {
            "long_name": "Optimum air temperature for plant growth",
            "units": "C",
            "definition": ""
        },
        "t_wet_i": {
            "long_name": "Instantaneous wet bulb temperature",
            "units": "C",
            "definition": ""
        },
        "t_wet_k_i": {
            "long_name": "Instantaneous wet bulb temperature in kelvin",
            "units": "K",
            "definition": ""
        },
        "trans_24": {
            "long_name": "Daily atmospheric transmissivity",
            "units": "-",
            "definition": ""
        },
        "u2m_24": {
            "long_name": "Daily average eastward wind speed at 2 meter",
            "units": "m s-1",
            "definition": ""
        },
        "u2m_i": {
            "long_name": "Instantaneous eastward wind speed at 2 meter",
            "units": "m s-1",
            "definition": ""
        },
        "u_24": {
            "long_name": "Daily wind speed at observation height",
            "units": "m s-1",
            "definition": ""
        },
        "u_b_24": {
            "long_name": "Daily wind speed at blending height",
            "units": "m s-1",
            "definition": ""
        },
        "u_b_i_bare": {
            "long_name": "Instantaneous wind speed at blending height for bare soil",
            "units": "m s-1",
            "definition": ""
        },
        "u_b_i_full": {
            "long_name": "Instantaneous wind speed at blending height for full vegetation",
            "units": "m s-1",
            "definition": ""
        },
        "u_i": {
            "long_name": "Instantaneous wind speed at observation height",
            "units": "m s-1",
            "definition": ""
        },
        "u_i_soil": {
            "long_name": "Instantaneous wind speed just above soil surface",
            "units": "m s-1",
            "definition": ""
        },
        "u_star_24_init": {
            "long_name": "Initial estimate of the daily friction velocity",
            "units": "m s-1",
            "definition": ""
        },
        "u_star_24_soil_init": {
            "long_name": "Initial estimate of the daily friction velocity for soil",
            "units": "m s-1",
            "definition": ""
        },
        "u_star_i_bare": {
            "long_name": "Instantaneous friction velocity bare soil",
            "units": "m s-1",
            "definition": ""
        },
        "u_star_i_full": {
            "long_name": "Instantaneous friction velocity vegetation",
            "units": "m s-1",
            "definition": ""
        },
        "v2m_24": {
            "long_name": "Daily average northward wind speed at 2 meter",
            "units": "m s-1",
            "definition": ""
        },
        "v2m_i": {
            "long_name": "Instantaneous northward wind speed at 2 meter",
            "units": "m s-1",
            "definition": ""
        },
        "vc": {
            "long_name": "Vegetation cover",
            "units": "-",
            "definition": ""
        },
        "vhc": {
            "long_name": "Volumetric heat capacity",
            "units": "J m-3 K-1",
            "definition": ""
        },
        "vp_24": {
            "long_name": "Daily vapour pressure",
            "units": "mbar",
            "definition": ""
        },
        "vp_i": {
            "long_name": "Instantaneous vapour pressure",
            "units": "mbar",
            "definition": ""
        },
        "vpd_24": {
            "long_name": "Daily vapour pressure deficit",
            "units": "mbar",
            "definition": ""
        },
        "vpd_slope": {
            "long_name": "Vapour pressure stress curve slope",
            "units": "mbar-1",
            "definition": ""
        },
        "ws": {
            "long_name": "Sunset hour angle",
            "units": "radians",
            "definition": ""
        },
        "wv_i": {
            "long_name": "Instantaneous total column atmospheric water vapor",
            "units": "kg m-2",
            "definition": ""
        },
        "z": {
            "long_name": "Elevation above sea level",
            "units": "m",
            "definition": ""
        },
        "z0m": {
            "long_name": "Surface roughness",
            "units": "m",
            "definition": ""
        },
        "z0m_bare": {
            "long_name": "Surface roughness (bare soil)",
            "units": "m",
            "definition": ""
        },
        "z0m_full": {
            "long_name": "Surface roughness (fully vegetated soil)",
            "units": "m",
            "definition": ""
        },
        "aod550_i": {
            "long_name": "",
            "units": "",
            "definition": ""
        },
        "fraction_h_bare": {
            "long_name": "",
            "units": "",
            "definition": ""
        },
        "fraction_h_full": {
            "long_name": "",
            "units": "",
            "definition": ""
        },
        "z_obst": {
            "long_name": "Obstacle height",
            "units": "m",
            "definition": ""
        },
        "z_obst_max": {
            "long_name": "Maximum obstacle height",
            "units": "m",
            "definition": ""
        },
        "z_oro": {
            "long_name": "Orographic roughness",
            "units": "m",
            "definition": ""
        },
        'nd_min': {
            'long_name': '', 
            'units': '', 
            'definition': ''},
        'nd_max': {
            'long_name': '', 
            'units': '', 
            'definition': ''},
        'vc_pow': {
            'long_name': '', 
            'units': '', 
            'definition': ''},
        'vc_min': {
            'long_name': '', 
            'units': '', 
            'definition': ''},
        'vc_max': {
            'long_name': '', 
            'units': '', 
            'definition': ''},
        'lai_pow': {
            'long_name': '', 
            'units': '', 
            'definition': ''},
        'diffusion_slope': {
            'long_name': '', 
            'units': '', 
            'definition': ''},
        'diffusion_intercept': {
            'long_name': '', 
            'units': '', 
            'definition': ''},
        't_min': {
            'long_name': '', 
            'units': '', 
            'definition': ''},
        't_max': {
            'long_name': '', 
            'units': '', 
            'definition': ''},
        'vp_slope': {
            'long_name': '', 
            'units': '', 
            'definition': ''},
        'vp_offset': {
            'long_name': '', 
            'units': '', 
            'definition': ''},
        'int_max': {
            'long_name': '', 
            'units': '', 
            'definition': ''},
        'tenacity': {
            'long_name': '', 
            'units': '', 
            'definition': ''},
        'rcan_max': {
            'long_name': '', 
            'units': '', 
            'definition': ''},
        'ndvi_obs_min': {
            'long_name': '', 
            'units': '', 
            'definition': ''},
        'ndvi_obs_max': {
            'long_name': '', 
            'units': '', 
            'definition': ''},
        'obs_fr': {
            'long_name': '', 
            'units': '', 
            'definition': ''},
        'z_obs': {
            'long_name': '', 
            'units': '', 
            'definition': ''},
        'z_b': {
            'long_name': '', 
            'units': '', 
            'definition': ''},
        'c1': {
            'long_name': '', 
            'units': '', 
            'definition': ''},
        'iter_h': {
            'long_name': '', 
            'units': '', 
            'definition': ''},
        'r_soil_pow': {
            'long_name': '', 
            'units': '', 
            'definition': ''},
        'r_soil_min': {
            'long_name': '', 
            'units': '', 
            'definition': ''},
        'se_top': {
            'long_name': '', 
            'units': '', 
            'definition': ''},
        'porosity': {
            'long_name': '', 
            'units': '', 
            'definition': ''},
        'r0_grass': {
            'long_name': '', 
            'units': '', 
            'definition': ''},
        'eps_a': {
            'long_name': '', 
            'units': '', 
            'definition': ''},
        "tau_co2_o2": {
            'long_name': '', 
            'units': '', 
            'definition': ''},
        "year": {
            'long_name': '', 
            'units': '', 
            'definition': ''},
        "t_air_k_min": {
            'long_name': '', 
            'units': '', 
            'definition': ''},
        "t_air_k_max": {
            'long_name': '', 
            'units': '', 
            'definition': ''},
        "t_air_k_12": {
            'long_name': '', 
            'units': '', 
            'definition': ''},
        "t_dep": {
            'long_name': '', 
            'units': '', 
            'definition': ''},
        "k_m": {
            'long_name': '', 
            'units': '', 
            'definition': ''},
        "k_0": {
            'long_name': '', 
            'units': '', 
            'definition': ''},
        "npp": {
            'long_name': '', 
            'units': 'gC m-2', 
            'definition': ''},
        "npp_max": {
            'long_name': '', 
            'units': '', 
            'definition': ''},
        "co2_fert": {
            'long_name': '', 
            'units': '', 
            'definition': ''},
        "f_par": {
            'long_name': '', 
            'units': '', 
            'definition': ''},
        "apar": {
            'long_name': '', 
            'units': '', 
            'definition': ''},
        "a_d": {
            'long_name': '', 
            'units': '', 
            'definition': ''},
        "co2_act": {
            'long_name': '', 
            'units': '', 
            'definition': ''},
        "t_dew_i": {
            'long_name': '', 
            'units': '', 
            'definition': ''},
        "t_dew_24": {
            'long_name': '', 
            'units': '', 
            'definition': ''},
    }

    return defs