"""
    The biomass module contains all functions related to biomass production
    data components.

"""
import numpy as np
import xarray as xr

def fpar(ndvi, fpar_slope=1.257, fpar_offset=-0.161): # TODO: UPDATE WITH METHODOLOGY
    r"""
    Computes the fraction of absorbed PAR.
    PAR is photosynthetically active radiation, which is the solar radiation
    in the spectral range from 400-700 nm that is used by plants in
    photosynthesis. This function computes the fpar as a linear function of
    the ndvi, and is bounded between 0 and 1.

    .. math ::
        f_{par}=\Delta_{p} \cdot I_{ndvi} + c_{p}

    Parameters
    ----------
    ndvi : float
        normalized difference vegetation index
        :math:`I_{ndvi}`
        [-]

    fpar_slope : float
        slope of the fpar-ndvi curve
        :math:`\Delta_{p}`
        [-]

    fpar_offset : float
        offset of the fpar-ndvi curve
        :math:`c_{p}`
        [-]

    Returns
    -------
    f_par : float
        fraction of absorbed photosynthetically active radiation
        :math:`f_{par}`
        [-]

    Examples
    --------
    >>> import qattara.components.biomass as bio
    >>> bio.fpar(0.4)
    0.34179999999999999

    """
    x = fpar_offset + fpar_slope * ndvi

    if isinstance(x, xr.DataArray):
        out = x.clip(0.0,1.0)
    else:
        out = np.clip(x, 0.0, 1.0)
    return out


def par(ra_24):
    r"""
    Computes the photosynthetically active radiation (PAR). PAR is the solar
    radiation in the spectral range from 400-700 nm that is used by plants
    for photosynthesis.

    .. math ::
        PAR=r_{PAR} \cdot S^{\downarrow}

    where the following constant is used

    * :math:`r_{PAR}` = ratio par/solar radiation [-] = 0.48

    Parameters
    ----------
    ra_24 : float
       incoming shortwave radiation
       :math:`S^{\downarrow}`
       [W/m2]

    Returns
    -------
    apar : float
       photosynthetically active radiation
       :math:`PAR`
       [W/m2]

    Examples
    --------
    >>> import qattara.components.biomass as bio
    >>> bio.par(400.0)
    192.0

    """
    ABSORBED_RADIATION = 0.48

    return ABSORBED_RADIATION * ra_24

def co2_level_annual(year):
    r"""
    Computes the annual CO2 level based on the linear relation of Roel Van Hoolst (Copernicus).

    Parameters
    ----------
    year : integer
        year of interest
        :math:`y`
        [-]

    Returns
    -------
    co2_act : float
        actual annual CO2 level
        :math:`CO_2act`
        [ppmv]

    Examples
    --------
    #>>> from qattara import biomass
    #>>> import numpy as np
    #>>> biomass.t_air_kelvin(np.array([12.5,18.3]))
    #array([ 285.65,  291.45])
    """
    return 2.0775 * year - 3785.783


def temperature_dependency(t_air_k_12, dh_ap=52750, d_s=704.98, dh_dp=211000):
    r"""
    Computes the temperature dependency factor of GPP.

    Parameters
    ----------
    t_air_k_12 : float
        daytime air temperature
        :math:`T_{a,12}`
        [K]

    dh_ap : float
        activation energy
        :math:`\delta DH_{a,P}`
        [J/mol]

    d_s : float
        entropy of the denaturation equilibrium of CO2
        :math:`\delta S`
        [J/K.mol]

    dh_dp : float
        deactivation energy
        :math:`\delta H_{d,P}`
        [J/mol]

    Returns
    -------
    t_dep : float
        temperature dependency factor
        :math:`p(T_{atm})`
        [-]

    """
    R_G = 8.3144 # The molar gas constant in J K -1 mol-1

    c1 = 21.77
    x = R_G * t_air_k_12
    y = c1 - dh_ap / x
    z = (d_s * t_air_k_12 - dh_dp) / x
    return np.exp(y) / (1 + np.exp(z))


def co2_o2_specificity_ratio(t_air_k_12):
    r"""
    Computes the CO2/O2 specificity ratio

    Parameters
    ----------
    t_air_k_12 : float
        daytime air temperature
        :math:`T_{a,12}`
        [K]

    Returns
    -------
    tau_co2_o2 : float
        CO2/O2 specificity ratio
        :math:`\tau_CO{2}O{2}`
        [-]
    """
    R_G = 8.3144 # The molar gas constant in J K -1 mol-1

    a_t = 7.87 * 10 ** -5
    e_t = -42869.9
    x = R_G * t_air_k_12
    return a_t * np.exp(-e_t / x)


def inhibition_constant_o2(t_air_k_12):
    r"""
    Computes the inhibition constant for O2

    Parameters
    ----------
    t_air_k_12 : float
        daytime air temperature
        :math:`T_{a,12}`
        [K]

    Returns
    -------
    k_0 : float
        inhibition constant for O2
        :math:`K_0`
        [% O2]
    """
    R_G = 8.3144 # The molar gas constant in J K -1 mol-1

    a_0 = 8240
    e_0 = 13913.5
    x = R_G * t_air_k_12
    return a_0 * np.exp(-e_0 / x)


def affinity_constant_co2(t_air_k_12):
    r"""
    Computes the affinity constant for CO2 of Rubisco

    Parameters
    ----------
    t_air_k_12 : float
        daytime air temperature
        :math:`T_{a,12}`
        [K]

    Returns
    -------
    k_m : float
        affinity constant for CO2 of Rubisco
        :math:`K_m`
        [% CO2]
    """
    a1 = 2.419 * 10 ** 13
    a2 = 1.976 * 10 ** 22
    e1 = 59400
    e2 = 109600

    R_G = 8.3144 # The molar gas constant in J K -1 mol-1
    
    if isinstance(t_air_k_12, xr.DataArray):
        x1 = a1 * np.exp(-e1 / (R_G * t_air_k_12))
        x2 = a2 * np.exp(-e2 / (R_G * t_air_k_12))
        k_m = xr.where(t_air_k_12 >= 288.13, x1, x2)
    else:
        k_m = np.zeros(t_air_k_12.shape)
        k_m[t_air_k_12 >= 288.13] = a1 * np.exp(-e1 / (R_G * t_air_k_12[t_air_k_12 >= 288.13]))
        k_m[t_air_k_12 < 288.13] = a2 * np.exp(-e2 / (R_G * t_air_k_12[t_air_k_12 < 288.13]))
    return k_m


def co2_fertilisation(tau_co2_o2, k_m, k_0, co2_act, o2=20.9, co2_ref=281):
    r"""
    Computes the normalized CO2 fertilization factor (Veroustraete, 1994). No fertilization means values equal to 1. Fertilization means values larger than 1.

    Parameters
    ----------
    tau_co2_o2 : float
        CO2/O2 specificity ratio
        :math:`\tau_CO{2}O{2}`
        [-]

    k_m : float
        affinity constant for CO2 of Rubisco
        :math:`K_m`
        [% CO2]

    k_0 : float
        inhibition constant for O2
        :math:`K_0`
        [% O2]

    co2_act : float
        actual annual CO2 level
        :math:`CO_2act`
        [ppmv]

    o2 : float
        O2 concentration
        :math:`O_2`
        [%]

    co2_ref : float
        reference CO2 level
        :math:`CO_2ref`
        [ppmv]

    Returns
    -------
    co2_fert : float
        CO2 fertilization effect
        :math:`CO_2fert`
        [-]

    """
    Y = o2 / (2 * tau_co2_o2)
    Z = k_m * (1 + o2 / k_0)
    return (co2_act - Y) / (co2_ref - Y) * (Z + co2_ref) / (Z + co2_act)


def autotrophic_respiration(t_air_k_24, ar_slo=0.0, ar_int=0.5):
    r"""
    Computes the fraction lost to autotrophic respiration

    Parameters
    ----------
    t_air_k_24 : float
        daily air temperature
        :math:`T_{a,24}`
        [K]

    ar_slo : float
        slope to determine autotrophic respiratory fraction
        :math:`AR_{slo}`
        [-]

    ar_int : float
        intercept to determine autotrophic respiratory fraction
        :math:`AR_{int}`
        [-]

    Returns
    -------
    a_d : float
         autotrophic respiratory fraction
        :math:`A_d`
        [-]
    """
    return ar_slo * t_air_k_24 + ar_int


def net_primary_production_max(t_dep, co2_fert, a_d, apar, gcgdm=0.45):
    r"""
    Computes the maximum Net Primary Production.

    Parameters
    ----------
    t_dep : float
        temperature dependency factor
        :math:`p(T_{atm})`
        [-]

    co2_fert : float
        CO2 fertilization effect
        :math:`CO_2fert`
        [-]

    a_d : float
        autotrophic respiratory fraction
        :math:`A_d`
        [-]

    apar : float
        photosynthetic active radiation
        :math:`PAR`
        [W/m2]

    gcgdm : float
        conversion factor enabling the conversion from DM to C
        :math:`gCgDM`
        [-]

    Returns
    -------
    npp_max : float
        maximum net primary production
        :math:`NPP_{max}`
        [gC/m2]

    """
    apar_megajoule_per_day = apar * (60 * 60 * 24) * 10 ** (-6)  # second to day, joule to megajoule
    return t_dep * co2_fert * apar_megajoule_per_day * gcgdm * (1 - a_d)


def net_primary_production(npp_max, f_par, stress_moist, phot_eff=2.49):
    r"""
    Computes the maximum Net primary Production.

    Parameters
    ----------
    npp_max : float
        maximum net primary production
        :math:`NPP_{max}`
        [-]

    phot_eff : float
        Default Radiation Use Efficiency
        :math:` \eps`
        [gDM/MJ(APAR)]

    f_par : float
        fraction of absorbed photosynthetically active radiation
        :math:`f_{par}`
        [-]

    stress_moist : float
        stress factor of root zone moisture
        :math:`S_{m}`
        [-]

    Returns
    -------
    npp : float
        net primary production
        :math:`NPP`
        [gC/m2]
    """
    return npp_max * phot_eff * f_par * stress_moist

