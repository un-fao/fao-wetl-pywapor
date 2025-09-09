import copy
import os

from pywapor.collect.protocol import cog
from pywapor.general.processing_functions import open_ds


def default_vars(product_name, req_vars):
    """Given a `product_name` and a list of requested variables, returns a dictionary
    with metadata on which exact layers need to be requested from the server, how they should
    be renamed, and how their dimensions are defined.

    Parameters
    ----------
    product_name : str
        Name of the product.
    req_vars : list
        List of variables to be collected.

    Returns
    -------
    dict
        Metadata on which exact layers need to be requested from the server.
    """
    variables = {
        "WaPOR2": {
            "Band1": [("lat", "lon"), "z_oro"],
            "Band2": [("lat", "lon"), "vpd_slope"],
            "Band3": [("lat", "lon"), "t_amp"],
            "Band4": [("lat", "lon"), "rn_offset"],
            "Band5": [("lat", "lon"), "r0_full"],
            "Band6": [("lat", "lon"), "r0_bare"],
            "Band7": [("lat", "lon"), "rs_min"],
            "Band8": [("lat", "lon"), "rn_slope"],
            "Band9": [("lat", "lon"), "t_opt"],
            "Band10": [("lat", "lon"), "lw_slope"],
            "Band11": [("lat", "lon"), "land_mask"],
            "Band12": [("lat", "lon"), "lw_offset"],
            "Band13": [("lat", "lon"), "z_obst_max"],
            "crs": [(), "spatial_ref"],
        },
        # 'WaPOR3': {
        #         "Band2": [("lat", "lon"), "lw_offset"],
        #         "Band1": [("lat", "lon"), "lw_slope"],
        #         "crs": [(), "spatial_ref"],
        # },
        "WaPOR3": {
            "Band1": [("lat", "lon"), "lw_offset"],
            "Band2": [("lat", "lon"), "lw_slope"],
            "Band3": [("lat", "lon"), "rn_offset"],
            "Band4": [("lat", "lon"), "rn_slope"],
            "Band5": [("lat", "lon"), "t_amp"],
            "Band6": [("lat", "lon"), "t_opt"],
            "Band7": [("lat", "lon"), "vpd_slope"],
            "crs": [(), "spatial_ref"],
        },
    }

    req_dl_vars = {
        "WaPOR2": {
            "z_oro": ["Band1", "crs"],
            "vpd_slope": ["Band2", "crs"],
            "t_amp": ["Band3", "crs"],
            "rn_offset": ["Band4", "crs"],
            "r0_full": ["Band5", "crs"],
            "r0_bare": ["Band6", "crs"],
            "rs_min": ["Band7", "crs"],
            "rn_slope": ["Band8", "crs"],
            "t_opt": ["Band9", "crs"],
            "lw_slope": ["Band10", "crs"],
            "land_mask": ["Band11", "crs"],
            "lw_offset": ["Band12", "crs"],
            "z_obst_max": ["Band13", "crs"],
        },
        # "WaPOR3": {
        #     "lw_offset": ["Band2", "crs"],
        #     "lw_slope": ["Band1", "crs"],
        # },
        "WaPOR3": {
            "lw_offset": ["Band1", "crs"],
            "lw_slope": ["Band2", "crs"],
            "rn_offset": ["Band3", "crs"],
            "rn_slope": ["Band4", "crs"],
            "t_amp": ["Band5", "crs"],
            "t_opt": ["Band6", "crs"],
            "vpd_slope": ["Band7", "crs"],
        },
    }

    out = {
        val: variables[product_name][val]
        for sublist in map(req_dl_vars[product_name].get, req_vars)
        for val in sublist
    }

    return out


def scale_factor(ds, var, scale=0.01):
    ds[var] = ds[var] * scale
    return ds


def default_post_processors(product_name, req_vars):
    """Given a `product_name` and a list of requested variables, returns a dictionary with a
    list of functions per variable that should be applied after having collected the data
    from a server.

    Parameters
    ----------
    product_name : str
        Name of the product.
    req_vars : list
        List of variables to be collected.

    Returns
    -------
    dict
        Functions per variable that should be applied to the variable.
    """
    post_processors = {
        "WaPOR2": {
            "land_mask": [],
            "lw_offset": [],
            "lw_slope": [],
            "r0_bare": [],
            "r0_full": [],
            "rn_offset": [],
            "rn_slope": [],
            "rs_min": [],
            "t_amp": [],
            "t_opt": [],
            "vpd_slope": [],
            "z_obst_max": [],
            "z_oro": [],
        },
        # 'WaPOR3': {
        #             "lw_offset": [],
        #             "lw_slope": [],
        #             },
        "WaPOR3": {
            "lw_offset": [],
            "lw_slope": [],
            "rn_offset": [],
            "rn_slope": [],
            "t_amp": [],
            "t_opt": [],
            "vpd_slope": [],
        },
    }

    out = {k: v for k, v in post_processors[product_name].items() if k in req_vars}

    return out


def url_func(product_name):
    """Returns a url at which to collect MERRA2 data.

    Parameters
    ----------
    product_name : None
        Not used.

    Returns
    -------
    str
        The url.
    """
    url = f"https://storage.googleapis.com/fao-cog-data/{product_name}.tif"
    return url


def most_recent(product_name, *args):
    return None


def download(
    folder,
    latlim,
    lonlim,
    product_name,
    req_vars,
    variables=None,
    post_processors=None,
    **kwargs,
):
    """Download MODIS data and store it in a single netCDF file.

    Parameters
    ----------
    folder : str
        Path to folder in which to store results.
    latlim : list
        Latitude limits of area of interest.
    lonlim : list
        Longitude limits of area of interest.
    timelim : list
        Period for which to prepare data.
    product_name : str
        Name of the product to download.
    req_vars : list
        Which variables to download for the selected product.
    variables : dict, optional
        Metadata on which exact layers need to be requested from the server, by default None.
    post_processors : dict, optional
        Functions per variable that should be applied to the variable, by default None.

    Returns
    -------
    xr.Dataset
        Downloaded data.
    """
    folder = os.path.join(folder, "STATICS")

    fn = os.path.join(folder, f"{product_name}.nc")
    req_vars_orig = copy.deepcopy(req_vars)
    if os.path.isfile(fn):
        existing_ds = open_ds(fn)
        req_vars_new = list(set(req_vars).difference(set(existing_ds.data_vars)))
        if len(req_vars_new) > 0:
            req_vars = req_vars_new
            existing_ds = existing_ds.close()
        else:
            return existing_ds[req_vars_orig]

    spatial_buffer = True
    if spatial_buffer:
        dx = dy = 2 / 6
        latlim = [latlim[0] - dy, latlim[1] + dy]
        lonlim = [lonlim[0] - dx, lonlim[1] + dx]

    coords = {"x": ("lon", lonlim), "y": ("lat", latlim)}

    if isinstance(variables, type(None)):
        variables = default_vars(product_name, req_vars)

    if isinstance(post_processors, type(None)):
        post_processors = default_post_processors(product_name, req_vars)
    else:
        default_processors = default_post_processors(product_name, req_vars)
        post_processors = {
            k: {True: default_processors[k], False: v}[v == "default"]
            for k, v in post_processors.items()
            if k in req_vars
        }

    ds = cog.download(fn, product_name, coords, variables, post_processors, url_func)

    return ds[req_vars_orig]


if __name__ == "__main__":
    req_vars = [
        # 'land_mask',
        "lw_offset",
        "lw_slope",
        # 'r0_bare',
        # 'r0_full',
        "rn_offset",
        "rn_slope",
        # 'rs_min',
        # 't_amp',
        "t_amp",
        "t_opt",
        "vpd_slope",
        # 'z_obst_max',
        # 'z_oro'
    ]

    product_name = "WaPOR3"

    folder = r"/Users/hmcoerver/Local/new_statics"
    # latlim = [28.9, 29.7]
    # lonlim = [30.2, 31.2]
    latlim = [21.9692194682626933, 21.9939120838340507]
    lonlim = [91.9371349243682801, 91.9657566608824339]

    variables = None
    post_processors = None

    ds = download(
        folder,
        latlim,
        lonlim,
        product_name,
        req_vars,
        variables=None,
        post_processors=None,
    )
