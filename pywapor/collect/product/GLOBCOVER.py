import copy
import os
from functools import partial

from pywapor.collect.protocol import cog
from pywapor.enhancers import lulc
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
        "2009_V2.3_Global": {
            "Band1": [("lat", "lon"), "lulc"],
            "crs": [(), "spatial_ref"],
        },
    }

    req_dl_vars = {
        "2009_V2.3_Global": {
            "lulc": ["Band1", "crs"],
            "rs_min": ["Band1", "crs"],
            "z_obst_max": ["Band1", "crs"],
            "land_mask": ["Band1", "crs"],
            "lue_max": ["Band1", "crs"],
        },
    }

    out = {
        val: variables[product_name][val]
        for sublist in map(req_dl_vars[product_name].get, req_vars)
        for val in sublist
    }

    return out


def remove_var(ds, var):
    return ds.drop_vars([var])


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
        "2009_V2.3_Global": {
            "lulc": [],
            "rs_min": [
                partial(
                    lulc.lulc_to_x,
                    in_var="lulc",
                    out_var="rs_min",
                    convertor=lulc.globcover_to_rs_min(),
                )
            ],
            "z_obst_max": [
                partial(
                    lulc.lulc_to_x,
                    in_var="lulc",
                    out_var="z_obst_max",
                    convertor=lulc.globcover_to_z_obst_max(),
                )
            ],
            "land_mask": [
                partial(
                    lulc.lulc_to_x,
                    in_var="lulc",
                    out_var="land_mask",
                    convertor=lulc.globcover_to_land_mask(),
                )
            ],
            "lue_max": [
                partial(
                    lulc.lulc_to_x,
                    in_var="lulc",
                    out_var="lue_max",
                    convertor=lulc.globcover_to_lue_max(),
                )
            ],
        },
    }

    out = {k: v for k, v in post_processors[product_name].items() if k in req_vars}

    return out


def url_func(product_name):
    """Returns a url at which to collect GLOBCOVER data.

    Parameters
    ----------
    product_name : None
        Not used.

    Returns
    -------
    str
        The url.
    """
    return r"http://due.esrin.esa.int/files/GLOBCOVER_L4_200901_200912_V2.3.color.tif"


def most_recent(product_name, *args):
    return None


def download(
    folder,
    latlim,
    lonlim,
    product_name,
    req_vars=["lulc"],
    variables=None,
    post_processors=None,
    **kwargs,
):
    """Download GLOBCOVER data and store it in a single netCDF file.

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
    product_name : str, optional
        Name of the product to download, by default "P05".
    req_vars : list, optional
        Which variables to download for the selected product, by default ["p"].
    variables : dict, optional
        Metadata on which exact layers need to be requested from the server, by default None.
    post_processors : dict, optional
        Functions per variable that should be applied to the variable, by default None.

    Returns
    -------
    xr.Dataset
        Downloaded data.
    """

    folder = os.path.join(folder, "GLOBCOVER")

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
        latlim = [latlim[0] - 0.01, latlim[1] + 0.01]
        lonlim = [lonlim[0] - 0.01, lonlim[1] + 0.01]

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

    ds = cog.download(
        fn, product_name, coords, variables, post_processors, url_func, ndv=0
    )

    return ds[req_vars_orig]


if __name__ == "__main__":
    product_name = "2009_V2.3_Global"

    folder = r"/Users/hmcoerver/Local/globcov_test"
    latlim = [28.9, 29.7]
    lonlim = [30.2, 31.2]
    variables = None
    post_processors = None
    req_vars = ["lulc", "rs_min", "land_mask"]

    ds = download(
        folder,
        latlim,
        lonlim,
        product_name,
        req_vars=req_vars,
        variables=variables,
        post_processors=post_processors,
    )
