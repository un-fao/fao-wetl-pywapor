import os
import rasterio
import rioxarray.merge
import tempfile
import warnings
import requests
import copy
import urllib.parse
import xarray as xr
import multiprocessing
import time
from joblib import Parallel, delayed
import numpy as np
from pywapor.general.logger import log
from rasterio.crs import CRS
from bs4 import BeautifulSoup
from functools import partial
from urllib.parse import urlsplit, urlunsplit
from pywapor.general.processing_functions import open_ds, is_corrupt_or_empty
from pywapor.enhancers.apply_enhancers import apply_enhancers
from pywapor.collect.protocol.crawler import download_url, download_urls
from pywapor.general.processing_functions import save_ds, process_ds, remove_ds


def open_remote_dataset_with_retry(url, decode_coords="all", retries=3, backoff_seconds=5, **kwargs):
    """Open a remote OPeNDAP dataset with exponential backoff.

    NASA OPeNDAP endpoints can occasionally return transient network or netCDF I/O
    failures. Retrying a few times keeps short-lived outages from aborting a
    collection run.
    """
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=xr.SerializationWarning)
            return xr.open_dataset(url, decode_coords=decode_coords, **kwargs)
        except Exception as err:  # pragma: no cover - handled by retry path
            last_error = err
            if attempt >= retries:
                raise
            log.warning(
                "--> Remote dataset open failed for `%s` (attempt %s/%s): %s. Retrying in %ss.",
                url,
                attempt,
                retries,
                err,
                backoff_seconds * attempt,
            )
            time.sleep(backoff_seconds * attempt)
    raise last_error


def make_opendap_url(base_url, order):
