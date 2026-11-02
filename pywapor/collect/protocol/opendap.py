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


# OPENDAP NOTE: NASA remote endpoints occasionally fail with short-lived netCDF I/O errors.
# Retrying a few times avoids aborting an entire collection run because of a transient outage.
def open_remote_dataset_with_retry(url, decode_coords="all", retries=3, backoff_seconds=5, **kwargs):
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=xr.SerializationWarning)
            return xr.open_dataset(url, decode_coords=decode_coords, **kwargs)
        except Exception as err:
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
