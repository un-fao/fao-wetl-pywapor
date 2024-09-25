import tracemalloc
import datetime
from pywapor.general.logger import log, adjust_logger
import types
import os
import numpy as np
import xarray as xr

def format_bytes(size):
    """Convert bytes to KB, MB, GB or TB.

    Parameters
    ----------
    size : int
        Total bytes.

    Returns
    -------
    tuple
        Converted size and label.
    """
    power = 2**10
    n = 0
    power_labels = {0: '', 1: 'K', 2: 'M', 3: 'G', 4: 'T'}
    while size > power:
        size /= power
        n += 1
    return size, power_labels[n]+'B'

def f_size(nbytes):
    return "{0:.2f}{1}".format(*format_bytes(nbytes))

def performance_check(func):
    """Add memory usage and elapsed time logger to a function.

    Parameters
    ----------
    func : function
        Function to monitor

    Returns
    -------
    function
        Function with added logging and a new `label` keyword argument.
    """
    def wrapper_func(*args, **kwargs):
        if "label" in kwargs.keys():
            label = kwargs.pop("label")
        else:
            label = f"`{func.__module__}.{func.__name__}`"
        log.info(f"--> {label}").add()
        t1 = datetime.datetime.now()
        tracemalloc.start()
        out = func(*args, **kwargs)
        mem_test = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        t2 = datetime.datetime.now()
        size, size_label = format_bytes(mem_test[1]-mem_test[0])
        first_string = f"> peak-memory-usage: {size:.1f}{size_label}, execution-time: {t2-t1}"
        if isinstance(out, xr.Dataset):
            fp = out.encoding.get("source", "")
            if os.path.isfile(fp):
                filesize = f_size(os.path.getsize(fp))
                first_string += f", filesize: {filesize}."
            log.info(first_string)
            log_non_unified_chunks(out)
        else:
            first_string += "."
            log.info(first_string)
        log.sub()
        return out
    wrapper_func.__module__ = func.__module__
    wrapper_func.__name__ = func.__name__
    setattr(wrapper_func, "decorated", True)
    return wrapper_func

def make_chunk_log_string(da):
    pref_sizes = da.encoding.get("preferred_chunks", {k: "n/a" for k in da.chunksizes.keys()})
    actual_sizes = da.chunksizes
    before_ = {True: "\x1b[30;41m", False: ""}
    after_ = {True: "\x1b[0m", False: ""}
    string_ =   "chunk|pref|dim: [" + \
                ", ".join([f"{k}: {before_[v[0] != pref_sizes[k]]}{v[0]}|{pref_sizes[k]}{after_[v[0] != pref_sizes[k]]}|{sum(v)}" for k, v in actual_sizes.items()]) + \
                f"], crs: {da.rio.crs}"
    return string_

def log_non_unified_chunks(ds):
    report = dict()
    for var in ds.data_vars:
        string_ = make_chunk_log_string(ds[var])
        if string_ in report.keys():
            report[string_].append(var)
        else:
            report[string_] = [var]
    for k, v in report.items():
        v_str = '`, `'.join(v)
        log.info(f"> vars: [`{v_str}`]").add()
        log.info(k).sub()

def decorate_function(obj, decorator):
    """Apply a decorator to a function if it hasn't already been decorated by this function.

    Parameters
    ----------
    obj : function
        Function to be decorated.
    decorator : function
        Decorator function.
    """
    module = obj.__module__
    name = obj.__name__
    if isinstance(obj, types.FunctionType) and not hasattr(obj, 'decorated'):
        setattr(module, name, decorator(obj))

@performance_check
def test(n, k = 100):
    x = np.random.random((n,k,1000))**2
    return x
