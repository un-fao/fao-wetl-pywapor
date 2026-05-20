"""Pytest config for the pywapor test suite.

Auto-marks AppEEARS-backed tests as `serial` and deselects them when the run
is using pytest-xdist. AppEEARS issues one bearer token per Earthdata account,
so parallel workers sharing the same credentials invalidate each other's
tokens and get intermittent 403s. Run the serial pass separately:

    pytest -n auto                 # parallel; serial tests deselected
    pytest -m serial               # serial pass (no `-n`)
"""

import pytest

# Product modules that go through pywapor/collect/protocol/appeears.py.
_APPEEARS_PRODUCT_PREFIXES = ("MODIS.", "SRTM.")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "serial: test shares a single Earthdata/AppEEARS account; must not run under xdist.",
    )


def _is_appeears_item(item):
    return any(f"[{p}" in item.nodeid for p in _APPEEARS_PRODUCT_PREFIXES)


def _xdist_active(config):
    numprocesses = getattr(config.option, "numprocesses", None)
    if numprocesses not in (None, 0, "no"):
        return True
    dist = getattr(config.option, "dist", "no")
    return dist not in (None, "no")


def pytest_collection_modifyitems(config, items):
    for item in items:
        if _is_appeears_item(item):
            item.add_marker(pytest.mark.serial)

    if not _xdist_active(config):
        return

    markexpr = config.getoption("markexpr", default="") or ""
    if "serial" in markexpr:
        return

    keep, deselected = [], []
    for item in items:
        if "serial" in {m.name for m in item.iter_markers()}:
            deselected.append(item)
        else:
            keep.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = keep
