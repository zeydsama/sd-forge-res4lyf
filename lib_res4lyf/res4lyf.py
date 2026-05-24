"""Minimal RES4LYF runtime stub for forge-neo.

The upstream `res4lyf.py` registers ComfyUI server routes and patches
`comfy.model_sampling.time_snr_shift`. Neither is available in
forge-neo, so we expose only the bits the vendored sampler code uses.
"""

import logging

_logger = logging.getLogger("sd-forge-res4lyf")

_DEBUG_LOGS = False


def set_debug_logs(enabled: bool) -> None:
    global _DEBUG_LOGS
    _DEBUG_LOGS = bool(enabled)


def is_debug_logging_enabled() -> bool:
    return _DEBUG_LOGS


def RESplain(*args, debug="info"):
    """Replacement for ComfyUI's RESplain console helper.

    Routes log lines through Python's logging module so they show up in
    the forge-neo console.
    """
    if isinstance(debug, bool):
        log_type = "debug" if debug else "info"
    else:
        log_type = debug

    if log_type == "debug" and not is_debug_logging_enabled():
        return

    msg = " ".join(str(a) for a in args)
    if log_type == "debug":
        _logger.debug(msg)
    elif log_type == "warning" or log_type == "warn":
        _logger.warning(msg)
    elif log_type == "error":
        _logger.error(msg)
    else:
        _logger.info(msg)


def init(check_imports=None):
    """No-op kept for source-level compatibility with the upstream init()."""
    RESplain("sd-forge-res4lyf: lib_res4lyf.res4lyf.init()", debug=True)


# Some upstream code reads these at import time.
display_sampler_category = False


def get_display_sampler_category():
    return display_sampler_category
