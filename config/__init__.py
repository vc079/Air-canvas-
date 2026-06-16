"""Compatibility shim: expose the on-disk `Config/` directory as
the importable `config` package so lowercase imports work across
platforms where the directory may be capitalised.

This module sets the package `__path__` to point at the existing
`Config` package location when available, allowing `import config.gestures`
to load `Config/gestures.py` transparently.
"""
try:
    import importlib
    _cfg = importlib.import_module("Config")
    __path__ = list(getattr(_cfg, "__path__", []))
except Exception:
    # If the above fails, fall back to the normal package layout.
    __path__ = __path__  # type: ignore
