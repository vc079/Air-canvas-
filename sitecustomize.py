import sys
try:
    import Config as _Config
    sys.modules['config'] = _Config
except Exception:
    pass
