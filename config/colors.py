"""Compatibility shim for legacy `config.colors` imports."""

from config.colours import *

__all__ = [
    "Color",
    "PALETTE",
    "UI_COLORS",
    "get_color_by_name",
    "get_swatch_rects",
    "color_at_x",
    "next_color",
    "prev_color",
    "COLOR_ORDER",
    "COLOR_BGR",
    "DEFAULT_COLOR",
]
