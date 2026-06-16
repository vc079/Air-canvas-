"""
assets/asset_loader.py
----------------------
Single access point for every asset the Air Canvas HUD needs.

Responsibilities
----------------
* Load mode icons from  assets/icons/  and cache them in memory.
* Locate and load the preferred HUD font from  assets/fonts/  with
  a safe fallback chain so the app never crashes on a missing file.
* Auto-generate missing icons on first run (calls icon_generator).
* Expose a simple API so the main loop never touches file paths.

Usage
-----
    from assets.asset_loader import AssetLoader

    assets = AssetLoader()             # loads everything once at startup
    icon   = assets.get_icon("draw")   # numpy BGRA array, or None
    font   = assets.get_pil_font(20)   # PIL ImageFont for HUD text

    # For OpenCV-only projects (no PIL):
    assets.draw_icon(frame, "draw", x=10, y=10, size=48)
"""

from __future__ import annotations

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from functools import lru_cache

# ── directory layout ──────────────────────────────────────────────────
_ASSETS_DIR = Path(__file__).parent
ICONS_DIR   = _ASSETS_DIR / "icons"
FONTS_DIR   = _ASSETS_DIR / "fonts"

# ── icon filenames that must exist ───────────────────────────────────
ICON_NAMES = ["idle", "draw", "drag", "color_picker", "snapshot", "erase"]

# ── preferred font filename (place a .ttf in assets/fonts/) ──────────
PREFERRED_FONT = "RobotoMono-Regular.ttf"   # swap to any .ttf you like

# ── system font fallback chain (tried in order if preferred missing) ──
_SYSTEM_FONT_CANDIDATES = [
    # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
    # macOS
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    # Windows
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
]


# ======================================================================
class AssetLoader:
    """
    Loads and caches icons + fonts once at construction time.

    Parameters
    ----------
    auto_generate : bool
        If True (default), missing icons are generated automatically
        by calling  icon_generator.generate_all()  before loading.
    icon_size : int
        Icons are resized to this square dimension after loading (px).
    """

    def __init__(self, auto_generate: bool = True, icon_size: int = 48):
        self.icon_size    = icon_size
        self._icons:  dict[str, np.ndarray | None] = {}
        self._font_path:  str | None = None

        if auto_generate:
            self._ensure_icons_exist()

        self._load_icons()
        self._resolve_font_path()

    # ------------------------------------------------------------------ #
    # Icons                                                               #
    # ------------------------------------------------------------------ #

    def get_icon(self, name: str) -> np.ndarray | None:
        """
        Return the cached BGRA icon array for *name*, or ``None``.

        Parameters
        ----------
        name : str
            One of: idle, draw, drag, color_picker, snapshot, erase.
        """
        return self._icons.get(name)

    def get_icon_bgr(self, name: str) -> np.ndarray | None:
        """Return the icon as a 3-channel BGR array (alpha discarded)."""
        icon = self._icons.get(name)
        if icon is None:
            return None
        return cv2.cvtColor(icon, cv2.COLOR_BGRA2BGR)

    def draw_icon(
        self,
        frame:    np.ndarray,
        name:     str,
        x:        int,
        y:        int,
        size:     int | None = None,
        opacity:  float = 1.0,
    ) -> np.ndarray:
        """
        Alpha-composite an icon onto *frame* at pixel position (x, y).

        The icon is placed so that (x, y) is its **top-left** corner.
        Pixels outside the frame boundary are silently clipped.

        Parameters
        ----------
        frame   : BGR numpy array (modified in-place).
        name    : Icon name key.
        x, y    : Top-left pixel coordinates on *frame*.
        size    : Override the default icon_size for this draw call.
        opacity : Global opacity multiplier [0.0 – 1.0].

        Returns
        -------
        frame  (the same array, for chaining)
        """
        icon = self._icons.get(name)
        if icon is None:
            return frame

        sz = size or self.icon_size
        if icon.shape[0] != sz:
            icon = cv2.resize(icon, (sz, sz), interpolation=cv2.INTER_AREA)

        fh, fw = frame.shape[:2]
        # Clamp destination rectangle to frame bounds
        x1, y1 = max(x, 0), max(y, 0)
        x2, y2 = min(x + sz, fw), min(y + sz, fh)
        if x1 >= x2 or y1 >= y2:
            return frame

        # Crop icon to the visible region
        ix1 = x1 - x
        iy1 = y1 - y
        ix2 = ix1 + (x2 - x1)
        iy2 = iy1 + (y2 - y1)
        icon_crop = icon[iy1:iy2, ix1:ix2]

        # Alpha blend
        alpha = (icon_crop[:, :, 3:4].astype(float) / 255.0) * opacity
        src   = icon_crop[:, :, :3].astype(float)
        dst   = frame[y1:y2, x1:x2].astype(float)
        blended = src * alpha + dst * (1 - alpha)
        frame[y1:y2, x1:x2] = blended.astype(np.uint8)
        return frame

    def icon_names(self) -> list[str]:
        """Return the list of successfully loaded icon names."""
        return [k for k, v in self._icons.items() if v is not None]

    # ------------------------------------------------------------------ #
    # Fonts                                                               #
    # ------------------------------------------------------------------ #

    @property
    def font_path(self) -> str | None:
        """Resolved path of the font file in use, or None (use cv2 font)."""
        return self._font_path

    def get_pil_font(self, size: int = 18):
        """
        Return a PIL ``ImageFont`` object at the requested *size*.

        Falls back to ``ImageFont.load_default()`` if PIL/Pillow is not
        installed or no font file was resolved.

        Parameters
        ----------
        size : int  Point size.

        Returns
        -------
        PIL.ImageFont.FreeTypeFont | PIL.ImageFont.ImageFont
        """
        try:
            from PIL import ImageFont   # type: ignore
            if self._font_path:
                return ImageFont.truetype(self._font_path, size)
            return ImageFont.load_default()
        except ImportError:
            return None   # Caller should fall back to cv2.putText

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _ensure_icons_exist(self):
        """Auto-generate any missing icon PNGs via icon_generator."""
        missing = [
            n for n in ICON_NAMES
            if not (ICONS_DIR / f"{n}.png").exists()
        ]
        if not missing:
            return

        try:
            # Lazy import to avoid circular deps
            from assets.icon_generator import generate_all
        except ImportError:
            try:
                sys.path.insert(0, str(_ASSETS_DIR.parent))
                from assets.icon_generator import generate_all
            except ImportError:
                print("[AssetLoader] WARNING: icon_generator not found. "
                      "Run assets/icon_generator.py manually.")
                return

        print(f"[AssetLoader] Generating {len(missing)} missing icon(s) …")
        generate_all(output_dir=str(ICONS_DIR))

    def _load_icons(self):
        """Load all PNGs from ICONS_DIR into the cache."""
        ICONS_DIR.mkdir(parents=True, exist_ok=True)

        for name in ICON_NAMES:
            path = ICONS_DIR / f"{name}.png"
            if path.exists():
                img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    # Ensure BGRA (some editors save without alpha channel)
                    if img.shape[2] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                    if img.shape[0] != self.icon_size:
                        img = cv2.resize(
                            img,
                            (self.icon_size, self.icon_size),
                            interpolation=cv2.INTER_AREA,
                        )
                    self._icons[name] = img
                    continue

            print(f"[AssetLoader] WARNING: icon not found → {path}")
            self._icons[name] = None

    def _resolve_font_path(self):
        """Find the best available font file."""
        # 1. Preferred font in assets/fonts/
        preferred = FONTS_DIR / PREFERRED_FONT
        if preferred.exists():
            self._font_path = str(preferred)
            return

        # 2. Any .ttf or .otf in assets/fonts/
        FONTS_DIR.mkdir(parents=True, exist_ok=True)
        for ext in ("*.ttf", "*.otf"):
            candidates = list(FONTS_DIR.glob(ext))
            if candidates:
                self._font_path = str(candidates[0])
                return

        # 3. System font fallback chain
        for path in _SYSTEM_FONT_CANDIDATES:
            if os.path.exists(path):
                self._font_path = path
                return

        # 4. Nothing found — cv2 built-in font will be used
        print("[AssetLoader] INFO: No TrueType font found. "
              f"Drop a .ttf into '{FONTS_DIR}/' for custom HUD text.")
        self._font_path = None

    # ------------------------------------------------------------------ #
    # Repr                                                                 #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        loaded   = sum(1 for v in self._icons.values() if v is not None)
        font_str = Path(self._font_path).name if self._font_path else "cv2 built-in"
        return (
            f"AssetLoader(icons={loaded}/{len(ICON_NAMES)} loaded, "
            f"icon_size={self.icon_size}px, font={font_str!r})"
        )
