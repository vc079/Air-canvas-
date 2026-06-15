"""
assets/icon_generator.py
------------------------
Generates every mode icon used by the Air Canvas HUD overlay and saves
them as PNG files inside  assets/icons/.

Run once to populate the icons folder:

    python -m assets.icon_generator          # from project root
    # or
    python assets/icon_generator.py

All icons are 64 × 64 pixels with a transparent background (BGRA).
They are drawn entirely with cv2 / numpy — no external image files or
design tools required.

Icons generated
---------------
idle.png         – closed fist outline
draw.png         – index-finger pointer with ink trail
drag.png         – two-finger pinch with arrows
color_picker.png – palette / paint-drop symbol
snapshot.png     – camera shutter outline
erase.png        – open hand with eraser box
"""

import os
import cv2
import numpy as np

# ── output directory (relative to project root) ──────────────────────
ICONS_DIR = os.path.join(os.path.dirname(__file__), "icons")
SIZE      = 64          # icon canvas size (pixels)
CX, CY    = SIZE // 2, SIZE // 2   # centre point


# =====================================================================
# Internal drawing helpers
# =====================================================================

def _blank() -> np.ndarray:
    """Return a transparent 64×64 BGRA canvas."""
    return np.zeros((SIZE, SIZE, 4), dtype=np.uint8)


def _circle(img, cx, cy, r, color, thickness=-1):
    cv2.circle(img, (cx, cy), r, color, thickness, cv2.LINE_AA)


def _line(img, p1, p2, color, thickness=2):
    cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)


def _rect(img, p1, p2, color, thickness=2):
    cv2.rectangle(img, p1, p2, color, thickness, cv2.LINE_AA)


def _poly(img, pts, color, thickness=2):
    pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=color,
                  thickness=thickness, lineType=cv2.LINE_AA)


WHITE  = (255, 255, 255, 255)
GREY   = (160, 160, 160, 255)
GREEN  = (  0, 220,   0, 255)
CYAN   = (255, 220,   0, 255)
RED    = (  0,   0, 220, 255)
ORANGE = (  0, 160, 255, 255)
PURPLE = (220,   0, 220, 255)
TRANSP = (  0,   0,   0,   0)


# =====================================================================
# Individual icon draw functions
# =====================================================================

def _draw_idle(img: np.ndarray):
    """Closed fist — rounded rectangle outline."""
    # Palm blob
    cv2.ellipse(img, (CX, CY + 4), (18, 20), 0, 0, 360, GREY, 2, cv2.LINE_AA)
    # Three knuckle bumps along the top
    for dx in (-10, 0, 10):
        _circle(img, CX + dx, CY - 16, 5, GREY, 2)
    # Thumb stub on the side
    cv2.ellipse(img, (CX - 20, CY), (6, 10), -30, 0, 360, GREY, 2, cv2.LINE_AA)


def _draw_draw(img: np.ndarray):
    """Index finger pointing down with a small ink dot at the tip."""
    # Finger shaft
    pts = [(CX - 5, CY - 26), (CX + 5, CY - 26),
           (CX + 5, CY + 10), (CX - 5, CY + 10)]
    _poly(img, pts, GREEN)
    # Rounded fingertip
    _circle(img, CX, CY - 26, 5, GREEN, 2)
    # Ink dot
    _circle(img, CX, CY + 16, 4, CYAN, -1)
    # Short trail
    _line(img, (CX, CY + 10), (CX, CY + 13), CYAN, 2)


def _draw_drag(img: np.ndarray):
    """Two fingers spread with a small handle circle between them."""
    # Left finger
    pts_l = [(CX - 16, CY - 22), (CX - 8, CY - 22),
             (CX - 8,  CY +  8), (CX - 16, CY + 8)]
    _poly(img, pts_l, WHITE)
    _circle(img, CX - 12, CY - 22, 4, WHITE, 2)

    # Right finger
    pts_r = [(CX + 8,  CY - 22), (CX + 16, CY - 22),
             (CX + 16, CY +  8), (CX + 8,  CY +  8)]
    _poly(img, pts_r, WHITE)
    _circle(img, CX + 12, CY - 22, 4, WHITE, 2)

    # Handle circle between fingers
    _circle(img, CX, CY - 8, 6, GREEN, 2)

    # Horizontal move arrows
    _line(img, (CX - 26, CY + 18), (CX + 26, CY + 18), GREY, 2)
    for dx, sign in [(-26, -1), (26, 1)]:
        tip = (CX + dx, CY + 18)
        _line(img, tip, (CX + dx - sign * 7, CY + 14), GREY, 2)
        _line(img, tip, (CX + dx - sign * 7, CY + 22), GREY, 2)


def _draw_color_picker(img: np.ndarray):
    """Three coloured swatches stacked in a fan."""
    colors = [
        (  0,   0, 220, 255),   # red
        (  0, 200,   0, 255),   # green
        (220,   0,   0, 255),   # blue
    ]
    offsets = [(-14, 6), (0, -2), (14, 6)]
    for (ox, oy), col in zip(offsets, colors):
        cv2.ellipse(img,
                    (CX + ox, CY + oy + 4),
                    (10, 18), 0, 0, 360,
                    col, -1, cv2.LINE_AA)
        cv2.ellipse(img,
                    (CX + ox, CY + oy + 4),
                    (10, 18), 0, 0, 360,
                    WHITE, 1, cv2.LINE_AA)

    # Index-finger pointer on top
    _line(img, (CX + 14, CY - 26), (CX + 14, CY - 10), CYAN, 3)
    _circle(img, CX + 14, CY - 28, 3, CYAN, -1)


def _draw_snapshot(img: np.ndarray):
    """Classic camera body with shutter circle."""
    # Camera body
    _rect(img, (CX - 22, CY - 10), (CX + 22, CY + 16), WHITE)
    # Viewfinder bump
    _rect(img, (CX - 8, CY - 18), (CX + 8, CY - 10), WHITE)
    # Lens – outer ring
    _circle(img, CX, CY + 3, 11, WHITE, 2)
    # Lens – inner filled
    _circle(img, CX, CY + 3, 6, ORANGE, -1)
    # Flash dot
    _circle(img, CX - 16, CY - 6, 3, ORANGE, -1)


def _draw_erase(img: np.ndarray):
    """Open hand silhouette with a dashed eraser bounding box."""
    # Five finger stubs
    finger_xs = [CX - 20, CX - 10, CX, CX + 10, CX + 20]
    for fx in finger_xs:
        _rect(img, (fx - 4, CY - 26), (fx + 4, CY - 8), GREY)
        _circle(img, fx, CY - 26, 4, GREY, 2)

    # Palm
    cv2.ellipse(img, (CX, CY + 4), (20, 14), 0, 0, 360, GREY, 2, cv2.LINE_AA)

    # Dashed bounding box (drawn with short line segments)
    margin = 4
    x1, y1, x2, y2 = CX - 28, CY - 30, CX + 28, CY + 20
    dash, gap = 6, 4
    # Top & bottom
    for y in (y1, y2):
        x = x1
        while x < x2:
            cv2.line(img, (x, y), (min(x + dash, x2), y), RED, 1, cv2.LINE_AA)
            x += dash + gap
    # Left & right
    for x in (x1, x2):
        y = y1
        while y < y2:
            cv2.line(img, (x, y), (x, min(y + dash, y2)), RED, 1, cv2.LINE_AA)
            y += dash + gap


# =====================================================================
# Dispatch table & public API
# =====================================================================

_ICONS: dict[str, callable] = {
    "idle":         _draw_idle,
    "draw":         _draw_draw,
    "drag":         _draw_drag,
    "color_picker": _draw_color_picker,
    "snapshot":     _draw_snapshot,
    "erase":        _draw_erase,
}


def generate_all(output_dir: str = ICONS_DIR, size: int = SIZE) -> list[str]:
    """
    Generate every icon and save it as a PNG.

    Parameters
    ----------
    output_dir : str   Destination folder (created if absent).
    size       : int   Icon dimensions in pixels (square).

    Returns
    -------
    list[str]  Paths of all files written.
    """
    os.makedirs(output_dir, exist_ok=True)
    written = []

    for name, draw_fn in _ICONS.items():
        img  = _blank()
        draw_fn(img)

        # Optionally resize if caller wants a different size
        if size != SIZE:
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

        path = os.path.join(output_dir, f"{name}.png")
        cv2.imwrite(path, img)
        written.append(path)
        print(f"  ✓  {path}")

    return written


def generate_one(name: str, output_dir: str = ICONS_DIR) -> str:
    """
    Generate a single icon by name.

    Parameters
    ----------
    name : str  One of: idle, draw, drag, color_picker, snapshot, erase.

    Returns
    -------
    str  Path of the saved file.

    Raises
    ------
    KeyError if the name is not found.
    """
    if name not in _ICONS:
        raise KeyError(f"Unknown icon '{name}'. Available: {list(_ICONS)}")
    os.makedirs(output_dir, exist_ok=True)
    img = _blank()
    _ICONS[name](img)
    path = os.path.join(output_dir, f"{name}.png")
    cv2.imwrite(path, img)
    return path


# =====================================================================
# Entry point
# =====================================================================

if __name__ == "__main__":
    print("Generating Air Canvas icons …")
    paths = generate_all()
    print(f"\nDone — {len(paths)} icons saved to '{ICONS_DIR}/'")
