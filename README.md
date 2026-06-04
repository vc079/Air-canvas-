# Air Canvas 🎨🖐️

A real-time, computer-vision virtual whiteboard. Draw, erase, and manipulate
digital ink in the air using hand gestures — no stylus, no touchscreen.

Built with **Python 3.7+**, **OpenCV**, and **Google MediaPipe**.

---

## ✨ Features

| Fingers | Mode | What it does |
|--------:|------|--------------|
| 0 (fist) | **Idle** | Pauses all tracking — move freely without drawing |
| 1 (index) | **Draw** | Index fingertip draws continuous ink |
| 2 (index + middle) | **Drag** | Pinch grip moves the entire canvas |
| 3 | **Color palette** | Hover index tip over a color box to switch ink |
| 4 | **Snapshot** | Saves canvas as `.png` (3-second anti-spam cooldown) |
| 5 (open hand) | **Wipe eraser** | Bounding box around your hand erases ink beneath it |

**Multi-hand safety** — a second hand in frame pauses all drawing and
displays an on-screen warning until only one hand remains.

---

## 🛠️ Prerequisites

- Python 3.7 or newer
- A working webcam

---

## 🚀 Installation

```bash
# 1. Clone or download the repository
git clone https://github.com/your-username/air_canvas.git
cd air_canvas

# 2. (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python main.py
```

Press **Esc** to exit cleanly.

---

## 📁 Project Structure

```
air_canvas/
├── main.py                  # Entry point
├── requirements.txt
├── README.md
├── .gitignore
├── LICENSE
│
├── core/                    # Logic modules
│   ├── hand_tracker.py      # MediaPipe wrapper, landmark extraction
│   ├── gesture_engine.py    # Finger count → mode resolver
│   └── canvas_renderer.py   # numpy canvas + OpenCV compositing
│
├── ui/                      # On-screen overlays
│   ├── color_palette.py     # Color menu (3-finger mode)
│   ├── hud_overlay.py       # Mode label, snapshot flash, drag handle
│   ├── eraser_box.py        # Bounding-box eraser visualisation
│   └── multi_hand_warn.py   # Warning banner for > 1 hand
│
├── utils/                   # Reusable helpers
│   ├── compositing.py       # bitwise_and / bitwise_or pipeline
│   ├── smoothing.py         # Line interpolation for smooth strokes
│   ├── snapshot.py          # PNG save with cooldown + flash timer
│   └── finger_counter.py    # Y-coordinate landmark math
│
├── config/                  # Tunable constants (no magic numbers elsewhere)
│   ├── settings.py          # Camera, window, cooldown
│   ├── colors.py            # Palette hex values
│   ├── gestures.py          # Finger-count → gesture name map
│   └── camera.py            # Resolution presets
│
├── assets/
│   ├── icons/               # Optional UI icons
│   ├── fonts/               # Optional custom fonts
│   └── snapshots/           # Saved .png outputs land here
│
├── tests/
│   ├── test_gestures.py
│   ├── test_compositing.py
│   └── test_snapshot.py
│
└── docs/
    ├── gesture_guide.md
    ├── architecture.md
    └── demo.gif
```

---

## 🧠 How It Works

### Hand tracking
MediaPipe identifies **21 3-D landmarks** on each hand. The app compares
each fingertip's Y-coordinate against its lower knuckle to determine whether
that finger is extended or curled.

### Rendering pipeline
Two layers are maintained separately:

1. **Live webcam frame** — flipped horizontally so movement feels natural.
2. **Canvas** — a pitch-black `numpy` array the same size as the frame.

### Compositing
On each frame:
```
mask       = bitwise_and(canvas,  canvas,  mask=gray_canvas)
frame_cut  = bitwise_and(frame,   frame,   mask=bitwise_not(gray_canvas))
output     = bitwise_or (frame_cut, mask)
```
This carves the ink shape out of the live feed and drops the colour layer
into those slots — creating the illusion of augmented-reality ink.

---

## ⚙️ Configuration

All tunable values are in `config/`. No magic numbers in logic files.

| File | What it controls |
|------|-----------------|
| `settings.py` | Camera index, resolution, window name, snapshot cooldown |
| `colors.py` | Palette colours (hex → BGR tuples) |
| `gestures.py` | Finger-count to gesture name mapping |
| `camera.py` | Resolution presets (720p, 1080p, etc.) |

---

## 🧪 Running Tests

```bash
pip install pytest
pytest tests/
```

---

## 📸 Snapshots

Saved automatically to `assets/snapshots/` when you hold up **4 fingers**.
Filenames are timestamped: `snapshot_20240601_143022.png`.

---

## 🛑 Exiting

Make sure the camera window is active, then press **Esc**.

---

## 📄 License

MIT — see [LICENSE](LICENSE).