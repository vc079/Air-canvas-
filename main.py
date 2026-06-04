"""
main.py — Air Canvas entry point
---------------------------------
Initialises the webcam, wires up all core modules,
and runs the main gesture-to-drawing event loop.

Usage:
    python main.py

Press  Esc  to exit cleanly.
"""

import cv2
import numpy as np

from core.hand_tracker import HandTracker
from core.gesture_engine import GestureEngine
from core.canvas_renderer import CanvasRenderer
from ui.color_palette import ColorPalette
from ui.hud_overlay import HUDOverlay
from ui.eraser_box import EraserBox
from ui.multi_hand_warn import MultiHandWarning
from utils.snapshot import SnapshotManager
from config.settings import CAM_INDEX, FRAME_WIDTH, FRAME_HEIGHT, WINDOW_NAME


def main() -> None:
    # ── Camera setup ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {CAM_INDEX}. "
            "Check CAM_INDEX in config/settings.py."
        )

    # ── Module initialisation ─────────────────────────────────────────────────
    tracker   = HandTracker()
    engine    = GestureEngine()
    renderer  = CanvasRenderer(FRAME_WIDTH, FRAME_HEIGHT)
    palette   = ColorPalette()
    hud       = HUDOverlay()
    eraser    = EraserBox()
    warn      = MultiHandWarning()
    snapshot  = SnapshotManager()

    print(f"[Air Canvas] Running — press Esc to quit.")

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Air Canvas] Failed to read frame. Exiting.")
            break

        # Mirror so movement feels natural
        frame = cv2.flip(frame, 1)

        # Hand detection
        results = tracker.process(frame)
        num_hands = tracker.count_hands(results)

        # Multi-hand guard — pause all drawing when > 1 hand detected
        if num_hands > 1:
            frame = warn.draw(frame)
            engine.reset()
        else:
            landmarks = tracker.get_landmarks(results, frame.shape)

            if landmarks:
                finger_count = tracker.count_fingers(landmarks)
                gesture      = engine.resolve(finger_count, landmarks)

                # ── Gesture dispatch ──────────────────────────────────────────
                if gesture == "draw":
                    tip = tracker.index_tip(landmarks)
                    renderer.draw(tip)

                elif gesture == "drag":
                    midpoint = tracker.pinch_midpoint(landmarks)
                    renderer.drag(midpoint)

                elif gesture == "palette":
                    tip = tracker.index_tip(landmarks)
                    new_color = palette.hover(tip)
                    if new_color:
                        renderer.set_color(new_color)
                    frame = palette.draw(frame, renderer.color)

                elif gesture == "snapshot":
                    snapshot.capture(renderer.canvas)

                elif gesture == "erase":
                    bbox = tracker.hand_bbox(landmarks, frame.shape)
                    renderer.erase(bbox)
                    frame = eraser.draw(frame, bbox)

                elif gesture == "idle":
                    renderer.reset_stroke()

                # Draw drag handle when in drag mode
                if gesture == "drag":
                    mid = tracker.pinch_midpoint(landmarks)
                    frame = hud.draw_drag_handle(frame, mid)

            else:
                renderer.reset_stroke()

        # ── Composite canvas layer onto live feed ─────────────────────────────
        output = renderer.composite(frame)

        # ── HUD (mode label, snapshot flash, etc.) ────────────────────────────
        mode_name = engine.current_mode
        output = hud.draw(output, mode_name, renderer.color, snapshot.flash_active)
        snapshot.tick()  # Decrement flash / cooldown timers

        cv2.imshow(WINDOW_NAME, output)

        # Esc key to quit
        if cv2.waitKey(1) & 0xFF == 27:
            print("[Air Canvas] Esc pressed — shutting down.")
            break

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    tracker.close()
    print("[Air Canvas] Closed.")


if __name__ == "__main__":
    main()