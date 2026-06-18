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
    # Added cv2.CAP_DSHOW to bypass Windows default privacy driver blocks
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
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
                    renderer.push_history()
                    tip = tracker.index_tip(landmarks)
                    renderer.draw(tip)

                elif gesture == "drag":
                    renderer.push_history()
                    midpoint = tracker.pinch_midpoint(landmarks)
                    renderer.drag(midpoint)

                elif gesture == "palette":
                    renderer.close_action()
                    tip = tracker.index_tip(landmarks)
                    new_color = palette.hover(tip)
                    if new_color:
                        renderer.set_color(new_color)
                    frame = palette.draw(frame, renderer.color)

                elif gesture == "snapshot":
                    renderer.close_action()
                    snapshot.capture(renderer.canvas)

                elif gesture == "erase":
                    renderer.push_history()
                    bbox = tracker.hand_bbox(landmarks, frame.shape)
                    renderer.erase(bbox)
                    frame = eraser.draw(frame, bbox)

                elif gesture == "idle":
                    renderer.close_action()
                    renderer.reset_stroke()

                # Reset palette dwell state whenever we are NOT in palette
                # mode, so a stale hover candidate from a previous palette
                # session doesn't silently carry over (e.g. instantly
                # re-confirming a colour the moment the palette reopens).
                #if gesture != "palette":
                #    palette.reset_hover()

                # Draw drag handle when in drag mode
                if gesture == "drag":
                    mid = tracker.pinch_midpoint(landmarks)
                    frame = hud.draw_drag_handle(frame, mid)

            else:
                renderer.close_action()
                renderer.reset_stroke()

        # ── Composite canvas layer onto live feed ─────────────────────────────
        output = renderer.composite(frame)

        # ── HUD (mode label, snapshot flash, etc.) ────────────────────────────
        mode_name = engine.current_mode
        output = hud.draw(output, mode_name, renderer.color, snapshot.flash_active)
        snapshot.tick()  # Decrement flash / cooldown timers

        cv2.imshow(WINDOW_NAME, output)

        # ── Keyboard shortcuts ─────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # Esc
            print("[Air Canvas] Esc pressed — shutting down.")
            break

        elif key in (ord("z"), ord("Z")):
            # Undo. Any in-progress gesture action is closed first so the
            # undo always reverts a *complete* prior action rather than a
            # partially-drawn stroke that hasn't been snapshotted yet.
            renderer.close_action()
            if renderer.undo():
                print("[Air Canvas] Undo.")
            else:
                print("[Air Canvas] Nothing to undo.")

        elif key in (ord("y"), ord("Y")):
            renderer.close_action()
            if renderer.redo():
                print("[Air Canvas] Redo.")
            else:
                print("[Air Canvas] Nothing to redo.")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    tracker.close()
    print("[Air Canvas] Closed.")


if __name__ == "__main__":
    main()