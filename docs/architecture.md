# 🏛️ Advanced Air Canvas: System Architecture

## Overview
The Advanced Air Canvas is a real-time computer vision application designed with a modular, decoupled architecture. The system separates the camera event loop, mathematical tracking logic, visual overlays, and configuration settings to ensure scalability and ease of testing.

## Core Pipeline
The application operates on a continuous frame-by-frame loop managed by `main.py`. The data flow follows a strict pipeline:

1. **Capture:** `cv2.VideoCapture` pulls a raw frame from the webcam.
2. **Detection (`core/hand_tracker.py`):** The frame is passed to MediaPipe to extract 21 3D hand landmarks.
3. **Resolution (`core/gesture_engine.py`):** Fingertip and knuckle Y-coordinates are compared to determine the current finger count and resolve the intended user state (e.g., Draw, Drag, Erase).
4. **Processing (`core/canvas_renderer.py`):** Based on the resolved gesture, drawing coordinates are mapped to a secondary black numpy array (the "Canvas Layer").
5. **Compositing (`utils/compositing.py`):** The Canvas Layer and the Live Webcam Feed are merged using bitwise masking operations.
6. **Overlay (`ui/`):** Heads-Up Display (HUD) elements, multi-hand warnings, and color palettes are drawn on top of the final composite.

## Module Breakdown

### `/core` (Business Logic)
Contains the heavy lifting for the application. These modules handle the mathematical translations between physical space and digital outputs. They are strictly decoupled from the UI layer.

### `/ui` (Presentation Layer)
Responsible for all non-ink visual elements. Modules here utilize OpenCV drawing functions (`cv2.rectangle`, `cv2.putText`) to create interactive menus, bounding boxes, and system warnings.

### `/utils` (Helper Functions)
Stateless utility functions that process discrete inputs and outputs. This includes bitwise image compositing, coordinate smoothing algorithms (to reduce jitter), and file I/O for snapshots.

### `/config` (Constants & Tunables)
Centralized configuration. All magic numbers (camera indices, frame dimensions, RGB color tuples, threshold values) are stored here to prevent hardcoding across the application.