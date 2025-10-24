# Technical Overview

## Overview
This documentation describes the architecture and functionality of the Hand and Gesture Tracking System.

---

## 1. Core Modules

### `main.py`
- The central entry point of the project.
- Handles camera input (via `cv2.VideoCapture`) and uses **MediaPipe** to detect hand landmarks.
- Calculates hand orientation and gesture metrics.
- Optionally integrates **Intel RealSense** for depth-based analysis.
- Defines `handdetec` class with methods:
  - `find_hands()`
  - `find_position()`
  - `calculate_angles()`
  - `pixels_to_meters()`
  - `handOrientation()`
  - `handGestures()`
  - `drawKeyPoints()`
  - `wristBendAngles()`
  - `getRealDistance`
  - `degreesToSteps()`
  - 

### `realsense.py`
- Provides RealSense camera handling using the `pyrealsense2` library.
- Responsible for initializing camera streams, capturing depth/color frames, and returning processed data.
- Used internally by `main.py` or experimental tracking modules.

### `Gesture_recognition.py`
- Builds on `Hand_tracking` or `main.py` to classify specific gestures.
- Uses `mediapipe.tasks` for gesture recognition pipelines.
- Contains `hand_gestures()` and `main()` functions for testing and extending gesture logic.

---

## 2. Archived Modules

### `Five_fingered.py`
- Original, extended prototype with hardware sensor integration (`Adafruit BNO055`, `pyfirmata`).
- Provides lower-level access to orientation and servo control.
- Useful reference for combining hand tracking with external devices.

### `Hand-Track2.py`
- Transitional version before `main.py`.
- Similar structure but less modular and less refined.

### `Guesture_recognition.py`
- Transitional version before `main.py`.
- Similar structure but less modular and less refined.

### `Claw_tracking.py`
- Experimental script focused on detecting “claw” hand poses.
- Early feature testing using Mediapipe and math-based metrics.

### `fresh.py`
- Minimal stub (2 lines) — safe to ignore or remove.

---

## 3. Dependencies

- **OpenCV (cv2)** — video processing and display
- **MediaPipe** — hand landmark detection and gesture tracking
- **NumPy** — vector and geometry calculations
- **pyrealsense2** — Intel RealSense camera SDK
- **matplotlib** — visualization and debugging
- **Adafruit + pyfirmata** — optional hardware integration

---

## 4. Data Flow
```
Camera Input (RealSense / Webcam)
        ↓
OpenCV Frame Capture
        ↓
MediaPipe Hand Landmarks
        ↓
Position & Angle Computation
        ↓
Gesture Classification (Gesture_recognition.py)
        ↓
Visualization / Hardware Response
```

---

## 5. Future Work
- Introduce a configuration file for device and model parameters.
- Create a unit test suite for gesture classification.
- Modularize the gesture recognition pipeline for real-time ML inference.
