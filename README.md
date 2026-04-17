# Hand and Gesture Tracking System

A computer vision project for **hand and gesture tracking** using **MediaPipe**, **OpenCV**, and **Intel RealSense** depth camera integration.  
This repository contains several iterations and supporting modules for detecting hands, estimating orientation, and recognizing gestures in real time.

---

## 🚀 Features

- Real-time **hand detection** and tracking using `mediapipe` and `cv2`.
- Integration with **Intel RealSense** (`pyrealsense2`) for depth measurement.
- **Gesture recognition** module built on top of tracking.
- Modular design for future expansion (hardware integration, new gestures, etc.).
- Includes archived experimental versions for reference.

---

## 🧩 Tech Stack

- **Language:** Python >= 3.11, <3.14
- **Libraries:** `mediapipe`, `opencv-python`, `numpy`, `pyrealsense2`, `matplotlib`, `adafruit_bno055`, `pyfirmata`

---

## 📂 Folder Structure

```
project-root/
├── docs/                               # Documentation
│   └── overview.md                     # Technical documentation
├── robot_arm/
├── src/
│   └── archive/                        # Older / experimental versions
│       └── ashraf-openCV/
│       └── gestures/
│           ├──Guesture_recognition.py  # Gesture recognition logic
│           ├──guesture.py
│       └── PreviousVersions/
│           ├── Hand_tracking Tests.py
│           ├── Hand-Track2.py
│           ├── Hand_tracking.py
│           ├── realsense.py            # RealSense depth camera helper
│           ├── Old_main.py             # Older version of main
│       ├── Claw_tracking.py
│       ├── Five-fingered.py
│       ├── Hand_tracking.py
│   ├── main.py                         # Current main
├── tests/                              # Test scripts
│   ├── Hand_tracking Tests.py
│   ├── Gesture_test.py
├── pyproject.toml                     # Poetry package tthat has all needed libraries
└── README.md
```

---

## ⚙️ Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/NEU-Robot-Arm-Programmer/Computer-Vision.git
   cd Computer Vision
   ```

2. Install Poestry if you dont have it already

   **macOS**

   ```bash
   brew install poetry
   ```

3. Set up the environment and install dependencies:

   ```bash
   poetry env use python3.12
   poetry install --no-root
   ```

4. Activate the virtual environments:

   ```bash
      poetry shell
   ```

> **python version:\*** This project needs between 3.11 and 3.14

### 🎥 Intel RealSense Camera (Apple Silicon / arm64)

`pyrealsense2` is **not available on PyPI for arm64 Macs**. If you need RealSense depth camera support, install the native library via Homebrew instead:

```bash
brew install librealsense
```

For Python bindings, build from source:
[https://github.com/IntelRealSense/librealsense](https://github.com/IntelRealSense/librealsense)

RealSense is optional — the project will still run 2D hand detection and gesture recognition without it.

## ▶️ Usage

Run the main script:

```bash
  python src/main.py
```

If you have an Intel RealSense camera connected, it will use `pyrealsense2` for depth information.  
Without a RealSense device, it will still perform 2D hand detection and gesture recognition.

---

## 🧠 How It Works

1. The **MediaPipe** pipeline detects hand landmarks in real time.
2. **OpenCV** handles video frame processing and visualization.
3. **NumPy** assists in geometric calculations (angles, distances).
4. Optional **RealSense** module enhances depth-based gesture analysis.

---

## 🧰 Development Notes

- Older prototypes are preserved in `src/archive/`.
- The `Gesture_recognition.py` script demonstrates gesture classification examples.
- The codebase is modular — you can integrate new gesture rules or sensor inputs easily.

---

## 🧾 Future Improvements

- Expand hardware integration (Arduino, servos, etc.).
- Build GUI for live visualizations and calibration.
- Improve Communication with other drivers

---

## 👤 Authors

**Thomas Rowan**
**Stephen Sodipo**
**And members from the Robot Arm Software team**

Feel free to fork and experiment!

---

## 🪪 License

This project is released under the MIT License.
