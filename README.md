# E.V.E — External Visual Experience

Two artistic webcam programmes launched from a single terminal UI.

## Setup

```bash
pip install -r requirements.txt
```

Download the MediaPipe model files and place them in the project directory:
- [`hand_landmarker.task`](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task)
- [`face_landmarker.task`](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task)
- [`pose_landmarker.task`](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task)

## Run

```bash
python launcher.py
```

Use **↑ ↓** to select a programme, **Tab** to switch between sections (programme / camera / display), **← →** to change the selection within a section, and **Enter** to launch.

---

## Programmes

### ASCII Cam

Renders live webcam video as ASCII art directly in the terminal.

Optional arguments (passed automatically by the launcher, or use directly):

```bash
python ascii_cam.py --camera 1 --fps 20 --scale 1.0 --backend auto
```

- `--camera` — camera index
- `--fps` — target terminal refresh rate
- `--scale` — terminal width scale (0.2 – 1.0)
- `--backend` — camera backend: `auto`, `dshow`, `msmf`, `any`

Stop with **Ctrl+C**.

### Wireframe

Real-time webcam overlay using MediaPipe — draws face mesh, hand skeleton, and full body pose landmarks simultaneously.

```bash
python wireframe.py --camera 1
```

- `--camera` — camera index

Stop with **Q**.

---

## Notes

- Camera 0 = Laptop Cam, Camera 1 = Webcam. Additional cameras are auto-detected.
- Both programmes cannot share the same camera simultaneously. The launcher warns if you select the same index for both.
- If you get frame-grab issues on Windows, close other apps using the camera (Teams, Zoom, browser tabs).
