import cv2
import mediapipe as mp
import numpy as np
import argparse

GIF_A = "Muybridge_race_horse_animated.gif"
GIF_B = "ascii-animation.gif"

parser = argparse.ArgumentParser()
parser.add_argument("--camera", type=int, default=1)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--mon-x",  type=int, default=None)
parser.add_argument("--mon-y",  type=int, default=None)
parser.add_argument("--mon-w",  type=int, default=None)
parser.add_argument("--mon-h",  type=int, default=None)
_args = parser.parse_args()


def load_gif_frames(path, target_w, target_h):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (target_w, target_h))
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"Could not load any frames from {path}")
    return frames


W, H = _args.width, _args.height

print("Loading GIFs...")
frames_a = load_gif_frames(GIF_A, W, H)
frames_b = load_gif_frames(GIF_B, W, H)
print(f"  {GIF_A}: {len(frames_a)} frames")
print(f"  {GIF_B}: {len(frames_b)} frames")

vision = mp.tasks.vision
RunningMode = vision.RunningMode

hand_opts = vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
)
hand_landmarker = vision.HandLandmarker.create_from_options(hand_opts)

cap = cv2.VideoCapture(_args.camera)
cv2.namedWindow("GIF Slider", cv2.WINDOW_NORMAL)

if _args.mon_x is not None:
    import ctypes
    mw, mh = _args.mon_w or W, _args.mon_h or H
    cv2.moveWindow("GIF Slider", _args.mon_x, _args.mon_y)
    cv2.resizeWindow("GIF Slider", mw, mh)
    cv2.waitKey(1)
    hwnd = ctypes.windll.user32.FindWindowW(None, "GIF Slider")
    if hwnd:
        ctypes.windll.user32.MoveWindow(hwnd, _args.mon_x, _args.mon_y, mw, mh, True)
    cv2.setWindowProperty("GIF Slider", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
else:
    cv2.resizeWindow("GIF Slider", W, H)

frame_idx_a = 0
frame_idx_b = 0
tick_a = 0
tick_b = 0

# slider state — 0.0 = all A, 1.0 = all B
slider = 0.5
target_slider = 0.5
SMOOTH = 0.12          # lerp factor per frame
SLIDER_W = 6           # width of the divider line in px

timestamp_ms = 0
FONT = cv2.FONT_HERSHEY_SIMPLEX


def blend_frames(fa, fb, t):
    """Composite: left portion is fa, right portion is fb, split at x = t*W."""
    split = int(np.clip(t, 0.0, 1.0) * W)
    out = np.empty_like(fa)
    out[:, :split] = fa[:, :split]
    out[:, split:] = fb[:, split:]

    # divider line
    x1 = max(0, split - SLIDER_W // 2)
    x2 = min(W, x1 + SLIDER_W)
    out[:, x1:x2] = (255, 255, 255)

    # arrow hints on the divider
    cy = H // 2
    # left arrow <
    pts_l = np.array([
        [x1 - 18, cy],
        [x1 - 4, cy - 14],
        [x1 - 4, cy + 14],
    ], np.int32)
    # right arrow >
    pts_r = np.array([
        [x2 + 18, cy],
        [x2 + 4, cy - 14],
        [x2 + 4, cy + 14],
    ], np.int32)
    cv2.fillPoly(out, [pts_l], (255, 255, 255))
    cv2.fillPoly(out, [pts_r], (255, 255, 255))

    return out


def draw_label(img, text, x, y):
    cv2.putText(img, text, (x, y), FONT, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


gif_tick = 0
GIF_ADVANCE_EVERY = 2   # advance gif frame every N camera frames

while cap.isOpened():
    ret, cam_frame = cap.read()
    if not ret:
        break

    cam_frame = cv2.flip(cam_frame, 1)
    rgb = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
    timestamp_ms += 33

    hand_detected = False
    if result.hand_landmarks:
        # use the wrist (landmark 0) x-position of the first detected hand
        wrist = result.hand_landmarks[0][0]
        target_slider = wrist.x          # already normalised 0–1
        hand_detected = True

    # smooth slider toward target
    slider += (target_slider - slider) * SMOOTH

    # advance gif frames
    gif_tick += 1
    if gif_tick >= GIF_ADVANCE_EVERY:
        gif_tick = 0
        frame_idx_a = (frame_idx_a + 1) % len(frames_a)
        frame_idx_b = (frame_idx_b + 1) % len(frames_b)

    composite = blend_frames(frames_a[frame_idx_a], frames_b[frame_idx_b], slider)

    # labels
    draw_label(composite, "< " + GIF_A.replace(".gif", ""), 10, H - 16)
    draw_label(composite, GIF_B.replace(".gif", "") + " >", W - 260, H - 16)

    if not hand_detected:
        draw_label(composite, "show hand to control slider", W // 2 - 180, 36)

    cv2.imshow("GIF Slider", composite)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
