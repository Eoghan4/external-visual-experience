import argparse
import cv2
import mediapipe as mp
import numpy as np
import time
import random

parser = argparse.ArgumentParser()
parser.add_argument("--camera", type=int, default=1)
_args = parser.parse_args()

vision = mp.tasks.vision
RunningMode = vision.RunningMode
draw = vision.drawing_utils

HandConn = vision.HandLandmarksConnections
FaceConn = vision.FaceLandmarksConnections
PoseConn = vision.PoseLandmarksConnections

GREEN = draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
GREEN_THICK = draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)

hand_opts = vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
)
face_opts = vision.FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=RunningMode.VIDEO,
    num_faces=4,
)
pose_opts = vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path="pose_landmarker.task"),
    running_mode=RunningMode.VIDEO,
    num_poses=4,
)

hand_landmarker = vision.HandLandmarker.create_from_options(hand_opts)
face_landmarker = vision.FaceLandmarker.create_from_options(face_opts)
pose_landmarker = vision.PoseLandmarker.create_from_options(pose_opts)

cap = cv2.VideoCapture(_args.camera)
timestamp_ms = 0

cv2.namedWindow("Wireframe", cv2.WINDOW_NORMAL)

FONT = cv2.FONT_HERSHEY_SIMPLEX
COL_GREEN = (0, 255, 0)
COL_DIM = (0, 140, 0)
COL_BLACK = (0, 0, 0)

SCAN_LABELS = [
    "SCANNING",
    "FACIAL RECOGNITION IN PROGRESS",
    "ANALYSING BIOMETRICS",
    "MAPPING SKELETAL STRUCTURE",
    "IDENTIFYING SUBJECT",
    "RUNNING THREAT ASSESSMENT",
    "CROSS-REFERENCING DATABASE",
    "TRIANGULATING POSITION",
]

# --- bar state ---
bar_fill = 0.0           # 0.0 – 1.0
bar_speed = 0.3          # current fill rate (frac/sec)
bar_paused = False
bar_pause_until = 0.0
bar_next_change = 0.0    # when to pick a new speed/pause

# --- label state ---
label_index = 0
label_switch_at = time.time() + random.uniform(2.5, 5.0)
dot_tick = 0
dot_tick_at = time.time() + 0.5


def update_bar(now, dt):
    global bar_fill, bar_speed, bar_paused, bar_pause_until, bar_next_change

    if bar_paused:
        if now >= bar_pause_until:
            bar_paused = False
            bar_next_change = now + random.uniform(0.4, 1.5)
    else:
        bar_fill = min(1.0, bar_fill + bar_speed * dt)
        if bar_fill >= 1.0:
            bar_fill = 0.0

        if now >= bar_next_change:
            if random.random() < 0.25:
                bar_paused = True
                bar_pause_until = now + random.uniform(0.3, 1.2)
            else:
                bar_speed = random.uniform(0.05, 0.8)
            bar_next_change = now + random.uniform(0.4, 1.8)


def update_labels(now):
    global label_index, label_switch_at, dot_tick, dot_tick_at

    if now >= dot_tick_at:
        dot_tick = (dot_tick + 1) % 3
        dot_tick_at = now + 0.45

    if now >= label_switch_at:
        label_index = random.randint(0, len(SCAN_LABELS) - 1)
        label_switch_at = now + random.uniform(2.5, 6.0)


def draw_hud(img, has_face, has_hands, has_body):
    h, w = img.shape[:2]

    # --- status labels (top-left) ---
    labels = []
    if has_face:
        count = has_face
        labels.append(f"FACE{'S' if count > 1 else ''} DETECTED ({count})")
    if has_hands:
        count = has_hands
        labels.append(f"HAND{'S' if count > 1 else ''} DETECTED ({count})")
    if has_body:
        count = has_body
        labels.append(f"PERSON{'S' if count > 1 else ''} DETECTED ({count})")

    for i, text in enumerate(labels):
        y = 30 + i * 28
        cv2.putText(img, text, (12, y), FONT, 0.65, COL_BLACK, 4, cv2.LINE_AA)
        cv2.putText(img, text, (12, y), FONT, 0.65, COL_GREEN, 1, cv2.LINE_AA)

    # --- scanning progress bar (bottom) ---
    bar_w = int(w * 0.5)
    bar_h = 14
    bar_x = (w - bar_w) // 2
    bar_y = h - 36

    fill_px = int(bar_w * bar_fill)

    dots = "." * (dot_tick + 1)
    label = SCAN_LABELS[label_index] + dots

    # background track
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), COL_DIM, -1)
    # fill
    if fill_px > 0:
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill_px, bar_y + bar_h), COL_GREEN, -1)
    # border
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), COL_GREEN, 1)

    # label centred above bar
    (tw, _), _ = cv2.getTextSize(label, FONT, 0.5, 1)
    tx = bar_x + (bar_w - tw) // 2
    ty = bar_y - 8
    cv2.putText(img, label, (tx, ty), FONT, 0.5, COL_BLACK, 3, cv2.LINE_AA)
    cv2.putText(img, label, (tx, ty), FONT, 0.5, COL_GREEN, 1, cv2.LINE_AA)


prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    dt = now - prev_time
    prev_time = now

    update_bar(now, dt)
    update_labels(now)

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
    face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
    pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)

    annotated = np.copy(rgb)

    for hand_lms in hand_result.hand_landmarks:
        draw.draw_landmarks(
            annotated, hand_lms,
            HandConn.HAND_CONNECTIONS,
            GREEN_THICK,
            GREEN,
        )

    for face_lms in face_result.face_landmarks:
        draw.draw_landmarks(
            annotated, face_lms,
            FaceConn.FACE_LANDMARKS_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=GREEN,
        )
        draw.draw_landmarks(
            annotated, face_lms,
            FaceConn.FACE_LANDMARKS_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=GREEN_THICK,
        )

    for pose_lms in pose_result.pose_landmarks:
        draw.draw_landmarks(
            annotated, pose_lms,
            PoseConn.POSE_LANDMARKS,
            GREEN_THICK,
            GREEN,
        )

    out = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

    draw_hud(
        out,
        has_face=len(face_result.face_landmarks),
        has_hands=len(hand_result.hand_landmarks),
        has_body=len(pose_result.pose_landmarks),
    )

    cv2.imshow("Wireframe", out)

    timestamp_ms += 33
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
