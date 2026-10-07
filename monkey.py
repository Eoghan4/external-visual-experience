"""
Monkey Mode — E.V.E
  Detects faces, renders monkey.png over each one (sized + rotated to match).
  Bananas rain from the top; when one touches a person's body it gets eaten.
  Total bananas eaten saved to monkey_bananas.txt.
  Q to quit.
"""

import argparse
import math
import os
import random
import time

import cv2
import mediapipe as mp
import numpy as np

# ── Args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--camera", type=int, default=1)
_args = parser.parse_args()

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
BANANA_FILE = os.path.join(SCRIPT_DIR, "monkey_bananas.txt")

def load_bananas():
    try:
        with open(BANANA_FILE) as f:
            return int(f.read().strip())
    except Exception:
        return 0

def save_bananas(n):
    with open(BANANA_FILE, "w") as f:
        f.write(str(n))

# ── Load sprites (BGRA) ────────────────────────────────────────────────────────
_monkey_src = cv2.imread(os.path.join(SCRIPT_DIR, "monkey.png"), cv2.IMREAD_UNCHANGED)
_banana_src = cv2.imread(os.path.join(SCRIPT_DIR, "banana.png"), cv2.IMREAD_UNCHANGED)

if _monkey_src is None or _banana_src is None:
    raise FileNotFoundError("monkey.png / banana.png not found in script directory")

# ── MediaPipe ──────────────────────────────────────────────────────────────────
vision      = mp.tasks.vision
RunningMode = vision.RunningMode

# Face landmark detector — gives us precise face geometry for placement + rotation
face_opts = vision.FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=RunningMode.VIDEO,
    num_faces=6,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
)
face_lm = vision.FaceLandmarker.create_from_options(face_opts)

# Pose detector — body bounding box for banana collision
pose_opts = vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path="pose_landmarker.task"),
    running_mode=RunningMode.VIDEO,
    num_poses=6,
)
pose_lm = vision.PoseLandmarker.create_from_options(pose_opts)

VIS_MIN = 0.4

FONT = cv2.FONT_HERSHEY_SIMPLEX

# ── Sprite overlay helper ──────────────────────────────────────────────────────
def _overlay_sprite(canvas, sprite_bgra, cx, cy, size, angle_deg):
    """
    Paste sprite_bgra (BGRA) onto canvas (BGR) centred at (cx, cy),
    scaled to (size x size), rotated by angle_deg.
    Respects alpha channel.
    """
    if size < 4:
        return

    # Resize
    s = cv2.resize(sprite_bgra, (size, size), interpolation=cv2.INTER_AREA)

    # Rotate around centre
    M   = cv2.getRotationMatrix2D((size / 2, size / 2), angle_deg, 1.0)
    s   = cv2.warpAffine(s, M, (size, size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    bgr   = s[:, :, :3]
    alpha = s[:, :,  3].astype(float) / 255.0

    # Destination rect (clip to canvas bounds)
    ih, iw = canvas.shape[:2]
    x0 = int(cx) - size // 2
    y0 = int(cy) - size // 2
    x1, y1 = x0 + size, y0 + size

    # Source rect in sprite
    sx0 = max(0, -x0);  sy0 = max(0, -y0)
    sx1 = size - max(0, x1 - iw)
    sy1 = size - max(0, y1 - ih)

    # Destination rect clipped
    dx0 = max(0, x0);  dy0 = max(0, y0)
    dx1 = min(iw, x1); dy1 = min(ih, y1)

    if dx1 <= dx0 or dy1 <= dy0 or sx1 <= sx0 or sy1 <= sy0:
        return

    a = alpha[sy0:sy1, sx0:sx1, np.newaxis]
    canvas[dy0:dy1, dx0:dx1] = (
        bgr[sy0:sy1, sx0:sx1] * a + canvas[dy0:dy1, dx0:dx1] * (1 - a)
    ).astype(np.uint8)


# ── Face → monkey placement ────────────────────────────────────────────────────
# MediaPipe face mesh landmark indices
_NOSE_TIP   = 1
_FOREHEAD   = 10    # approx top of head
_CHIN       = 152
_LEFT_EAR   = 234
_RIGHT_EAR  = 454

def _face_params(lms, fw, fh):
    """
    Return (cx, cy, size, angle_deg) for placing the monkey sprite over a face.
    cx/cy = centre of face, size = width across ears * scale factor,
    angle_deg = roll angle so the mask rotates with a tilted head.
    """
    def pt(i):
        return lms[i].x * fw, lms[i].y * fh

    lx, ly = pt(_LEFT_EAR)
    rx, ry = pt(_RIGHT_EAR)
    tx, ty = pt(_FOREHEAD)
    bx, by = pt(_CHIN)

    # Centre between ears
    cx = (lx + rx) / 2
    cy = (ty + by) / 2

    # Size: span from ear to ear, scaled up so the mask covers forehead to chin
    ear_span  = math.hypot(rx - lx, ry - ly)
    face_h    = math.hypot(bx - tx, by - ty)
    size      = int(max(ear_span, face_h) * 1.55)

    # Roll: angle of the ear-to-ear line relative to horizontal
    angle_deg = -math.degrees(math.atan2(ry - ly, rx - lx))

    return int(cx), int(cy), size, angle_deg


# ── Banana objects ─────────────────────────────────────────────────────────────
BANANA_W   = 70     # display width in pixels (height scales with aspect)
BANANA_SPD_MIN = 120
BANANA_SPD_MAX = 280
BANANA_SPIN    = 90   # degrees/second spin range
SPAWN_INTERVAL = 0.55  # seconds between new bananas


class Banana:
    _aspect = _banana_src.shape[0] / _banana_src.shape[1]   # h/w

    def __init__(self, fw):
        self.x     = float(random.randint(BANANA_W, fw - BANANA_W))
        self.y     = float(-BANANA_W)
        self.speed = random.uniform(BANANA_SPD_MIN, BANANA_SPD_MAX)
        self.angle = random.uniform(0, 360)
        self.spin  = random.uniform(-BANANA_SPIN, BANANA_SPIN)
        self.w     = BANANA_W
        self.h     = int(BANANA_W * self.aspect)
        self.eaten = False

    @property
    def aspect(self):
        return Banana._aspect

    def tick(self, dt):
        self.y     += self.speed * dt
        self.angle += self.spin  * dt

    def off_screen(self, fh):
        return self.y > fh + self.h * 2

    def centre(self):
        return int(self.x), int(self.y)

    def hits_body(self, pose_result, fw, fh):
        """True if this banana overlaps with any detected person's torso/limb area."""
        bx, by = self.x, self.y
        br = (self.w + self.h) / 4   # rough collision radius

        for lms in pose_result.pose_landmarks:
            # Check a handful of body landmarks: shoulders, hips, wrists, elbows
            for idx in (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28):
                lm = lms[idx]
                if lm.visibility < VIS_MIN:
                    continue
                px, py = lm.x * fw, lm.y * fh
                if math.hypot(bx - px, by - py) < br + 28:
                    return True
        return False


# ── Drawing ────────────────────────────────────────────────────────────────────
def _draw_text(img, text, x, y, scale, color, thickness=1):
    cv2.putText(img, text, (x, y), FONT, scale, (0, 0, 0),   thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, scale, color,        thickness,     cv2.LINE_AA)


def _draw_centered(img, text, y, scale, color, thickness):
    (tw, _), _ = cv2.getTextSize(text, FONT, scale, thickness)
    _draw_text(img, text, (img.shape[1] - tw) // 2, y, scale, color, thickness)


def draw_frame(img, bananas, total_bananas, face_result, pose_result):
    fh, fw = img.shape[:2]

    # ── Bananas ────────────────────────────────────────────────────────────────
    for b in bananas:
        bw = b.w
        bh = b.h
        # Draw resized banana (non-square: resize to bw wide, then overlay)
        s = cv2.resize(_banana_src, (bw, bh), interpolation=cv2.INTER_AREA)
        # Rotate
        diag = int(math.hypot(bw, bh)) + 2
        canvas_s = np.zeros((diag, diag, 4), dtype=np.uint8)
        ox, oy   = (diag - bw) // 2, (diag - bh) // 2
        canvas_s[oy:oy+bh, ox:ox+bw] = s
        M  = cv2.getRotationMatrix2D((diag / 2, diag / 2), b.angle, 1.0)
        rs = cv2.warpAffine(canvas_s, M, (diag, diag), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
        # Paste onto frame
        cx, cy = int(b.x), int(b.y)
        _paste_bgra(img, rs, cx - diag // 2, cy - diag // 2)

    # ── Monkey masks ──────────────────────────────────────────────────────────
    for lms in face_result.face_landmarks:
        cx, cy, size, angle = _face_params(lms, fw, fh)
        _overlay_sprite(img, _monkey_src, cx, cy, size, angle)

    # ── HUD ───────────────────────────────────────────────────────────────────
    bar_h = 54
    bar   = img[:bar_h].copy()
    cv2.rectangle(img, (0, 0), (fw, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(bar, 0.35, img[:bar_h], 0.65, 0, img[:bar_h])

    _draw_text(img, f"BANANAS EATEN: {total_bananas}", 14, 36, 1.1, (0, 210, 255), 2)


def _paste_bgra(canvas, sprite_bgra, x0, y0):
    """Paste a BGRA sprite onto a BGR canvas at top-left (x0, y0)."""
    ih, iw = canvas.shape[:2]
    sh, sw = sprite_bgra.shape[:2]

    sx0 = max(0, -x0);  sy0 = max(0, -y0)
    sx1 = sw - max(0, x0 + sw - iw)
    sy1 = sh - max(0, y0 + sh - ih)
    dx0 = max(0, x0);   dy0 = max(0, y0)
    dx1 = dx0 + (sx1 - sx0)
    dy1 = dy0 + (sy1 - sy0)

    if dx1 <= dx0 or dy1 <= dy0 or sx1 <= sx0 or sy1 <= sy0:
        return

    bgr   = sprite_bgra[sy0:sy1, sx0:sx1, :3]
    alpha = sprite_bgra[sy0:sy1, sx0:sx1,  3].astype(float) / 255.0
    a     = alpha[:, :, np.newaxis]
    canvas[dy0:dy1, dx0:dx1] = (bgr * a + canvas[dy0:dy1, dx0:dx1] * (1 - a)).astype(np.uint8)


# ── Main loop ──────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(_args.camera)

total_bananas = load_bananas()
bananas       = []
next_spawn    = time.time() + 1.0
timestamp_ms  = 0
prev_time     = time.time()

cv2.namedWindow("Monkey Mode", cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    now       = time.time()
    dt        = min(now - prev_time, 0.05)
    prev_time = now

    frame  = cv2.flip(frame, 1)
    fh, fw = frame.shape[:2]
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    face_result  = face_lm.detect_for_video(mp_img, timestamp_ms)
    pose_result  = pose_lm.detect_for_video(mp_img, timestamp_ms)
    timestamp_ms += 33

    # Spawn
    if now >= next_spawn:
        bananas.append(Banana(fw))
        next_spawn = now + SPAWN_INTERVAL

    # Tick + collision
    dirty = False
    for b in bananas:
        b.tick(dt)
        if not b.eaten and b.hits_body(pose_result, fw, fh):
            b.eaten        = True
            total_bananas += 1
            dirty          = True

    if dirty:
        save_bananas(total_bananas)

    bananas = [b for b in bananas if not b.off_screen(fh) and not b.eaten]

    draw_frame(frame, bananas, total_bananas, face_result, pose_result)
    cv2.imshow("Monkey Mode", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
