"""
Fruit Ninja — pose-controlled for E.V.E
  Single window: live webcam feed with fruit, bombs, slice trails, and HUD
  composited directly on top.

Controls (body):
  Raise either hand above shoulder  →  START (title / game-over screen)
  Move wrists quickly through fruit →  SLICE
  Hit a bomb                        →  GAME OVER
  Q                                 →  quit
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

# ── High-score file ────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HS_FILE    = os.path.join(SCRIPT_DIR, "fruit_ninja_highscore.txt")

def load_highscore():
    try:
        with open(HS_FILE) as f:
            return int(f.read().strip())
    except Exception:
        return 0

def save_highscore(score):
    with open(HS_FILE, "w") as f:
        f.write(str(score))

# ── MediaPipe ──────────────────────────────────────────────────────────────────
vision      = mp.tasks.vision
RunningMode = vision.RunningMode

pose_opts = vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path="pose_landmarker.task"),
    running_mode=RunningMode.VIDEO,
    num_poses=4,
)
pose_lm = vision.PoseLandmarker.create_from_options(pose_opts)

RAISE_THRESHOLD = 0.18
VIS_MIN         = 0.5

def _any_hand_raised(pose_result) -> bool:
    for lms in pose_result.pose_landmarks:
        for wi, si in ((15, 11), (16, 12)):
            w, s = lms[wi], lms[si]
            if w.visibility < VIS_MIN or s.visibility < VIS_MIN:
                continue
            if (s.y - w.y) > RAISE_THRESHOLD:
                return True
    return False

# ── Canvas dims — set once we read the first frame ────────────────────────────
GW = GH = None   # filled in main loop on first frame
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ── Colours ────────────────────────────────────────────────────────────────────
BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
GRAY  = (160, 160, 160)

FRUIT_DEFS = [
    {"name": "watermelon", "outer": ( 34, 139,  34), "inner": ( 50, 205,  50), "points": 1},
    {"name": "orange",     "outer": (  0, 100, 255), "inner": (  0, 165, 255), "points": 1},
    {"name": "apple",      "outer": ( 50,  50, 200), "inner": ( 80,  80, 255), "points": 1},
    {"name": "banana",     "outer": (  0, 200, 220), "inner": (  0, 235, 255), "points": 2},
    {"name": "pineapple",  "outer": (  0, 180, 255), "inner": ( 50, 220, 255), "points": 2},
]
BOMB_OUTER = ( 30,  30,  30)
BOMB_INNER = (170, 170, 170)

# ── Physics (normalised to GH so it scales with resolution) ───────────────────
# These are set after GW/GH are known — see _init_physics()
GRAVITY  = None
MIN_VX = MAX_VX = MIN_VY = MAX_VY = None
FRUIT_R  = None
BOMB_R   = None

def _init_physics():
    global GRAVITY, MIN_VX, MAX_VX, MIN_VY, MAX_VY, FRUIT_R, BOMB_R
    GRAVITY = GH * 3.8
    MIN_VX  = GW * 0.08
    MAX_VX  = GW * 0.28
    MIN_VY  = GH * -2.10
    MAX_VY  = GH * -1.50
    FRUIT_R = max(24, GH // 15)
    BOMB_R  = max(20, GH // 17)

# ── Slicing ────────────────────────────────────────────────────────────────────
SLICE_MIN_SPEED = None   # set after GW known

def _init_slice_speed():
    global SLICE_MIN_SPEED
    SLICE_MIN_SPEED = GW * 0.28

SLICE_HISTORY   = 5
SLICE_FLASH_DUR = 0.30

# ── Spawning ───────────────────────────────────────────────────────────────────
BASE_SPAWN_INTERVAL = 1.3
MIN_SPAWN_INTERVAL  = 0.45
BOMB_CHANCE_START   = 0.10
BOMB_CHANCE_MAX     = 0.22
RAMP_DURATION       = 90.0


# ── Fruit / bomb objects ───────────────────────────────────────────────────────
class Obj:
    def __init__(self, x, vx, vy, is_bomb, fruit_def):
        self.x       = float(x)
        self.y       = float(GH + FRUIT_R)
        self.vx      = float(vx)
        self.vy      = float(vy)
        self.is_bomb = is_bomb
        self.fd      = fruit_def
        self.r       = BOMB_R if is_bomb else FRUIT_R
        self.sliced  = False
        self.slice_t = 0.0
        self.half_dx = [0.0, 0.0]
        self.half_dy = [0.0, 0.0]

    def tick(self, dt):
        self.x  += self.vx * dt
        self.y  += self.vy * dt
        self.vy += GRAVITY * dt
        if self.sliced:
            self.half_dx[0] -= GW * 0.06 * dt
            self.half_dx[1] += GW * 0.06 * dt
            self.half_dy[0] -= GH * 0.06 * dt
            self.half_dy[1] += GH * 0.06 * dt

    def off_screen(self):
        return self.y > GH + FRUIT_R * 4

    def centre(self):
        return int(self.x), int(self.y)

    def hits(self, px, py):
        return math.hypot(self.x - px, self.y - py) < self.r


# ── Drawing helpers ────────────────────────────────────────────────────────────
def _draw_fruit(img, obj):
    cx, cy = obj.centre()
    r = obj.r

    if obj.sliced:
        for i, (dx, dy) in enumerate(zip(obj.half_dx, obj.half_dy)):
            hx, hy = int(cx + dx), int(cy + dy)
            a0, a1 = i * 180, i * 180 + 180
            oc = BOMB_OUTER if obj.is_bomb else obj.fd["outer"]
            ic = BOMB_INNER if obj.is_bomb else obj.fd["inner"]
            cv2.ellipse(img, (hx, hy), (r, r),     0, a0, a1, oc, -1)
            cv2.ellipse(img, (hx, hy), (r - 5, r - 5), 0, a0, a1, ic, -1)
        return

    if obj.is_bomb:
        cv2.circle(img, (cx, cy), r,     BOMB_OUTER, -1)
        cv2.circle(img, (cx, cy), r - 5, BOMB_INNER, -1)
        cv2.line(img,   (cx, cy - r), (cx + 8, cy - r - 14), (0, 180, 255), 3)
        cv2.circle(img, (cx + 10, cy - r - 16), 4, (0, 230, 255), -1)
        cv2.putText(img, "X", (cx - 9, cy + 7), FONT, 0.65, (20, 20, 20), 2, cv2.LINE_AA)
    else:
        cv2.circle(img, (cx, cy), r,     obj.fd["outer"], -1)
        cv2.circle(img, (cx, cy), r - 5, obj.fd["inner"], -1)
        cv2.circle(img, (cx - r // 3, cy - r // 3), max(3, r // 5), WHITE, -1)
        lbl = obj.fd["name"][0].upper()
        (tw, _), _ = cv2.getTextSize(lbl, FONT, 0.55, 1)
        cv2.putText(img, lbl, (cx - tw // 2, cy + 6), FONT, 0.55, (20, 20, 20), 2, cv2.LINE_AA)


def _draw_text(img, text, x, y, scale, color, thickness=1):
    cv2.putText(img, text, (x, y), FONT, scale, BLACK, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, scale, color, thickness,     cv2.LINE_AA)


def _draw_centered(img, text, y, scale, color, thickness):
    (tw, _), _ = cv2.getTextSize(text, FONT, scale, thickness)
    _draw_text(img, text, (GW - tw) // 2, y, scale, color, thickness)


# ── Splashes ───────────────────────────────────────────────────────────────────
_splashes = []

def _add_splash(cx, cy, color):
    _splashes.append({"x": cx, "y": cy, "r": FRUIT_R + 4, "col": color, "born": time.time()})

def _draw_splashes(img):
    now  = time.time()
    keep = []
    for sp in _splashes:
        age = now - sp["born"]
        if age > 0.45:
            continue
        alpha   = 1.0 - age / 0.45
        r       = int(sp["r"] + age * GH * 0.12)
        overlay = img.copy()
        cv2.circle(overlay, (sp["x"], sp["y"]), r, sp["col"], -1)
        cv2.addWeighted(overlay, alpha * 0.50, img, 1 - alpha * 0.50, 0, img)
        keep.append(sp)
    _splashes[:] = keep


# ── Slice trail ────────────────────────────────────────────────────────────────
_slice_trail = []   # [(x, y, t)]

def _add_trail_point(x, y):
    _slice_trail.append((x, y, time.time()))

def _draw_trail(img):
    now  = time.time()
    keep = [(x, y, t) for x, y, t in _slice_trail if now - t < 0.30]
    _slice_trail[:] = keep
    if len(keep) < 2:
        return
    for i in range(1, len(keep)):
        age   = now - keep[i][1]
        alpha = max(0.0, 1.0 - age / 0.30)
        thick = max(1, int(5 * alpha))
        col   = (int(200 * alpha), int(235 * alpha), int(255 * alpha))
        cv2.line(img, (keep[i-1][0], keep[i-1][1]), (keep[i][0], keep[i][1]),
                 col, thick, cv2.LINE_AA)


# ── Game logic ─────────────────────────────────────────────────────────────────
class FruitNinja:
    WAITING = "waiting"
    PLAYING = "playing"
    DEAD    = "dead"

    def __init__(self):
        self.state       = self.WAITING
        self.score       = 0
        self.highscore   = load_highscore()
        self.objs        = []
        self.next_spawn  = 0.0
        self.play_start  = None
        self.dead_time   = None
        self.slice_flash = 0.0
        self._hand_was_raised = False
        self._wrist_hist = {"L": [], "R": []}

    def _elapsed(self):
        return time.time() - self.play_start if self.play_start else 0

    def _spawn_interval(self):
        t = min(self._elapsed(), RAMP_DURATION)
        return BASE_SPAWN_INTERVAL - (BASE_SPAWN_INTERVAL - MIN_SPAWN_INTERVAL) * (t / RAMP_DURATION)

    def _bomb_chance(self):
        t = min(self._elapsed(), RAMP_DURATION)
        return BOMB_CHANCE_START + (BOMB_CHANCE_MAX - BOMB_CHANCE_START) * (t / RAMP_DURATION)

    def _spawn_obj(self):
        is_bomb = random.random() < self._bomb_chance()
        fd      = None if is_bomb else random.choice(FRUIT_DEFS)
        x       = random.randint(int(GW * 0.08), int(GW * 0.92))
        vx      = random.uniform(-MAX_VX, MAX_VX)
        vy      = random.uniform(MIN_VY, MAX_VY)
        self.objs.append(Obj(x, vx, vy, is_bomb, fd))

    def _try_slice(self, pose_result):
        now         = time.time()
        newly_sliced = []

        for lms in pose_result.pose_landmarks:
            for side, wi in (("L", 15), ("R", 16)):
                w = lms[wi]
                if w.visibility < VIS_MIN:
                    continue
                gx = int(w.x * GW)
                gy = int(w.y * GH)

                hist = self._wrist_hist[side]
                hist.append((gx, gy, now))
                while len(hist) > SLICE_HISTORY:
                    hist.pop(0)

                if len(hist) < 2:
                    continue

                ox, oy, ot = hist[0]
                speed = math.hypot(gx - ox, gy - oy) / max(now - ot, 0.001)

                if speed < SLICE_MIN_SPEED:
                    continue

                _add_trail_point(gx, gy)

                for obj in self.objs:
                    if obj.sliced:
                        continue
                    if obj.hits(gx, gy):
                        obj.sliced  = True
                        obj.slice_t = now
                        newly_sliced.append(obj)

        return newly_sliced

    def tick(self, pose_result, dt):
        hand_raised = _any_hand_raised(pose_result)

        if self.state == self.WAITING:
            if hand_raised and not self._hand_was_raised:
                self._start()
            self._hand_was_raised = hand_raised
            return

        if self.state == self.DEAD:
            elapsed_dead = time.time() - self.dead_time if self.dead_time else 999
            if elapsed_dead > 1.8 and hand_raised and not self._hand_was_raised:
                self.__init__()
                self._start()
            self._hand_was_raised = hand_raised
            return

        # PLAYING
        self.slice_flash = max(0.0, self.slice_flash - dt)
        self.score = int(self._elapsed() * 8)

        now = time.time()
        if now >= self.next_spawn:
            self._spawn_obj()
            self.next_spawn = now + self._spawn_interval()

        sliced = self._try_slice(pose_result)
        for obj in sliced:
            if obj.is_bomb:
                self._die()
                return
            self.score      += obj.fd["points"] * 10
            self.slice_flash = SLICE_FLASH_DUR
            _add_splash(int(obj.x), int(obj.y), obj.fd["inner"])

        for obj in self.objs:
            obj.tick(dt)
        self.objs = [o for o in self.objs if not o.off_screen()]

        self._hand_was_raised = hand_raised

    def _start(self):
        self.state      = self.PLAYING
        self.play_start = time.time()
        self.next_spawn = time.time() + 0.5

    def _die(self):
        self.state     = self.DEAD
        self.dead_time = time.time()
        if self.score > self.highscore:
            self.highscore = self.score
            save_highscore(self.highscore)


# ── Composite draw: webcam frame + game overlays ───────────────────────────────
def draw_frame(img, gn: FruitNinja):
    """Draw everything onto img (BGR webcam frame, already flipped)."""
    h, w = img.shape[:2]

    # Splashes, trail, fruit all draw directly onto the webcam image
    _draw_splashes(img)
    _draw_trail(img)

    for obj in gn.objs:
        _draw_fruit(img, obj)

    # Slice flash: brief full-frame green tint + SLICE! text
    if gn.slice_flash > 0:
        alpha   = min(1.0, gn.slice_flash / SLICE_FLASH_DUR)
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (60, 255, 100), -1)
        cv2.addWeighted(overlay, alpha * 0.25, img, 1 - alpha * 0.25, 0, img)
        (tw, _), _ = cv2.getTextSize("SLICE!", FONT, 2.0, 5)
        tx = (w - tw) // 2
        cv2.putText(img, "SLICE!", (tx, h // 2), FONT, 2.0, BLACK,        8, cv2.LINE_AA)
        cv2.putText(img, "SLICE!", (tx, h // 2), FONT, 2.0, (60, 255, 80), 5, cv2.LINE_AA)

    # HUD — semi-transparent dark bar at top
    bar_h = 54
    bar   = img[:bar_h].copy()
    cv2.rectangle(img, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(bar, 0.35, img[:bar_h], 0.65, 0, img[:bar_h])

    _draw_text(img, f"HI  {gn.highscore:05d}", w - 230, 38, 1.0, GRAY,  1)
    _draw_text(img, f"{gn.score:05d}",          w -  92, 38, 1.0, WHITE, 1)
    _draw_text(img, "HIGH SCORE", w - 234, 16, 0.44, GRAY,  1)
    _draw_text(img, "SCORE",      w -  96, 16, 0.44, WHITE, 1)

    # State overlays
    if gn.state == FruitNinja.WAITING:
        _draw_centered(img, "FRUIT NINJA",               h // 2 - 70, 2.2, (60, 220, 60),  3)
        _draw_centered(img, "RAISE YOUR HAND TO START",  h // 2 + 10, 1.0, (0, 210, 255),  2)
        _draw_centered(img, "SWIPE YOUR ARMS TO SLICE",  h // 2 + 60, 0.75, GRAY, 1)
        _draw_centered(img, "AVOID BOMBS",                h // 2 + 100, 0.75, (80, 80, 210), 1)

    elif gn.state == FruitNinja.DEAD:
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 40), -1)
        cv2.addWeighted(overlay, 0.50, img, 0.50, 0, img)
        _draw_centered(img, "GAME OVER",            h // 2 - 60, 2.2, (60, 60, 230), 3)
        if gn.score >= gn.highscore and gn.score > 0:
            _draw_centered(img, "NEW HIGH SCORE!",  h // 2 + 10, 1.1, (0, 230, 255), 2)
        elapsed_dead = time.time() - gn.dead_time if gn.dead_time else 0
        if elapsed_dead > 1.8:
            _draw_centered(img, "RAISE HAND TO PLAY AGAIN", h // 2 + 70, 0.8, GRAY, 1)


# ── Main loop ──────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(_args.camera)

gn           = FruitNinja()
timestamp_ms = 0
prev_time    = time.time()
physics_init = False

cv2.namedWindow("Fruit Ninja", cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    now       = time.time()
    dt        = min(now - prev_time, 0.05)
    prev_time = now

    frame = cv2.flip(frame, 1)
    fh, fw = frame.shape[:2]

    # Initialise game dimensions from first real frame
    if not physics_init:
        g = globals()
        g["GW"] = fw
        g["GH"] = fh
        _init_physics()
        _init_slice_speed()
        physics_init = True

    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    pose_result  = pose_lm.detect_for_video(mp_img, timestamp_ms)
    timestamp_ms += 33

    gn.tick(pose_result, dt)

    draw_frame(frame, gn)
    cv2.imshow("Fruit Ninja", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
