import argparse
import cv2
import mediapipe as mp
import numpy as np
import time

parser = argparse.ArgumentParser()
parser.add_argument("--camera", type=int, default=1)
parser.add_argument("--left",  type=str, default="LEFT")
parser.add_argument("--right", type=str, default="RIGHT")
_args = parser.parse_args()

LEFT_CHOICE  = _args.left
RIGHT_CHOICE = _args.right

# ── MediaPipe setup ────────────────────────────────────────────────────────────
vision     = mp.tasks.vision
RunningMode = vision.RunningMode
draw        = vision.drawing_utils
PoseConn    = vision.PoseLandmarksConnections

pose_opts = vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path="pose_landmarker.task"),
    running_mode=RunningMode.VIDEO,
    num_poses=6,
)
pose_landmarker = vision.PoseLandmarker.create_from_options(pose_opts)

# ── Colours ────────────────────────────────────────────────────────────────────
GREEN       = (0, 255, 0)
GREEN_THICK = draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
GREEN_THIN  = draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
COL_BLACK   = (0,   0,   0)
COL_GREEN   = (0, 255,   0)
COL_DIM     = (0, 140,   0)
COL_LEFT    = (0, 200, 255)   # yellow-ish for left
COL_RIGHT   = (255, 100,  0)  # blue-ish for right
COL_WHITE   = (255, 255, 255)
FONT        = cv2.FONT_HERSHEY_SIMPLEX

# ── Vote state ─────────────────────────────────────────────────────────────────
votes_left  = 0
votes_right = 0

# Per-person raise state: track whether each pose slot currently has a hand up
# and whether they have already voted this raise (must lower to vote again).
# Key = pose index, value = {"left": bool, "right": bool}
# "raised" = hand is currently up; "voted" = counted this raise already
_person_state: dict[int, dict] = {}

RAISE_THRESHOLD = 0.20   # wrist y must be this much above shoulder y (normalised)
HOLD_SECONDS    = 0.7    # must hold the raise for this long before vote counts


def _is_raised(landmarks, wrist_idx, shoulder_idx) -> bool:
    w = landmarks[wrist_idx]
    s = landmarks[shoulder_idx]
    # y increases downward in image space; raised hand has smaller y
    return (s.y - w.y) > RAISE_THRESHOLD and w.visibility > 0.6 and s.visibility > 0.6


def process_poses(pose_result):
    global votes_left, votes_right, _person_state

    now = time.time()
    active_ids = set()

    for i, lms in enumerate(pose_result.pose_landmarks):
        active_ids.add(i)
        if i not in _person_state:
            _person_state[i] = {
                "left_raised": False,  "left_voted": False,  "left_held_since": None,
                "right_raised": False, "right_voted": False, "right_held_since": None,
            }

        ps = _person_state[i]

        # MediaPipe indices: 15=left wrist, 11=left shoulder, 16=right wrist, 12=right shoulder
        # After cv2.flip the image is mirrored, so anatomical left appears on screen-right
        # and anatomical right appears on screen-left.  We swap the labels here so that
        # "left" in our poll means the person's left as they see themselves (screen-left).
        # screen-left = anatomical right = wrist 16, shoulder 12
        # screen-right = anatomical left  = wrist 15, shoulder 11
        screen_left_up  = _is_raised(lms, 16, 12)   # anatomical right → screen left
        screen_right_up = _is_raised(lms, 15, 11)   # anatomical left  → screen right

        for side, is_up in (("left", screen_left_up), ("right", screen_right_up)):
            if is_up:
                if ps[f"{side}_held_since"] is None:
                    ps[f"{side}_held_since"] = now
                held = now - ps[f"{side}_held_since"]
                ps[f"{side}_raised"] = True
                if held >= HOLD_SECONDS and not ps[f"{side}_voted"]:
                    if side == "left":
                        votes_left += 1
                    else:
                        votes_right += 1
                    ps[f"{side}_voted"] = True
            else:
                ps[f"{side}_raised"]     = False
                ps[f"{side}_held_since"] = None
                ps[f"{side}_voted"]      = False

    for gone in list(_person_state.keys()):
        if gone not in active_ids:
            del _person_state[gone]


def draw_camera_window(img, pose_result):
    """Draw skeleton overlays and per-person vote indicators onto img (RGB)."""
    now = time.time()
    h, w = img.shape[:2]

    for i, lms in enumerate(pose_result.pose_landmarks):
        draw.draw_landmarks(img, lms, PoseConn.POSE_LANDMARKS, GREEN_THICK, GREEN_THIN)

        ps = _person_state.get(i, {})
        left_raised  = ps.get("left_raised",  False)
        right_raised = ps.get("right_raised", False)

        if not (left_raised or right_raised):
            continue

        # Pick whichever side is raised (left takes priority if both somehow up)
        if left_raised:
            choice     = LEFT_CHOICE
            col        = COL_LEFT
            held_since = ps.get("left_held_since")
            voted      = ps.get("left_voted", False)
        else:
            choice     = RIGHT_CHOICE
            col        = COL_RIGHT
            held_since = ps.get("right_held_since")
            voted      = ps.get("right_voted", False)

        # Head position from nose landmark (index 0)
        nose = lms[0]
        cx = int(nose.x * w)
        cy = int(nose.y * h) - 20

        if voted:
            label = f"VOTED: {choice}"
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.7, 2)
            tx = max(4, min(cx - tw // 2, w - tw - 4))
            ty = max(th + 4, cy)
            cv2.putText(img, label, (tx, ty), FONT, 0.7, COL_BLACK, 4, cv2.LINE_AA)
            cv2.putText(img, label, (tx, ty), FONT, 0.7, col,       2, cv2.LINE_AA)
        elif held_since is not None:
            # Show a hold progress bar
            pct    = min(1.0, (now - held_since) / HOLD_SECONDS)
            bar_w  = 120
            bar_h  = 14
            bx     = max(4, min(cx - bar_w // 2, w - bar_w - 4))
            by     = max(bar_h + 20, cy)

            label = f"HOLD: {choice}"
            (tw, _), _ = cv2.getTextSize(label, FONT, 0.55, 1)
            tx = max(4, min(cx - tw // 2, w - tw - 4))
            cv2.putText(img, label, (tx, by - bar_h - 4), FONT, 0.55, COL_BLACK, 3, cv2.LINE_AA)
            cv2.putText(img, label, (tx, by - bar_h - 4), FONT, 0.55, col,       1, cv2.LINE_AA)

            cv2.rectangle(img, (bx, by), (bx + bar_w, by + bar_h), COL_DIM, -1)
            cv2.rectangle(img, (bx, by), (bx + int(bar_w * pct), by + bar_h), col, -1)
            cv2.rectangle(img, (bx, by), (bx + bar_w, by + bar_h), col, 1)

    # HUD: mirror-aware — on screen, left side = left choice, right side = right choice
    instructions = [
        f"Raise LEFT  hand  ->  {LEFT_CHOICE}",
        f"Raise RIGHT hand  ->  {RIGHT_CHOICE}",
        "Hold for 0.7s to vote  |  Q to quit",
    ]
    for j, txt in enumerate(instructions):
        y = 28 + j * 26
        cv2.putText(img, txt, (10, y), FONT, 0.5, COL_BLACK, 3, cv2.LINE_AA)
        cv2.putText(img, txt, (10, y), FONT, 0.5, COL_DIM,   1, cv2.LINE_AA)

def draw_poll_window(img):
    """Draw the poll results onto a black canvas."""
    h, w = img.shape[:2]
    img[:] = 0

    total = votes_left + votes_right

    # ── Title ──────────────────────────────────────────────────────────────────
    title = "LIVE POLL"
    (tw, _), _ = cv2.getTextSize(title, FONT, 1.4, 3)
    cv2.putText(img, title, ((w - tw) // 2, 70), FONT, 1.4, COL_GREEN, 3, cv2.LINE_AA)

    # ── Instructions ──────────────────────────────────────────────────────────
    sub = f"Raise left hand: {LEFT_CHOICE}   |   Raise right hand: {RIGHT_CHOICE}"
    (sw, _), _ = cv2.getTextSize(sub, FONT, 0.6, 1)
    cv2.putText(img, sub, ((w - sw) // 2, 115), FONT, 0.6, COL_DIM, 1, cv2.LINE_AA)

    # Divider
    cv2.line(img, (40, 135), (w - 40, 135), COL_DIM, 1)

    # ── Choice blocks ──────────────────────────────────────────────────────────
    block_y   = 175
    block_h   = h - block_y - 80
    margin    = 50
    gap       = 30
    block_w   = (w - 2 * margin - gap) // 2

    left_x  = margin
    right_x = margin + block_w + gap

    for side, bx, count, choice, col in [
        ("LEFT",  left_x,  votes_left,  LEFT_CHOICE,  COL_LEFT),
        ("RIGHT", right_x, votes_right, RIGHT_CHOICE, COL_RIGHT),
    ]:
        # Border rect
        cv2.rectangle(img, (bx, block_y), (bx + block_w, block_y + block_h), col, 2)

        # Side label
        (lw, _), _ = cv2.getTextSize(f"RAISE {side} HAND", FONT, 0.55, 1)
        cv2.putText(img, f"RAISE {side} HAND",
                    (bx + (block_w - lw) // 2, block_y + 38),
                    FONT, 0.55, col, 1, cv2.LINE_AA)

        # Choice name
        (cw, _), _ = cv2.getTextSize(choice, FONT, 1.1, 2)
        cv2.putText(img, choice,
                    (bx + (block_w - cw) // 2, block_y + 90),
                    FONT, 1.1, COL_WHITE, 2, cv2.LINE_AA)

        # Vote count (big)
        cnt_str = str(count)
        (nw, _), _ = cv2.getTextSize(cnt_str, FONT, 3.5, 5)
        cv2.putText(img, cnt_str,
                    (bx + (block_w - nw) // 2, block_y + block_h - 80),
                    FONT, 3.5, col, 5, cv2.LINE_AA)

        # Percentage bar
        bar_x  = bx + 20
        bar_y  = block_y + block_h - 45
        bar_w  = block_w - 40
        bar_bh = 18
        pct    = count / total if total > 0 else 0
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_bh), COL_DIM, -1)
        if pct > 0:
            cv2.rectangle(img, (bar_x, bar_y),
                          (bar_x + int(bar_w * pct), bar_y + bar_bh), col, -1)
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_bh), col, 1)

        pct_lbl = f"{pct * 100:.0f}%"
        (pw, _), _ = cv2.getTextSize(pct_lbl, FONT, 0.5, 1)
        cv2.putText(img, pct_lbl,
                    (bar_x + (bar_w - pw) // 2, bar_y + bar_bh - 3),
                    FONT, 0.5, COL_BLACK, 2, cv2.LINE_AA)
        cv2.putText(img, pct_lbl,
                    (bar_x + (bar_w - pw) // 2, bar_y + bar_bh - 3),
                    FONT, 0.5, COL_WHITE, 1, cv2.LINE_AA)

    # Total votes footer
    footer = f"Total votes: {total}"
    (fw, _), _ = cv2.getTextSize(footer, FONT, 0.65, 1)
    cv2.putText(img, footer, ((w - fw) // 2, h - 20), FONT, 0.65, COL_DIM, 1, cv2.LINE_AA)


# ── Main loop ──────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(_args.camera)

POLL_W, POLL_H = 900, 600
poll_canvas = np.zeros((POLL_H, POLL_W, 3), dtype=np.uint8)

cv2.namedWindow("Poll Results", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Poll Results", POLL_W, POLL_H)
cv2.namedWindow("Poll Camera",   cv2.WINDOW_NORMAL)

timestamp_ms = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    pose_result = pose_landmarker.detect_for_video(mp_img, timestamp_ms)

    process_poses(pose_result)

    # Camera window
    annotated = np.copy(rgb)
    draw_camera_window(annotated, pose_result)
    cam_out = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    cv2.imshow("Poll Camera", cam_out)

    # Poll window
    draw_poll_window(poll_canvas)
    cv2.imshow("Poll Results", poll_canvas)

    timestamp_ms += 33
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
