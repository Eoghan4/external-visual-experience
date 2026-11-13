"""
GLITCH CAM — live camera feed with cycling glitch/datamosh effects.
E to cycle effects manually, Q to quit.
"""

import argparse
import cv2
import numpy as np
import random
import time

parser = argparse.ArgumentParser()
parser.add_argument("--camera", type=int, default=1)
_args = parser.parse_args()

cap = cv2.VideoCapture(_args.camera, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("GLITCH CAM", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("GLITCH CAM", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

EFFECTS = [
    "PIXEL SORT",
    "DATAMOSH",
    "CHROMATIC SPLIT",
    "VHS NOISE",
    "RGB MELT",
    "BLOCK GLITCH",
]

effect_idx   = 0

# Datamosh state
prev_frame      = None          # previous raw frame (BGR)
prev_gray       = None          # previous grayscale for optical flow
mosh_buffer     = None          # accumulated warp canvas
mosh_flow_accum = None          # accumulated flow field (h,w,2) float32
mosh_mode       = 0             # sub-mode index — changed only by E key
MOSH_MODES      = ["FLOW WARP", "I-FRAME DROP", "DELTA ECHO", "FEEDBACK STORM"]

# Pre-allocated remap coordinate grids — rebuilt only if frame size changes
_remap_base_x: np.ndarray | None = None
_remap_base_y: np.ndarray | None = None
_remap_shape: tuple = (0, 0)

# Half-res scratch buffers for optical flow (allocated on first use)
_flow_h: int = 0
_flow_w: int = 0

# Block glitch state
glitch_blocks: list[tuple] = []
glitch_timer = 0


# ── Effects ───────────────────────────────────────────────────────────────────

def pixel_sort(frame: np.ndarray) -> np.ndarray:
    """Sort pixel columns by brightness — bright columns drift upward."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    brightness = hsv[:, :, 2].astype(np.float32) / 255.0

    out = frame.copy()
    h, w = frame.shape[:2]

    # Sort every other column based on brightness threshold
    threshold = 0.45
    for x in range(0, w, 2):
        col = frame[:, x].copy()
        lum = brightness[:, x]
        mask = lum > threshold
        if mask.sum() < 2:
            continue
        indices = np.where(mask)[0]
        start, end = indices[0], indices[-1] + 1
        segment = col[start:end]
        # Sort segment rows by their luminance
        seg_lum = lum[start:end]
        order = np.argsort(seg_lum)
        col[start:end] = segment[order]
        out[:, x] = col

    return out


def _remap_warp(src: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Warp src by flow. Reuses pre-allocated coordinate grids."""
    global _remap_base_x, _remap_base_y, _remap_shape
    h, w = src.shape[:2]
    if (h, w) != _remap_shape:
        _remap_base_y, _remap_base_x = np.mgrid[0:h, 0:w].astype(np.float32)
        _remap_shape = (h, w)
    map_x = _remap_base_x + flow[:, :, 0]
    map_y = _remap_base_y + flow[:, :, 1]
    return cv2.remap(src, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _compute_flow(gray_prev: np.ndarray, gray_curr: np.ndarray) -> np.ndarray:
    """
    Farneback dense optical flow at half resolution, result scaled back up.
    Running at half-res cuts cost to ~25% of full-res.
    """
    global _flow_h, _flow_w
    h, w = gray_prev.shape
    fh, fw = h // 2, w // 2
    if fh != _flow_h or fw != _flow_w:
        _flow_h, _flow_w = fh, fw
    small_prev = cv2.resize(gray_prev, (fw, fh), interpolation=cv2.INTER_AREA)
    small_curr = cv2.resize(gray_curr, (fw, fh), interpolation=cv2.INTER_AREA)
    flow_small = cv2.calcOpticalFlowFarneback(
        small_prev, small_curr,
        None,
        pyr_scale=0.5, levels=3, winsize=13,
        iterations=2, poly_n=5, poly_sigma=1.2,
        flags=0,
    )
    # Scale back to full resolution — multiply displacement values by 2
    flow_full = cv2.resize(flow_small, (w, h), interpolation=cv2.INTER_LINEAR) * 2.0
    return flow_full


def datamosh(frame: np.ndarray, sub_mode: str) -> np.ndarray:
    """
    True optical-flow datamosh with four sub-modes:

    FLOW WARP      — accumulate flow across frames and warp the mosh buffer
                     forward. Moving objects smear in their motion direction.
    I-FRAME DROP   — periodically inject a fresh frame (I-frame) then let it
                     drift purely by accumulated flow — classic 'no-I-frame' look.
    DELTA ECHO     — only the motion delta (diff between frames) is composited
                     back; static regions turn black, motion trails burn in.
    FEEDBACK STORM — flow is amplified and fed back into itself each frame,
                     causing chaotic pixel avalanches.
    """
    global prev_frame, prev_gray, mosh_buffer, mosh_flow_accum

    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_gray is None or mosh_buffer is None:
        prev_gray       = gray.copy()
        mosh_buffer     = frame.copy().astype(np.float32)
        mosh_flow_accum = np.zeros((h, w, 2), dtype=np.float32)
        return frame

    flow = _compute_flow(prev_gray, gray)         # (h,w,2) per-pixel displacement

    if sub_mode == "FLOW WARP":
        # Accumulate flow — each frame the buffer warps a little further
        mosh_flow_accum += flow * 1.6
        # Decay accumulation slightly so it doesn't run away forever
        mosh_flow_accum *= 0.94
        warped = _remap_warp(mosh_buffer.astype(np.uint8), mosh_flow_accum)
        # Blend a tiny bit of the real frame back in to stop total collapse
        mosh_buffer = cv2.addWeighted(warped.astype(np.float32), 0.97,
                                      frame.astype(np.float32), 0.03, 0)
        out = np.clip(mosh_buffer, 0, 255).astype(np.uint8)

    elif sub_mode == "I-FRAME DROP":
        # Every ~90 frames slam in a clean I-frame, then drift purely by flow
        mosh_flow_accum += flow * 2.2
        mosh_flow_accum *= 0.90
        # Probabilistic I-frame injection
        if random.random() < 0.011:
            mosh_buffer     = frame.astype(np.float32)
            mosh_flow_accum = np.zeros((h, w, 2), dtype=np.float32)
        warped      = _remap_warp(mosh_buffer.astype(np.uint8), mosh_flow_accum)
        mosh_buffer = warped.astype(np.float32)
        out         = np.clip(warped, 0, 255).astype(np.uint8)

    elif sub_mode == "DELTA ECHO":
        # Compute per-pixel motion magnitude
        mag = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
        # Threshold: only pixels with significant motion contribute
        threshold = 2.5
        motion_mask = (mag > threshold).astype(np.float32)[:, :, np.newaxis]
        # Delta frame: difference between current and previous
        prev_f  = prev_frame.astype(np.float32)
        curr_f  = frame.astype(np.float32)
        delta   = np.abs(curr_f - prev_f)
        # Burn delta into buffer where motion exists
        mosh_buffer = mosh_buffer * 0.88 + delta * motion_mask * 1.4
        # Warp buffer by flow so trails follow motion direction
        mosh_buffer = _remap_warp(
            np.clip(mosh_buffer, 0, 255).astype(np.uint8),
            flow * 1.2
        ).astype(np.float32)
        # Composite: still areas show real frame dimly; motion areas show burns
        static_mask = 1.0 - motion_mask * 0.8
        out = np.clip(
            frame.astype(np.float32) * static_mask * 0.5 + mosh_buffer,
            0, 255
        ).astype(np.uint8)

    elif sub_mode == "FEEDBACK STORM":
        # Amplify flow massively and recursively warp
        amp = 3.8
        mosh_flow_accum = mosh_flow_accum * 0.80 + flow * amp
        # Clamp to prevent runaway NaN
        mosh_flow_accum = np.clip(mosh_flow_accum, -w * 0.4, w * 0.4)
        warped = _remap_warp(mosh_buffer.astype(np.uint8), mosh_flow_accum)
        # Inject real frame weakly — storm slowly consumes the image
        mosh_buffer = cv2.addWeighted(warped.astype(np.float32), 0.92,
                                      frame.astype(np.float32), 0.08, 0)
        # Hue shift the output based on flow magnitude for extra chaos
        out_bgr  = np.clip(mosh_buffer, 0, 255).astype(np.uint8)
        mag_norm = np.clip(
            np.sqrt(mosh_flow_accum[:,:,0]**2 + mosh_flow_accum[:,:,1]**2) / (w * 0.3),
            0, 1
        )
        hsv      = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,0] = (hsv[:,:,0] + mag_norm * 60) % 180
        hsv[:,:,1] = np.clip(hsv[:,:,1] + mag_norm * 80, 0, 255)
        out      = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    else:
        out = frame

    prev_gray  = gray.copy()
    return out


def chromatic_split(frame: np.ndarray) -> np.ndarray:
    """Offset R and B channels sideways — classic chromatic aberration."""
    h, w = frame.shape[:2]
    shift = max(4, w // 60)
    b, g, r = cv2.split(frame)
    r_shifted = np.roll(r,  shift, axis=1)
    b_shifted = np.roll(b, -shift, axis=1)
    out = cv2.merge([b_shifted, g, r_shifted])

    # Also add a faint vertical green tear
    tear_x = random.randint(0, w - 1)
    out[:, tear_x, 1] = np.clip(out[:, tear_x, 1].astype(int) + 60, 0, 255)
    return out


def vhs_noise(frame: np.ndarray) -> np.ndarray:
    """VHS-style scanlines, luma noise, and head-switching glitch band."""
    out = frame.copy().astype(np.float32)
    h, w = out.shape[:2]

    # Scanlines — dim every other row
    out[::2, :] *= 0.75

    # Luma noise
    noise = np.random.normal(0, 18, out.shape).astype(np.float32)
    out = np.clip(out + noise, 0, 255)

    # Random horizontal noise band (head-switching artifact)
    band_y = random.randint(h - h // 5, h - 2)
    band_h = random.randint(1, max(2, h // 30))
    band   = out[band_y:band_y + band_h]
    band  += np.random.normal(0, 60, band.shape)
    shift  = random.randint(-w // 5, w // 5)
    out[band_y:band_y + band_h] = np.clip(np.roll(band, shift, axis=1), 0, 255)

    # Slight color desaturation to mimic tape
    hsv = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= 0.65
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    return out.astype(np.uint8)


def rgb_melt(frame: np.ndarray) -> np.ndarray:
    """Slowly rotating hue + vertical channel drip."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    # Shift hue by time
    shift = int(time.time() * 40) % 180
    hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
    # Boost saturation
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.6, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Drip: shift individual columns downward by a sine wave
    h, w = out.shape[:2]
    drip_cols = np.sin(np.linspace(0, 4 * np.pi, w) + time.time() * 2)
    drip_amount = (drip_cols * (h // 15)).astype(int)
    for x in range(w):
        shift_y = drip_amount[x]
        if shift_y != 0:
            out[:, x] = np.roll(out[:, x], shift_y, axis=0)

    return out


def block_glitch(frame: np.ndarray, blocks: list) -> np.ndarray:
    """Random displaced rectangular blocks — compressed-video artifact."""
    out = frame.copy()
    h, w = out.shape[:2]
    for (sx, sy, bw, bh, dx, dy) in blocks:
        src_x2 = min(sx + bw, w)
        src_y2 = min(sy + bh, h)
        dst_x  = max(0, sx + dx)
        dst_y  = max(0, sy + dy)
        dst_x2 = min(dst_x + (src_x2 - sx), w)
        dst_y2 = min(dst_y + (src_y2 - sy), h)
        patch_w = min(dst_x2 - dst_x, src_x2 - sx)
        patch_h = min(dst_y2 - dst_y, src_y2 - sy)
        if patch_w > 0 and patch_h > 0:
            out[dst_y:dst_y + patch_h, dst_x:dst_x + patch_w] = \
                frame[sy:sy + patch_h, sx:sx + patch_w]
    return out


def _new_glitch_blocks(w, h):
    n = random.randint(4, 14)
    blocks = []
    for _ in range(n):
        bw = random.randint(w // 12, w // 3)
        bh = random.randint(2, h // 8)
        sx = random.randint(0, w - bw)
        sy = random.randint(0, h - bh)
        dx = random.randint(-w // 6, w // 6)
        dy = random.randint(-4, 4)
        blocks.append((sx, sy, bw, bh, dx, dy))
    return blocks


def draw_hud(frame: np.ndarray, label: str, hint_extra: str = "") -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    font  = cv2.FONT_HERSHEY_SIMPLEX
    fsize = max(0.5, w / 1600)
    thick = 1

    cv2.putText(out, f"[ {label} ]", (14, h - 16), font, fsize, (0, 255, 0), thick, cv2.LINE_AA)

    hint = f"E: next effect{hint_extra}   Q: quit"
    (tw, _), _ = cv2.getTextSize(hint, font, fsize * 0.8, thick)
    cv2.putText(out, hint, (w - tw - 14, h - 16), font, fsize * 0.8, (0, 200, 0), thick, cv2.LINE_AA)

    return out


# ── Main loop ─────────────────────────────────────────────────────────────────

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)   # mirror
    h, w  = frame.shape[:2]

    # Auto-cycle effect
    now = time.time()
    effect = EFFECTS[effect_idx]

    # ── Apply selected effect ─────────────────────────────────────────────────
    if effect == "PIXEL SORT":
        out = pixel_sort(frame)

    elif effect == "DATAMOSH":
        sub = MOSH_MODES[mosh_mode]
        out = datamosh(frame, sub)

    elif effect == "CHROMATIC SPLIT":
        out = chromatic_split(frame)

    elif effect == "VHS NOISE":
        out = vhs_noise(frame)

    elif effect == "RGB MELT":
        out = rgb_melt(frame)

    elif effect == "BLOCK GLITCH":
        glitch_timer -= 1
        if glitch_timer <= 0:
            glitch_blocks = _new_glitch_blocks(w, h)
            glitch_timer  = random.randint(3, 12)
        out = block_glitch(frame, glitch_blocks)

    else:
        out = frame

    prev_frame = frame.copy()

    hud_label = f"DATAMOSH  {MOSH_MODES[mosh_mode]}" if effect == "DATAMOSH" else effect
    hint_extra = "   M: sub-mode" if effect == "DATAMOSH" else ""
    out = draw_hud(out, hud_label, hint_extra)
    cv2.imshow("GLITCH CAM", out)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), ord('Q'), 27):
        break
    if key in (ord('e'), ord('E')):
        effect_idx    = (effect_idx + 1) % len(EFFECTS)
        glitch_blocks = []
        prev_gray     = None
        mosh_buffer   = None
        mosh_flow_accum = None
    if key in (ord('m'), ord('M')) and effect == "DATAMOSH":
        mosh_mode       = (mosh_mode + 1) % len(MOSH_MODES)
        prev_gray       = None
        mosh_buffer     = None
        mosh_flow_accum = None

cap.release()
cv2.destroyAllWindows()
