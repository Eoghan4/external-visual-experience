"""
DVD Bounce — an image bounces around the screen, changing colour on wall hits.
Usage: python dvd_bounce.py [--image path] [--speed 5] [--size 400] [--transparent]
"""

import argparse
import os
import random
import sys

import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--image",       default="NCAD_Primary_Logo_Black.png")
parser.add_argument("--speed",       type=float, default=5.0)
parser.add_argument("--width",       type=int,   default=1280)
parser.add_argument("--height",      type=int,   default=720)
parser.add_argument("--size",        type=int,   default=400, help="Logo width in pixels")
parser.add_argument("--transparent", action="store_true", help="Float logo over desktop with transparent background")
parser.add_argument("--mon-x",       type=int,   default=None)
parser.add_argument("--mon-y",       type=int,   default=None)
parser.add_argument("--mon-w",       type=int,   default=None)
parser.add_argument("--mon-h",       type=int,   default=None)
args = parser.parse_args()

# If monitor dimensions were passed, use them as the canvas size
if args.mon_w:
    args.width  = args.mon_w
if args.mon_h:
    args.height = args.mon_h

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load / generate logo ──────────────────────────────────────────────────────

def _make_dvd_logo(width=200):
    h = max(80, width // 3)
    img = np.zeros((h, width, 4), dtype=np.uint8)
    font      = cv2.FONT_HERSHEY_DUPLEX
    text      = "DVD"
    scale     = width / 120
    thickness = max(2, int(scale * 2))
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(img, text, ((width - tw) // 2, (h + th) // 2),
                font, scale, (255, 255, 255, 255), thickness, cv2.LINE_AA)
    return img


def _white_bg_to_alpha(bgra):
    gray = cv2.cvtColor(bgra[:, :, :3], cv2.COLOR_BGR2GRAY)
    bgra[:, :, 3] = 255 - gray
    return bgra


def load_logo(path, target_w):
    raw = None
    if path and os.path.isfile(path):
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            print(f"Warning: could not read {path}, using built-in logo.", file=sys.stderr)
    if raw is None:
        if path:
            print(f"Warning: {path!r} not found, using built-in logo.", file=sys.stderr)
        raw = _make_dvd_logo(target_w)

    if raw.ndim == 2:
        raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGRA)
    elif raw.shape[2] == 3:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2BGRA)

    if raw[:, :, 3].min() == 255:
        raw = _white_bg_to_alpha(raw)

    aspect = raw.shape[0] / raw.shape[1]
    new_w  = target_w
    new_h  = max(1, int(target_w * aspect))
    return cv2.resize(raw, (new_w, new_h), interpolation=cv2.INTER_AREA)


logo_base = load_logo(args.image, args.size)
logo_h, logo_w = logo_base.shape[:2]

# ── Colour cycling ────────────────────────────────────────────────────────────
COLOURS = [
    (0,   255, 255),
    (255, 0,   255),
    (255, 255, 0),
    (0,   180, 255),
    (0,   255, 0),
    (255, 100, 0),
]
colour_idx = 0

def recolour(logo_bgra, bgr):
    out  = logo_bgra.copy()
    mask = out[:, :, 3] > 0
    out[mask, 0] = bgr[0]
    out[mask, 1] = bgr[1]
    out[mask, 2] = bgr[2]
    return out

def next_colour():
    global colour_idx
    colour_idx = (colour_idx + 1) % len(COLOURS)
    return COLOURS[colour_idx]

# ── Physics ───────────────────────────────────────────────────────────────────
W, H = args.width, args.height
px = float(random.randint(0, max(0, W - logo_w)))
py = float(random.randint(0, max(0, H - logo_h)))
angle = random.uniform(30, 60)
speed = args.speed
dx = speed * np.cos(np.radians(angle))
dy = speed * np.sin(np.radians(angle))

current_colour = COLOURS[colour_idx]
logo_coloured  = recolour(logo_base, current_colour)

# ── Transparent mode (tkinter + Windows colorkey) ────────────────────────────
if args.transparent:
    import tkinter as tk
    from PIL import Image, ImageTk

    CHROMA = "#000000"   # the colour that becomes transparent

    root = tk.Tk()
    root.overrideredirect(True)                        # no title bar / borders
    root.attributes("-topmost", True)                  # always on top
    root.attributes("-transparentcolor", CHROMA)       # punch holes where black is
    win_x = args.mon_x if args.mon_x is not None else 0
    win_y = args.mon_y if args.mon_y is not None else 0
    root.geometry(f"{W}x{H}+{win_x}+{win_y}")
    root.configure(bg=CHROMA)

    canvas_tk = tk.Canvas(root, width=W, height=H,
                          bg=CHROMA, highlightthickness=0)
    canvas_tk.pack()

    img_ref = None   # keep a reference so GC doesn't collect it

    def step():
        global px, py, dx, dy, current_colour, logo_coloured, img_ref

        px += dx
        py += dy
        hit = False

        if px + logo_w >= W:
            px = W - logo_w; dx = -abs(dx); hit = True
        elif px <= 0:
            px = 0;           dx =  abs(dx); hit = True
        if py + logo_h >= H:
            py = H - logo_h; dy = -abs(dy); hit = True
        elif py <= 0:
            py = 0;           dy =  abs(dy); hit = True

        if hit:
            current_colour = next_colour()
            logo_coloured  = recolour(logo_base, current_colour)

        # Build an RGBA PIL image — black background = transparent via colorkey
        frame = np.zeros((H, W, 3), dtype=np.uint8)   # black = transparent
        x, y   = int(px), int(py)
        lh, lw = logo_coloured.shape[:2]
        x1, y1 = max(x, 0), max(y, 0)
        x2, y2 = min(x + lw, W), min(y + lh, H)
        if x2 > x1 and y2 > y1:
            sx, sy = x1 - x, y1 - y
            patch  = logo_coloured[sy:sy + (y2 - y1), sx:sx + (x2 - x1)]
            a      = patch[:, :, 3:4].astype(np.float32) / 255.0
            region = frame[y1:y2, x1:x2].astype(np.float32)
            blended = region * (1 - a) + patch[:, :, :3].astype(np.float32) * a
            frame[y1:y2, x1:x2] = blended.astype(np.uint8)

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_ref  = ImageTk.PhotoImage(pil_img)
        canvas_tk.delete("all")
        canvas_tk.create_image(0, 0, anchor="nw", image=img_ref)

        root.after(16, step)   # ~60 fps

    root.bind("<KeyPress-q>", lambda e: root.destroy())
    root.bind("<KeyPress-Q>", lambda e: root.destroy())
    root.bind("<Escape>",     lambda e: root.destroy())

    root.after(0, step)
    root.mainloop()

# ── Solid-background mode (OpenCV) ────────────────────────────────────────────
else:
    def composite(canvas, logo_bgra, x, y):
        x, y = int(x), int(y)
        lh, lw = logo_bgra.shape[:2]
        x1, y1 = max(x, 0), max(y, 0)
        x2, y2 = min(x + lw, canvas.shape[1]), min(y + lh, canvas.shape[0])
        if x2 <= x1 or y2 <= y1:
            return
        sx, sy = x1 - x, y1 - y
        patch  = logo_bgra[sy:sy + (y2 - y1), sx:sx + (x2 - x1)]
        a      = patch[:, :, 3:4].astype(np.float32) / 255.0
        region = canvas[y1:y2, x1:x2].astype(np.float32)
        blended = region * (1 - a) + patch[:, :, :3].astype(np.float32) * a
        canvas[y1:y2, x1:x2] = blended.astype(np.uint8)

    WIN = "DVD Bounce"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    if args.mon_x is not None:
        import ctypes
        cv2.moveWindow(WIN, args.mon_x, args.mon_y)
        cv2.resizeWindow(WIN, W, H)
        cv2.waitKey(1)
        hwnd = ctypes.windll.user32.FindWindowW(None, WIN)
        if hwnd:
            ctypes.windll.user32.MoveWindow(hwnd, args.mon_x, args.mon_y, W, H, True)
        cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(WIN, W, H)
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    while True:
        px += dx
        py += dy
        hit = False

        if px + logo_w >= W:
            px = W - logo_w; dx = -abs(dx); hit = True
        elif px <= 0:
            px = 0;           dx =  abs(dx); hit = True
        if py + logo_h >= H:
            py = H - logo_h; dy = -abs(dy); hit = True
        elif py <= 0:
            py = 0;           dy =  abs(dy); hit = True

        if hit:
            current_colour = next_colour()
            logo_coloured  = recolour(logo_base, current_colour)

        canvas[:] = 0
        composite(canvas, logo_coloured, px, py)
        cv2.imshow(WIN, canvas)

        key = cv2.waitKey(16) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break

    cv2.destroyAllWindows()
