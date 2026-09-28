"""
NCAD Display Programme Launcher
Arrow-key TUI to select programme, camera, and display.
"""

import ctypes
import ctypes.wintypes
import os
import subprocess
import sys
import time

# ── ANSI helpers ────────────────────────────────────────────────────────────
ESC          = "\x1b"
RESET        = f"{ESC}[0m"
BOLD         = f"{ESC}[1m"
DIM          = f"{ESC}[2m"
GREEN        = f"{ESC}[32m"
BRIGHT_GREEN = f"{ESC}[92m"
CYAN         = f"{ESC}[36m"
BRIGHT_CYAN  = f"{ESC}[96m"
YELLOW       = f"{ESC}[93m"
WHITE        = f"{ESC}[97m"
DARK_GRAY    = f"{ESC}[90m"
CLEAR_SCREEN = f"{ESC}[2J{ESC}[H"
HIDE_CURSOR  = f"{ESC}[?25l"
SHOW_CURSOR  = f"{ESC}[?25h"


def enable_ansi():
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetConsoleMode(
            ctypes.windll.kernel32.GetStdHandle(-11), 7
        )


# ── Camera detection ─────────────────────────────────────────────────────────
CAMERA_LABELS = {
    0: "Laptop Cam",
    1: "Webcam",
}

def get_cameras():
    """Return list of available camera indices by probing with cv2."""
    import cv2
    found = []
    for idx in range(8):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                found.append(idx)
        cap.release()
    return found


def camera_label(idx):
    return CAMERA_LABELS.get(idx, f"Camera {idx}")


# ── Monitor detection ────────────────────────────────────────────────────────
def get_monitors():
    monitors = []
    if sys.platform != "win32":
        return monitors
    user32 = ctypes.windll.user32

    def _cb(hmon, hdc, lprect, lparam):
        r = lprect.contents
        monitors.append((r.left, r.top, r.right, r.bottom))
        return 1

    Proc = ctypes.WINFUNCTYPE(
        ctypes.c_int,
        ctypes.c_ulong, ctypes.c_ulong,
        ctypes.POINTER(ctypes.wintypes.RECT),
        ctypes.c_double,
    )
    user32.EnumDisplayMonitors(0, 0, Proc(_cb), 0)
    return monitors


# ── Key reading ──────────────────────────────────────────────────────────────
if sys.platform == "win32":
    import msvcrt

    def read_key():
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            return {"H": "UP", "P": "DOWN", "K": "LEFT", "M": "RIGHT"}.get(
                msvcrt.getwch(), "UNKNOWN"
            )
        return {"\r": "ENTER", " ": "SPACE", "\x1b": "ESC", "\x03": "CTRL_C", "\t": "TAB"}.get(
            ch, ch.upper()
        )
else:
    import tty, termios

    def read_key():
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                seq = sys.stdin.read(2)
                return {"[A": "UP", "[B": "DOWN", "[C": "RIGHT", "[D": "LEFT"}.get(seq, "ESC")
            return {"\r": "ENTER", "\n": "ENTER", " ": "SPACE", "\x03": "CTRL_C", "\t": "TAB"}.get(
                ch, ch.upper()
            )
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ── Static data ───────────────────────────────────────────────────────────────
TITLE = [
    "  ███████╗   ██╗   ██╗███████╗",
    "  ██╔════╝   ██║   ██║██╔════╝",
    "  █████╗     ██║   ██║█████╗  ",
    "  ██╔══╝      ╚██╗██╔╝██╔══╝  ",
    "  ███████╗     ╚███╔╝ ███████╗ ",
    "  ╚══════╝      ╚══╝  ╚══════╝ ",
]
SUBTITLE = "  External Visual Experience"

PROGRAMMES = [
    {
        "id":     "ascii",
        "name":   "ASCII CAM",
        "desc":   "Live webcam feed rendered as ASCII art in the terminal",
        "detail": "Terminal rendering  │  Ctrl+C to stop",
    },
    {
        "id":     "wire",
        "name":   "WIREFRAME",
        "desc":   "Skeleton / face / hand landmark overlay on live video",
        "detail": "OpenCV window  │  Q to stop",
    },
    {
        "id":     "both",
        "name":   "BOTH",
        "desc":   "Launch Wireframe now, ASCII Cam in a new terminal window",
        "detail": "Wireframe + ASCII Cam launched simultaneously",
    },
]

# Which section the cursor is in: "prog" | "cam_ascii" | "cam_wire" | "display"
# For "both" mode there are two independent camera pickers.
# Navigation order depends on selected programme.
SECTION_ORDER = {
    "ascii": ["prog", "cam_ascii", "display"],
    "wire":  ["prog", "cam_wire",  "display"],
    "both":  ["prog", "cam_ascii", "cam_wire", "display"],
}


def hline(width, left="╔", mid="═", right="╗", color=DARK_GRAY):
    return f"{color}{left}{mid * (width - 2)}{right}{RESET}"


# ── Rendering ────────────────────────────────────────────────────────────────
def render(state):
    w = 52
    prog_id  = PROGRAMMES[state["prog"]]["id"]
    sections = SECTION_ORDER[prog_id]
    # If only 1 monitor, drop "display" from sections
    if len(state["monitors"]) <= 1 and "display" in sections:
        sections = [s for s in sections if s != "display"]
    active = state["section"] if state["section"] in sections else sections[0]

    lines = [CLEAR_SCREEN, HIDE_CURSOR]

    # Title
    lines.append(hline(w, color=GREEN))
    for tl in TITLE:
        lines.append(f"{GREEN}│{RESET}{BRIGHT_GREEN}{tl[:w-2]:<{w-2}}{RESET}{GREEN}│{RESET}")
    lines.append(hline(w, "╠", "═", "╣", color=GREEN))
    lines.append(f"{GREEN}│{RESET}{CYAN}{SUBTITLE:<{w-2}}{RESET}{GREEN}│{RESET}")
    lines.append(hline(w, "╚", "═", "╝", color=GREEN))
    lines.append("")

    # ── Programme section ────────────────────────────────────────────────────
    sec_active = active == "prog"
    hint = "↑↓ select" if sec_active else "Tab to focus"
    lines.append(
        f"  {BOLD}{WHITE if sec_active else DARK_GRAY}SELECT PROGRAMME{RESET}"
        f"  {DARK_GRAY}({hint}){RESET}"
    )
    lines.append("")
    for i, prog in enumerate(PROGRAMMES):
        is_sel = i == state["prog"]
        if is_sel and sec_active:
            prefix   = f"  {BRIGHT_GREEN}▶ {RESET}"
            name_col = f"{BOLD}{BRIGHT_GREEN}"
            desc_col = WHITE
        elif is_sel:
            prefix   = f"  {DARK_GRAY}▶ {RESET}"
            name_col = f"{BOLD}{DARK_GRAY}"
            desc_col = DARK_GRAY
        else:
            prefix   = "    "
            name_col = DARK_GRAY
            desc_col = DARK_GRAY
        lines.append(f"{prefix}{name_col}{prog['name']}{RESET}")
        lines.append(f"      {desc_col}{prog['desc']}{RESET}")
        if is_sel:
            lines.append(f"      {DIM}{DARK_GRAY}{prog['detail']}{RESET}")
        lines.append("")

    # ── Camera section(s) ────────────────────────────────────────────────────
    cameras = state["cameras"]
    if not cameras:
        lines.append(f"  {YELLOW}⚠  No cameras detected.{RESET}")
        lines.append("")
    else:
        cam_sections = [s for s in ["cam_ascii", "cam_wire"] if s in sections]
        cam_labels   = {"cam_ascii": "ASCII CAM CAMERA", "cam_wire": "WIREFRAME CAMERA"}
        cam_keys     = {"cam_ascii": "cam_ascii", "cam_wire": "cam_wire"}

        for sec_key in cam_sections:
            sec_active = active == sec_key
            hint = "← → select" if sec_active else "Tab to focus"
            lines.append(hline(w, "╔", "─", "╗", color=DARK_GRAY))
            lines.append(
                f"{DARK_GRAY}│{RESET}  "
                f"{BOLD}{WHITE if sec_active else DARK_GRAY}{cam_labels[sec_key]}{RESET}"
                f"  {DARK_GRAY}({hint}){RESET}"
            )
            lines.append(f"{DARK_GRAY}│{RESET}")
            for i, cam_idx in enumerate(cameras):
                is_sel = i == state[sec_key]
                lbl = f"[{cam_idx}]  {camera_label(cam_idx)}"
                if is_sel and sec_active:
                    indicator = f"  {BRIGHT_CYAN}▶ {BOLD}{lbl}{RESET}"
                elif is_sel:
                    indicator = f"  {DARK_GRAY}▶ {lbl}{RESET}"
                else:
                    indicator = f"    {DARK_GRAY}{lbl}{RESET}"
                lines.append(f"{DARK_GRAY}│{RESET}{indicator}")
            lines.append(f"{DARK_GRAY}│{RESET}")
            lines.append(hline(w, "╚", "─", "╝", color=DARK_GRAY))
            lines.append("")

    # ── Display section ──────────────────────────────────────────────────────
    monitors = state["monitors"]
    if len(monitors) > 1:
        sec_active = active == "display"
        hint = "← → select" if sec_active else "Tab to focus"
        lines.append(hline(w, "╔", "─", "╗", color=DARK_GRAY))
        lines.append(
            f"{DARK_GRAY}│{RESET}  "
            f"{BOLD}{WHITE if sec_active else DARK_GRAY}SELECT DISPLAY{RESET}"
            f"  {DARK_GRAY}({hint}){RESET}"
        )
        lines.append(f"{DARK_GRAY}│{RESET}")
        for i, (l, t, r, b) in enumerate(monitors):
            is_sel = i == state["display"]
            tag = f"Display {i + 1}  {r - l}×{b - t}"
            if is_sel and sec_active:
                indicator = f"  {BRIGHT_CYAN}▶ {BOLD}{tag}{RESET}"
            elif is_sel:
                indicator = f"  {DARK_GRAY}▶ {tag}{RESET}"
            else:
                indicator = f"    {DARK_GRAY}{tag}{RESET}"
            lines.append(f"{DARK_GRAY}│{RESET}{indicator}")
        lines.append(f"{DARK_GRAY}│{RESET}")
        lines.append(hline(w, "╚", "─", "╝", color=DARK_GRAY))
        lines.append("")

    # ── Warnings ─────────────────────────────────────────────────────────────
    if prog_id == "both" and cameras:
        ascii_cam = cameras[state["cam_ascii"]]
        wire_cam  = cameras[state["cam_wire"]]
        if ascii_cam == wire_cam:
            lines.append(
                f"  {YELLOW}⚠  Both programmes are set to {camera_label(ascii_cam)} "
                f"[{ascii_cam}]. One may fail to open.{RESET}"
            )
            lines.append("")

    lines.append(f"  {DARK_GRAY}Tab to switch section  │  Enter to launch  │  Esc to quit{RESET}")
    print("\n".join(lines), end="", flush=True)


# ── Launch helpers ────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON     = sys.executable


def _move_cv2_window(title, monitor):
    if sys.platform != "win32":
        return
    user32 = ctypes.windll.user32
    l, t, r, b = monitor
    for _ in range(40):
        hwnd = user32.FindWindowW(None, title)
        if hwnd:
            user32.MoveWindow(hwnd, l + 40, t + 40, (r - l) - 80, (b - t) - 80, True)
            return
        time.sleep(0.1)


def _open_new_window(title, script, extra_args, monitor=None, monitors=None):
    """Open a script in a fully detached new cmd window."""
    cam_args = " ".join(extra_args)
    # Try Windows Terminal first; fall back to cmd /k so the window stays open
    mon_str = ""
    if monitors and monitor is not None and len(monitors) > 1:
        l, t = monitors[monitor][:2]
        mon_str = f"--pos {l+40},{t+40} "
    wt = f'wt.exe {mon_str}-- {PYTHON} "{script}" {cam_args}'
    cmd = f'start "{title}" cmd /k "{PYTHON}" "{script}" {cam_args}'
    try:
        subprocess.Popen(wt, shell=True, cwd=SCRIPT_DIR)
    except Exception:
        subprocess.Popen(cmd, shell=True, cwd=SCRIPT_DIR)


def launch_ascii(camera_idx, monitor=None, monitors=None):
    script = os.path.join(SCRIPT_DIR, "ascii_cam.py")
    _open_new_window("ASCII Cam", script, ["--camera", str(camera_idx)], monitor, monitors)


def launch_wireframe(camera_idx, monitor=None, monitors=None):
    import threading
    script = os.path.join(SCRIPT_DIR, "wireframe.py")
    _open_new_window("Wireframe", script, ["--camera", str(camera_idx)], monitor, monitors)
    if monitors and monitor is not None and len(monitors) > 1:
        threading.Thread(
            target=_move_cv2_window,
            args=("Wireframe", monitors[monitor]),
            daemon=True,
        ).start()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    enable_ansi()

    print(CLEAR_SCREEN + HIDE_CURSOR, end="", flush=True)
    print(f"\n  {CYAN}Detecting cameras...{RESET}", flush=True)
    cameras  = get_cameras()
    monitors = get_monitors()

    if not cameras:
        print(f"\n  {YELLOW}No cameras found. Programmes may not work correctly.{RESET}")
        cameras = list(range(2))   # show 0 and 1 as fallback

    # Default camera selections: prefer index 1 if available, else 0
    default_cam = 1 if 1 in cameras else cameras[0]
    default_pos = cameras.index(default_cam)

    state = {
        "prog":     0,
        "cam_ascii": default_pos,
        "cam_wire":  default_pos,
        "display":  0,
        "section":  "prog",
        "cameras":  cameras,
        "monitors": monitors,
    }

    def active_sections():
        prog_id  = PROGRAMMES[state["prog"]]["id"]
        secs     = list(SECTION_ORDER[prog_id])
        if len(monitors) <= 1 and "display" in secs:
            secs.remove("display")
        return secs

    while True:
        render(state)
        key = read_key()

        if key in ("ESC", "CTRL_C", "Q"):
            break

        secs = active_sections()
        cur  = state["section"] if state["section"] in secs else secs[0]
        idx  = secs.index(cur)

        if key == "TAB":
            state["section"] = secs[(idx + 1) % len(secs)]

        elif cur == "prog":
            if key == "UP":
                state["prog"] = (state["prog"] - 1) % len(PROGRAMMES)
                state["section"] = "prog"
            elif key == "DOWN":
                state["prog"] = (state["prog"] + 1) % len(PROGRAMMES)
                state["section"] = "prog"

        elif cur in ("cam_ascii", "cam_wire"):
            if key == "LEFT":
                state[cur] = (state[cur] - 1) % len(cameras)
            elif key == "RIGHT":
                state[cur] = (state[cur] + 1) % len(cameras)

        elif cur == "display":
            if key == "LEFT":
                state["display"] = (state["display"] - 1) % len(monitors)
            elif key == "RIGHT":
                state["display"] = (state["display"] + 1) % len(monitors)

        if key == "ENTER":
            prog_id   = PROGRAMMES[state["prog"]]["id"]
            mon       = state["display"] if len(monitors) > 1 else None
            ascii_cam = cameras[state["cam_ascii"]] if cameras else 1
            wire_cam  = cameras[state["cam_wire"]]  if cameras else 1

            if prog_id == "ascii":
                launch_ascii(ascii_cam, mon, monitors)
            elif prog_id == "wire":
                launch_wireframe(wire_cam, mon, monitors)
            elif prog_id == "both":
                launch_wireframe(wire_cam, mon, monitors)
                launch_ascii(ascii_cam, mon, monitors)

            state["section"] = "prog"

    print(CLEAR_SCREEN + SHOW_CURSOR, end="", flush=True)
    print(f"{DARK_GRAY}Goodbye.{RESET}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(SHOW_CURSOR, end="", flush=True)
