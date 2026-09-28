import argparse
import shutil
import sys
import time

import cv2
import numpy as np

# Dark-to-light gradient for grayscale mapping.
ASCII_CHARS = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. "
RESET = "\x1b[0m"
BACKEND_MAP = {
    "any": cv2.CAP_ANY,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
}


def frame_to_ascii(frame: np.ndarray, width: int) -> str:
    """Convert a BGR frame to an ASCII art string."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Characters are typically taller than they are wide, so scale height down.
    height, original_width = gray.shape
    if original_width == 0:
        return ""

    aspect_ratio = height / original_width
    target_height = max(1, int(aspect_ratio * width * 0.5))

    resized = cv2.resize(gray, (width, target_height), interpolation=cv2.INTER_AREA)

    indices = (resized / 255 * (len(ASCII_CHARS) - 1)).astype(np.int32)
    rows = ["".join(ASCII_CHARS[i] for i in row) for row in indices]
    return "\n".join(rows)


def terminal_ascii_width(scale: float) -> int:
    cols = shutil.get_terminal_size((120, 40)).columns
    # Leave room so lines do not wrap and destroy the frame layout.
    return max(20, int((cols - 2) * scale))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render webcam video as live ASCII art in the terminal."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index (default: 0)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="Target refresh rate for terminal rendering (default: 20)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Width scale factor relative to terminal width, from 0.2 to 1.0 (default: 1.0)",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "any", "dshow", "msmf"],
        default="auto",
        help="Camera backend: auto, any, dshow, msmf (default: auto)",
    )
    return parser.parse_args()


def open_camera(camera_index: int, backend_choice: str) -> cv2.VideoCapture:
    if backend_choice == "auto":
        backend_names = ["dshow", "msmf", "any"]
    else:
        backend_names = [backend_choice]

    for name in backend_names:
        cap = cv2.VideoCapture(camera_index, BACKEND_MAP[name])
        if not cap.isOpened():
            cap.release()
            continue

        # Keep latency down when backend supports it.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Some cameras need warm-up frames before the first successful grab.
        for _ in range(30):
            ok, _ = cap.read()
            if ok:
                return cap
            time.sleep(0.03)

        cap.release()

    print(
        "Could not open webcam or grab frames. "
        "Close other camera apps and try --backend dshow or --backend msmf."
    )
    sys.exit(1)


def main() -> None:
    args = parse_args()
    if args.scale < 0.2 or args.scale > 1.0:
        raise ValueError("--scale must be between 0.2 and 1.0")

    cap = open_camera(args.camera, args.backend)

    frame_interval = 1.0 / max(1.0, args.fps)

    print("Starting ASCII webcam. Press Ctrl+C to stop.")
    print("\x1b[2J\x1b[H\x1b[?25l", end="")

    try:
        consecutive_failures = 0
        while True:
            start = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                consecutive_failures += 1
                if consecutive_failures > 60:
                    print("\nCamera stopped delivering frames. Exiting.")
                    break
                time.sleep(0.02)
                continue
            consecutive_failures = 0

            width = terminal_ascii_width(args.scale)
            art = frame_to_ascii(frame, width)

            # Move cursor to top-left and redraw the current frame.
            print("\x1b[H" + art, end="", flush=True)

            elapsed = time.perf_counter() - start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        print(RESET, end="")
        print("\x1b[?25h")
        print("Stopped.")


if __name__ == "__main__":
    main()
