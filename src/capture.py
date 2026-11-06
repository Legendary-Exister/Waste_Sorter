"""Capture own-camera empty-tray examples from a webcam or recorded video."""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common import load_config  # noqa: E402


CHECKLIST = (
    "Try lighting changes:",
    "  [ ] lights on and off",
    "  [ ] pass a hand shadow over the tray",
    "  [ ] nudge the camera slightly",
)


def _roi(frame, roi_cfg):
    if roi_cfg is None:
        return frame
    x, y, width, height = (int(value) for value in roi_cfg)
    left, top = max(0, x), max(0, y)
    right = min(frame.shape[1], x + width)
    bottom = min(frame.shape[0], y + height)
    if left >= right or top >= bottom:
        raise ValueError(
            f"configured ROI {roi_cfg!r} does not overlap frame dimensions "
            f"{frame.shape[1]}x{frame.shape[0]}"
        )
    return frame[top:bottom, left:right]


def _draw_overlay(frame, remaining: float | None):
    lines = [f"Countdown: {remaining:.1f}s" if remaining is not None else "Extracting video"]
    lines.extend(CHECKLIST)
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (12, 28 + index * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (40, 240, 40),
            2,
            cv2.LINE_AA,
        )
    return frame


def _can_show_window() -> bool:
    return os.name == "nt" or bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def capture_empty(seconds: float | None = None, video_path: Path | None = None) -> list[Path]:
    """Save every fifth ROI frame as an own empty-tray image."""
    cfg = load_config()
    camera_source = str(video_path) if video_path is not None else int(cfg.get("camera_index", 0))
    capture = cv2.VideoCapture(camera_source)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video source: {camera_source!r}")

    output_dir = ROOT / str(cfg["paths"]["own_root"]) / "empty"
    output_dir.mkdir(parents=True, exist_ok=True)
    session = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination_pattern = f"empty_{session}_*.jpg"
    while any(output_dir.glob(destination_pattern)):
        session = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        destination_pattern = f"empty_{session}_*.jpg"

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    show = _can_show_window()
    if not show:
        print("Lighting checklist: lights on/off; pass a hand shadow; nudge the camera slightly.")
    if video_path is not None and fps > 0 and total_frames > 0:
        duration = total_frames / fps
    else:
        duration = seconds

    saved: list[Path] = []
    started = time.monotonic()
    frame_index = 0
    try:
        while True:
            elapsed = time.monotonic() - started
            if seconds is not None and elapsed >= seconds:
                break
            ok, frame = capture.read()
            if not ok:
                if video_path is not None:
                    break
                if seconds is not None and time.monotonic() - started >= seconds:
                    break
                raise RuntimeError("Could not read a frame from the camera.")

            frame_index += 1
            roi = _roi(frame, cfg.get("roi"))
            if frame_index % 5 == 0:
                filename = output_dir / f"empty_{session}_{len(saved) + 1:04d}.jpg"
                if not cv2.imwrite(str(filename), roi):
                    raise OSError(f"Could not save captured frame to {filename}")
                saved.append(filename)
                print(f"Saved {filename.relative_to(ROOT)}")

            remaining = None
            if seconds is not None:
                remaining = max(0.0, seconds - (time.monotonic() - started))
            elif duration is not None:
                remaining = max(0.0, duration - frame_index / fps) if fps > 0 else None
            if show:
                display = _draw_overlay(frame.copy(), remaining)
                cv2.imshow("Empty-tray capture", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            elif seconds is not None:
                print(f"\rRecording empty tray: {remaining:.1f}s remaining", end="", flush=True)
    finally:
        capture.release()
        if show:
            cv2.destroyAllWindows()
    if seconds is not None and not show:
        print()
    print(f"Saved {len(saved)} empty-tray frame(s) to {output_dir.relative_to(ROOT)}")
    return saved


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--record-empty",
        type=float,
        metavar="SECONDS",
        help="record the configured live-camera ROI for this many seconds",
    )
    mode.add_argument(
        "--extract-empty",
        type=Path,
        metavar="VIDEO",
        help="extract every fifth ROI frame from a video file",
    )
    args = parser.parse_args(argv)
    if args.record_empty is not None and args.record_empty <= 0:
        parser.error("--record-empty must be greater than zero")
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    try:
        if args.record_empty is not None:
            capture_empty(seconds=args.record_empty)
        else:
            capture_empty(video_path=args.extract_empty)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"capture failed: {exc}") from exc


if __name__ == "__main__":
    main()