"""Live camera loop with a testable SortStateMachine.

Run:
    python src/live.py                      # webcam
    python src/live.py --video path/to.mp4  # recorded video

Keys (OpenCV window):
    r  – re-capture empty-tray reference
    c  – Continue after removing the item
    x  – Restart the sorter
    p  – pause / resume
    q  – quit
"""

from __future__ import annotations

import argparse
import base64
import logging
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common import get_classes, load_config  # noqa: E402
from infer import Classifier   # noqa: E402

_LOGGER = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# State machine – NO camera code inside
# ─────────────────────────────────────────────────────────────────────────────

class State(Enum):
    IDLE       = auto()
    PRESENT    = auto()
    STABLE     = auto()
    CLASSIFY   = auto()
    DECIDE     = auto()
    WAIT_CLEAR = auto()


@dataclass
class FrameFeatures:
    """Pre-computed per-frame values passed to the state machine each tick."""
    diff_to_empty: float   # mean-abs-diff of ROI vs empty-tray reference
    diff_to_prev:  float   # mean-abs-diff of ROI vs previous frame


@dataclass
class _SMConfig:
    empty_diff_threshold: float
    stable_diff_threshold: float   # diff_to_prev must stay below this
    stable_frames: int
    vote_frames: int
    confidence_threshold: float
    release_threshold: float | None = None
    reference_alpha: float = 0.02
    clear_timeout_s: float = 20.0
    empty_confidence_threshold: float = 0.80
    empty_label: str = ""
    fallback_label: str = ""


@dataclass
class SortStateMachine:
    """Pure state machine – all I/O is handled by the caller.

    Parameters
    ----------
    cfg:
        Runtime thresholds (built from config.yaml by the caller).
    classify_fn:
        Callable that receives the *current* ROI BGR crop and returns
        ``(label, confidence)``.  Called once per vote frame.
    on_decision:
        Callback invoked with a completed decision event dict when the machine
        reaches DECIDE.  May be ``None``.
    """

    cfg:           _SMConfig
    classify_fn:   Callable[[np.ndarray], tuple[str, float]]
    on_decision:   Callable[[dict], None] | None = None
    on_false_trigger: Callable[[], None] | None = None

    # ── mutable state ────────────────────────────────────────────────────────
    state:          State = field(default=State.IDLE, init=False)
    _consec:        int   = field(default=0, init=False)   # consecutive stable/clear frames
    _votes:         list[tuple[str, float]] = field(default_factory=list, init=False)
    last_event:     dict | None = field(default=None, init=False)
    false_trigger_count: int = field(default=0, init=False)
    _clear_started_at: float | None = field(default=None, init=False)
    _wait_clear_frames: int = field(default=0, init=False)
    _empty_votes: int = field(default=0, init=False)
    _clear_timed_out: bool = field(default=False, init=False)
    _continue_requested: threading.Event = field(
        default_factory=threading.Event, init=False, repr=False
    )
    _restart_requested: threading.Event = field(
        default_factory=threading.Event, init=False, repr=False
    )
    _recapture_reference: bool = field(default=False, init=False)

    @property
    def message(self) -> str:
        """Return the state guidance currently suitable for display."""
        if self.state == State.WAIT_CLEAR:
            if self._clear_timed_out:
                return "Remove the item, then press Continue"
            return "Waiting for the tray to clear"
        return ""

    # ── public API ───────────────────────────────────────────────────────────

    def tick(
        self,
        features: FrameFeatures,
        roi_crop: np.ndarray,
        now: float | None = None,
    ) -> dict | None:
        """Advance the state machine by one frame.

        Parameters
        ----------
        features:
            Pre-computed diff values for this frame.
        roi_crop:
            BGR crop of the region of interest (used in CLASSIFY).

        Returns
        -------
        A decision event dict if a decision was just made, else ``None``.
        """
        current_time = time.monotonic() if now is None else now
        if self._restart_requested.is_set():
            self._restart_requested.clear()
            self._continue_requested.clear()
            self.reset()
            self._recapture_reference = True
            return None
        if self._continue_requested.is_set():
            self._continue_requested.clear()
            self.reset()
            self._recapture_reference = True
            return None

        event: dict | None = None

        if self.state == State.IDLE:
            if features.diff_to_empty > self.cfg.empty_diff_threshold:
                self.state = State.PRESENT
                self._consec = 0

        elif self.state == State.PRESENT:
            if features.diff_to_empty <= self.cfg.empty_diff_threshold:
                # object gone before it settled
                self.state = State.IDLE
            elif features.diff_to_prev < self.cfg.stable_diff_threshold:
                self._consec += 1
                if self._consec >= self.cfg.stable_frames:
                    self.state = State.STABLE
                    self._consec = 0
            else:
                self._consec = 0

        elif self.state == State.STABLE:
            self._votes = []
            self.state = State.CLASSIFY

        elif self.state == State.CLASSIFY:
            label, confidence = self.classify_fn(roi_crop)
            self._votes.append((label, confidence))
            if len(self._votes) >= self.cfg.vote_frames:
                self.state = State.DECIDE

        elif self.state == State.DECIDE:
            majority_label = Counter(label for label, _ in self._votes).most_common(1)[0][0]
            if self.cfg.empty_label and majority_label == self.cfg.empty_label:
                self.false_trigger_count += 1
                _LOGGER.info(
                    "False trigger: classifier voted for empty (count=%d).",
                    self.false_trigger_count,
                )
                if self.on_false_trigger is not None:
                    self.on_false_trigger()
                self.state = State.IDLE
                self._consec = 0
                self._votes = []
            else:
                event = self._make_decision(roi_crop)
                if self.on_decision is not None:
                    self.on_decision(event)
                self.last_event = event
                self.state = State.WAIT_CLEAR
                self._consec = 0
                self._clear_started_at = current_time
                self._wait_clear_frames = 0
                self._empty_votes = 0
                self._clear_timed_out = False

        elif self.state == State.WAIT_CLEAR:
            release_threshold = (
                self.cfg.release_threshold
                if self.cfg.release_threshold is not None
                else self.cfg.empty_diff_threshold * 1.5
            )
            if features.diff_to_empty <= release_threshold:
                self._consec += 1
                if self._consec >= self.cfg.stable_frames:
                    self._release_clear()
            else:
                self._consec = 0
            self._wait_clear_frames += 1
            if self.state == State.WAIT_CLEAR and self._wait_clear_frames % 5 == 0:
                label, confidence = self.classify_fn(roi_crop)
                if (
                    label == self.cfg.empty_label
                    and confidence >= self.cfg.empty_confidence_threshold
                ):
                    self._empty_votes += 1
                    if self._empty_votes >= self.cfg.vote_frames:
                        self._release_clear()
                else:
                    self._empty_votes = 0
            if (
                self.state == State.WAIT_CLEAR
                and self._clear_started_at is not None
                and current_time - self._clear_started_at > self.cfg.clear_timeout_s
            ):
                self._clear_timed_out = True

        return event

    def reset(self) -> None:
        """Return to IDLE and clear all transient state."""
        self.state   = State.IDLE
        self._consec = 0
        self._votes  = []
        self._clear_started_at = None
        self._wait_clear_frames = 0
        self._empty_votes = 0
        self._clear_timed_out = False

    def request_continue(self) -> None:
        """Thread-safely request reference recapture and transition to IDLE."""
        self._continue_requested.set()

    def request_restart(self) -> None:
        """Thread-safely request a full state-machine restart and recapture."""
        self._restart_requested.set()

    def consume_reference_recapture(self) -> bool:
        """Return and clear the pending current-frame reference recapture flag."""
        requested = self._recapture_reference
        self._recapture_reference = False
        return requested

    def _release_clear(self) -> None:
        """Finish WAIT_CLEAR and clear its transient timers and counters."""
        self.state = State.IDLE
        self._consec = 0
        self._clear_started_at = None
        self._wait_clear_frames = 0
        self._empty_votes = 0
        self._clear_timed_out = False

    # ── private helpers ───────────────────────────────────────────────────────

    def _make_decision(self, roi_crop: np.ndarray) -> dict:
        labels      = [v[0] for v in self._votes]
        confidences = [v[1] for v in self._votes]

        majority_label = Counter(labels).most_common(1)[0][0]
        # Mean confidence over votes that agree with the majority
        agreeing = [c for lbl, c in zip(labels, confidences) if lbl == majority_label]
        mean_conf = float(np.mean(agreeing))

        uncertain = mean_conf < self.cfg.confidence_threshold
        final_label = (
            self.cfg.fallback_label if uncertain and self.cfg.fallback_label
            else majority_label
        )

        thumb = _encode_thumb(roi_crop)

        return {
            "ts":         time.time(),
            "label":      final_label,
            "confidence": mean_conf,
            "uncertain":  uncertain,
            "thumb":      thumb,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(
    roi: np.ndarray,
    empty_ref: np.ndarray,
    prev_gray: np.ndarray | None,
) -> tuple[FrameFeatures, np.ndarray]:
    """Compute diff_to_empty and diff_to_prev from a BGR ROI crop.

    Returns the features and the current blurred-grayscale image (use it
    as ``prev_gray`` on the next call).
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    diff_to_empty = float(np.mean(np.abs(gray.astype(np.float32) - empty_ref.astype(np.float32)))) / 255.0

    if prev_gray is not None and prev_gray.shape == gray.shape:
        diff_to_prev = float(np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32)))) / 255.0
    else:
        diff_to_prev = 0.0

    return FrameFeatures(diff_to_empty=diff_to_empty, diff_to_prev=diff_to_prev), gray


def capture_empty_ref(roi: np.ndarray) -> np.ndarray:
    """Compute the blurred-grayscale reference from an empty-tray ROI."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (7, 7), 0)


def update_empty_ref(
    empty_ref: np.ndarray,
    current_gray: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Return one exponential-moving-average update for the empty reference."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("reference alpha must be between 0 and 1.")
    if empty_ref.shape != current_gray.shape:
        return current_gray.copy()
    updated = (
        (1.0 - alpha) * empty_ref.astype(np.float32)
        + alpha * current_gray.astype(np.float32)
    )
    return np.clip(np.rint(updated), 0, 255).astype(np.uint8)


def get_roi(frame: np.ndarray, roi_cfg: list | None) -> np.ndarray:
    """Crop the frame to the configured ROI, or return the full frame."""
    if roi_cfg is None:
        return frame
    x, y, w, h = roi_cfg
    return frame[y : y + h, x : x + w]


# ─────────────────────────────────────────────────────────────────────────────
# Thumbnail helper
# ─────────────────────────────────────────────────────────────────────────────

def _encode_thumb(roi: np.ndarray, size: int = 128) -> str:
    """Resize ROI and return a data-URI string (JPEG, base64)."""
    thumb = cv2.resize(roi, (size, size), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not ok:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


# ─────────────────────────────────────────────────────────────────────────────
# On-screen OSD
# ─────────────────────────────────────────────────────────────────────────────

_PALETTE = {
    State.IDLE:       (180, 180, 180),
    State.PRESENT:    (0, 200, 255),
    State.STABLE:     (0, 255, 200),
    State.CLASSIFY:   (255, 200, 0),
    State.DECIDE:     (0, 255, 0),
    State.WAIT_CLEAR: (100, 100, 255),
}


def draw_osd(
    frame: np.ndarray,
    state: State,
    last_event: dict | None,
    fps: float,
    paused: bool,
    roi_cfg: list | None,
    false_trigger_count: int = 0,
    message: str = "",
) -> None:
    """Overlay state, label, confidence and FPS onto *frame* in-place."""
    h, w = frame.shape[:2]
    colour = _PALETTE.get(state, (200, 200, 200))

    # ROI rectangle
    if roi_cfg is not None:
        x, y, rw, rh = roi_cfg
        cv2.rectangle(frame, (x, y), (x + rw, y + rh), colour, 2)

    # State badge
    cv2.putText(frame, f"State: {state.name}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2, cv2.LINE_AA)

    # Last decision
    if last_event:
        tag = "?" if last_event["uncertain"] else ""
        txt = f"Last: {last_event['label']}{tag}  {last_event['confidence']:.0%}"
        cv2.putText(frame, txt, (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # FPS
    cv2.putText(frame, f"{fps:.1f} fps", (w - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.putText(
        frame,
        f"False triggers: {false_trigger_count}",
        (10, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    if message:
        cv2.putText(
            frame,
            message,
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )

    if paused:
        cv2.putText(frame, "PAUSED", (w // 2 - 60, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Main camera loop
# ─────────────────────────────────────────────────────────────────────────────

def run(
    video_source: str | int,
    classifier: Classifier,
    cfg: dict,
    on_decision: Callable[[dict], None] | None = None,
) -> None:
    """Open *video_source* and run the sort loop until the user quits."""
    roi_cfg    = cfg.get("roi")                           # [x,y,w,h] or None
    img_size   = int(cfg.get("image_size", 224))

    sm_cfg = _SMConfig(
        empty_diff_threshold  = float(cfg["empty_diff_threshold"]),
        stable_diff_threshold = 0.005,   # ~1.3/255; empirically quiet when still
        stable_frames         = int(cfg["stable_frames"]),
        vote_frames           = int(cfg["vote_frames"]),
        confidence_threshold  = float(cfg["confidence_threshold"]),
        release_threshold     = float(cfg.get(
            "release_threshold",
            float(cfg["empty_diff_threshold"]) * 1.5,
        )),
        reference_alpha       = float(cfg.get("reference_alpha", 0.02)),
        clear_timeout_s       = float(cfg.get("clear_timeout_s", 20.0)),
        empty_label           = str(cfg.get("classes", [""])[-1]),
        fallback_label        = get_classes()[-2],
    )

    def _classify(roi_crop: np.ndarray) -> tuple[str, float]:
        label, conf, _ = classifier.predict(roi_crop)
        return label, conf

    sm = SortStateMachine(cfg=sm_cfg, classify_fn=_classify, on_decision=on_decision)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {video_source!r}")

    paused     = False
    prev_gray: np.ndarray | None = None
    empty_ref: np.ndarray | None = None
    fps_acc    = 0.0
    fps_alpha  = 0.05          # EWM smoothing factor

    # ── capture empty reference from first readable frame ────────────────────
    while empty_ref is None:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Could not read any frame from video source.")
        roi = get_roi(frame, roi_cfg)
        empty_ref = capture_empty_ref(roi)
        prev_gray = empty_ref.copy()

    t_prev = time.perf_counter()

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("p"):
            paused = not paused
        if key == ord("c"):
            sm.request_continue()
            paused = False
        if key == ord("x"):
            sm.request_restart()
            paused = False
        if key == ord("r"):
            # Re-capture empty reference on current frame
            ok, frame = cap.read()
            if ok:
                roi = get_roi(frame, roi_cfg)
                empty_ref = capture_empty_ref(roi)
                prev_gray = empty_ref.copy()
                sm.reset()
            continue

        if paused:
            # Keep showing the last frame; don't advance the source
            if "frame" in dir():  # type: ignore[name-defined]
                draw_osd(
                    frame,
                    sm.state,
                    sm.last_event,
                    fps_acc,
                    paused,
                    roi_cfg,
                    sm.false_trigger_count,
                    sm.message,
                )  # type: ignore[name-defined]
                cv2.imshow("Waste Sorter", frame)  # type: ignore[name-defined]
            continue

        ok, frame = cap.read()
        if not ok:
            break   # end of video file or camera disconnect

        # ── FPS ──────────────────────────────────────────────────────────────
        t_now = time.perf_counter()
        inst_fps = 1.0 / max(t_now - t_prev, 1e-9)
        fps_acc  = inst_fps if fps_acc == 0.0 else fps_alpha * inst_fps + (1 - fps_alpha) * fps_acc
        t_prev   = t_now

        # ── features ─────────────────────────────────────────────────────────
        roi = get_roi(frame, roi_cfg)
        features, prev_gray = extract_features(roi, empty_ref, prev_gray)
        if (
            sm.state == State.IDLE
            and features.diff_to_prev < sm.cfg.stable_diff_threshold
        ):
            empty_ref = update_empty_ref(
                empty_ref,
                prev_gray,
                sm.cfg.reference_alpha,
            )

        # ── state machine tick ────────────────────────────────────────────────
        sm.tick(features, roi)
        if sm.consume_reference_recapture():
            empty_ref = prev_gray.copy()

        # ── OSD ──────────────────────────────────────────────────────────────
        draw_osd(
            frame,
            sm.state,
            sm.last_event,
            fps_acc,
            paused,
            roi_cfg,
            sm.false_trigger_count,
            sm.message,
        )
        cv2.imshow("Waste Sorter", frame)

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", metavar="PATH",
                        help="Path to a video file (omit to use the webcam)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    cfg  = load_config()

    onnx_path   = ROOT / cfg["paths"]["onnx"]
    labels_path = ROOT / cfg["paths"]["labels"]
    classifier  = Classifier(onnx_path, labels_path)

    if args.video:
        source: str | int = args.video
    else:
        source = int(cfg.get("camera_index", 0))

    def _print_event(event: dict) -> None:
        tag = " [uncertain]" if event["uncertain"] else ""
        print(
            f"[EVENT] ts={event['ts']:.3f}  label={event['label']}"
            f"  conf={event['confidence']:.3f}{tag}"
        )

    run(source, classifier, cfg, on_decision=_print_event)


if __name__ == "__main__":
    main()
