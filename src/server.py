"""FastAPI web server for the waste-sorter dashboard.

Runs the live camera loop in a background thread and streams its output
to the browser.

Start:
    python src/server.py                      # webcam
    python src/server.py --video path/to.mp4  # recorded video
"""

from __future__ import annotations

import argparse
import collections
import io
import logging
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Deque

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common import get_classes, load_config, save_config  # noqa: E402
from infer import Classifier                # noqa: E402
from live import (                          # noqa: E402
    SortStateMachine,
    State,
    _SMConfig,
    capture_empty_ref,
    draw_osd,
    extract_features,
    get_roi,
    update_empty_ref,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared application state  (written by the camera thread, read by endpoints)
# ─────────────────────────────────────────────────────────────────────────────

class AppState:
    def __init__(self) -> None:
        self._lock = threading.Lock()

        # latest JPEG frame for /stream
        self.latest_jpeg: bytes = b""

        # live scalars
        self.state_name: str   = State.IDLE.name
        self.fps: float        = 0.0
        self.paused: bool      = False
        self.camera_error: str = ""

        # last decision event (shared interface dict)
        self.last_event: dict | None = None
        self.message: str = ""
        self.false_trigger_count: int = 0

        # running counts  {label: n}
        self.counts: dict[str, int] = {}
        self.uncertain_count: int   = 0

        # rolling event log (newest first)
        self.log: Deque[dict] = collections.deque(maxlen=200)

        self._log_seq: int = 0          # monotonic counter used as thumb key

        # live-adjustable thresholds (written by POST /api/settings)
        self.confidence_threshold: float = 0.70

    # All mutations go through here so readers always see a consistent snapshot
    def update_frame(
        self,
        jpeg: bytes,
        state_name: str,
        fps: float,
        paused: bool,
        last_event: dict | None,
        message: str,
    ) -> None:
        with self._lock:
            self.latest_jpeg = jpeg
            self.state_name  = state_name
            self.fps         = fps
            self.paused     = paused
            self.message = message
            if last_event is not None:
                self.last_event = last_event

    def record_event(self, event: dict) -> None:
        with self._lock:
            label = event["label"]
            self.counts[label] = self.counts.get(label, 0) + 1
            if event["uncertain"]:
                self.uncertain_count += 1
            # Use a monotonic seq number so thumb URLs stay stable after the
            # deque wraps at maxlen=200.
            seq = self._log_seq
            self._log_seq += 1
            entry = {k: v for k, v in event.items() if k != "thumb"}
            entry["thumb_url"]  = f"/api/thumb/{seq}"
            entry["thumb_data"] = event.get("thumb", "")
            entry["_seq"]       = seq
            self.log.appendleft(entry)

    def record_false_trigger(self) -> None:
        """Increment the false-trigger count without adding a sort event."""
        with self._lock:
            self.false_trigger_count += 1

    def set_paused(self, paused: bool) -> None:
        with self._lock:
            self.paused = paused

    def clear_camera_error(self) -> None:
        """Clear a recorded camera or hardware error for operator restart."""
        with self._lock:
            self.camera_error = ""

    def set_camera_error(self, error: str) -> None:
        """Store a camera-thread error for the dashboard."""
        with self._lock:
            self.camera_error = error

    def get_snapshot(self) -> dict:
        with self._lock:
            return {
                "state":      self.state_name,
                "fps":        round(self.fps, 1),
                "paused":     self.paused,
                "confidence_threshold": round(self.confidence_threshold, 2),
                "last_event": {k: v for k, v in self.last_event.items()
                               if k not in ("thumb", "thumb_data")}
                              if self.last_event else None,
                "camera_error": self.camera_error,
                "message": self.message,
            }


_app_state = AppState()
_machine_lock = threading.Lock()
_active_machine: SortStateMachine | None = None
_logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Camera / sort loop  (runs in a daemon thread)
# ─────────────────────────────────────────────────────────────────────────────

def _camera_thread(video_source: str | int, cfg: dict) -> None:
    """Background thread: read frames, run state machine, push to _app_state."""
    try:
        _run_loop(video_source, cfg)
    except Exception as exc:  # noqa: BLE001
        _app_state.set_camera_error(str(exc))
        global _active_machine
        with _machine_lock:
            _active_machine = None


def _run_loop(video_source: str | int, cfg: dict) -> None:
    global _active_machine
    roi_cfg = cfg.get("roi")

    onnx_path   = ROOT / cfg["paths"]["onnx"]
    labels_path = ROOT / cfg["paths"]["labels"]
    classifier  = Classifier(onnx_path, labels_path)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {video_source!r}")

    # For video files, pace playback to the file's own FPS (fallback 30).
    _src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    _frame_delay = (1.0 / _src_fps) if isinstance(video_source, str) else 0.0

    # ── empty reference ───────────────────────────────────────────────────────
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Could not read first frame from video source.")
    roi       = get_roi(frame, roi_cfg)
    empty_ref = capture_empty_ref(roi)
    prev_gray = empty_ref.copy()

    # ── state machine ─────────────────────────────────────────────────────────
    def _build_sm_cfg() -> _SMConfig:
        return _SMConfig(
            empty_diff_threshold  = float(cfg["empty_diff_threshold"]),
            stable_diff_threshold = 0.005,
            stable_frames         = int(cfg["stable_frames"]),
            vote_frames           = int(cfg["vote_frames"]),
            confidence_threshold  = _app_state.confidence_threshold,
            release_threshold     = float(cfg.get(
                "release_threshold",
                float(cfg["empty_diff_threshold"]) * 1.5,
            )),
            reference_alpha       = float(cfg.get("reference_alpha", 0.02)),
            clear_timeout_s       = float(cfg.get("clear_timeout_s", 20.0)),
            empty_label           = get_classes()[-1],
            fallback_label        = get_classes()[-2],
        )

    def _classify(roi_crop: np.ndarray) -> tuple[str, float]:
        label, conf, _ = classifier.predict(roi_crop)
        return label, conf

    def _on_decision(event: dict) -> None:
        _app_state.record_event(event)

    def _on_false_trigger() -> None:
        _app_state.record_false_trigger()
        _logger.info("False trigger: classifier voted for the empty class.")

    sm = SortStateMachine(
        cfg=_build_sm_cfg(),
        classify_fn=_classify,
        on_decision=_on_decision,
        on_false_trigger=_on_false_trigger,
    )
    with _machine_lock:
        _active_machine = sm

    fps_acc   = 0.0
    fps_alpha = 0.05
    t_prev    = time.perf_counter()

    while True:
        # Allow the confidence threshold to be updated live
        sm.cfg.confidence_threshold = _app_state.confidence_threshold

        if _app_state.paused:
            time.sleep(0.05)
            # Keep serving the last frame while paused
            continue

        ok, frame = cap.read()
        if not ok:
            # Video file ended – loop it, or stop on camera disconnect
            if isinstance(video_source, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
                if not ok:
                    break
                sm.reset()
                roi = get_roi(frame, roi_cfg)
                empty_ref = capture_empty_ref(roi)
                prev_gray = empty_ref.copy()
            else:
                break  # real camera gone

        # ── FPS ──────────────────────────────────────────────────────────────
        t_now    = time.perf_counter()
        inst_fps = 1.0 / max(t_now - t_prev, 1e-9)
        fps_acc  = inst_fps if fps_acc == 0.0 else (
            fps_alpha * inst_fps + (1 - fps_alpha) * fps_acc
        )
        t_prev = t_now

        # ── features + state machine ──────────────────────────────────────────
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
        sm.tick(features, roi)
        if sm.consume_reference_recapture():
            empty_ref = prev_gray.copy()

        # ── OSD + JPEG encode ─────────────────────────────────────────────────
        display = frame.copy()
        draw_osd(
            display,
            sm.state,
            sm.last_event,
            fps_acc,
            _app_state.paused,
            roi_cfg,
            sm.false_trigger_count,
            sm.message,
        )
        _, buf = cv2.imencode(
            ".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 70]
        )
        jpeg = buf.tobytes()

        _app_state.update_frame(
            jpeg       = jpeg,
            state_name = sm.state.name,
            fps        = fps_acc,
            paused     = _app_state.paused,
            last_event = sm.last_event,
            message    = sm.message,
        )

        # Pace video-file playback to the source FPS so the loop doesn't run at
        # CPU speed (1000+ fps) and fire thousands of state-machine ticks/second.
        if _frame_delay:
            time.sleep(_frame_delay)

    cap.release()
    with _machine_lock:
        if _active_machine is sm:
            _active_machine = None


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Waste Sorter")

_WEB_DIR = ROOT / "web"


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    html_path = _WEB_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="web/index.html not found")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


# ── Motion-JPEG stream ────────────────────────────────────────────────────────

def _mjpeg_generator():
    while True:
        jpeg = _app_state.latest_jpeg
        if not jpeg:
            # send a 1×1 grey placeholder while the camera warms up
            placeholder = np.full((1, 1, 3), 128, dtype=np.uint8)
            _, buf = cv2.imencode(".jpg", placeholder)
            jpeg = buf.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpeg +
            b"\r\n"
        )
        time.sleep(1 / 30)   # cap at ~30 fps to the browser


@app.get("/stream")
async def stream() -> StreamingResponse:
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── Status ────────────────────────────────────────────────────────────────────

@app.get("/api/status")
async def api_status() -> dict:
    """Return state-machine status and current operator guidance."""
    return _app_state.get_snapshot()


# ── Stats ─────────────────────────────────────────────────────────────────────

@app.get("/api/stats")
async def api_stats() -> dict:
    with _app_state._lock:
        return {
            "counts":          dict(_app_state.counts),
            "uncertain_count": _app_state.uncertain_count,
            "total":           sum(_app_state.counts.values()),
            "false_trigger_count": _app_state.false_trigger_count,
        }


# ── Event log ─────────────────────────────────────────────────────────────────

@app.get("/api/log")
async def api_log(n: int = 50) -> dict:
    with _app_state._lock:
        entries = list(_app_state.log)[:n]
    # Return entries without the raw thumb_data blob (client fetches /api/thumb/:id)
    clean = [{k: v for k, v in e.items() if k != "thumb_data"} for e in entries]
    return {"events": clean}


@app.get("/api/thumb/{idx}")
async def api_thumb(idx: int) -> Response:
    """Return the JPEG thumbnail for log entry with sequence number *idx*."""
    with _app_state._lock:
        entries = list(_app_state.log)
    # Find by monotonic sequence number so URLs stay valid after deque wraps
    match = next((e for e in entries if e.get("_seq") == idx), None)
    if match is None:
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    data_uri: str = match.get("thumb_data", "")
    if not data_uri.startswith("data:image/jpeg;base64,"):
        raise HTTPException(status_code=404, detail="No thumbnail data")
    import base64
    raw = base64.b64decode(data_uri.split(",", 1)[1])
    return Response(content=raw, media_type="image/jpeg")


# ── Settings ──────────────────────────────────────────────────────────────────

class SettingsBody(BaseModel):
    confidence_threshold: float | None = Field(
        default=None, ge=0.30, le=0.99,
        description="Confidence threshold (0.30–0.99)"
    )
    paused: bool | None = None


@app.post("/api/settings")
async def api_settings(body: SettingsBody) -> dict:
    """Apply settings immediately and persist to config.yaml."""
    changed: dict = {}

    if body.confidence_threshold is not None:
        _app_state.confidence_threshold = body.confidence_threshold
        changed["confidence_threshold"] = body.confidence_threshold

    if body.paused is not None:
        _app_state.set_paused(body.paused)
        changed["paused"] = body.paused

    # Persist to config.yaml
    if changed:
        cfg = load_config()
        if "confidence_threshold" in changed:
            cfg["confidence_threshold"] = changed["confidence_threshold"]
        save_config(cfg)

    return {"ok": True, "applied": changed}


@app.post("/api/continue")
async def api_continue() -> dict:
    """Request current-frame reference recapture and resume in IDLE."""
    # TODO: add the PIN check for later phone-control work.
    with _machine_lock:
        machine = _active_machine
    if machine is None:
        raise HTTPException(status_code=503, detail="Sorter camera is not running.")
    machine.request_continue()
    _app_state.set_paused(False)
    return {"ok": True, "message": "Continue requested"}


@app.post("/api/restart")
async def api_restart() -> dict:
    """Request state-machine restart and clear camera/pause errors."""
    # TODO: add the PIN check for later phone-control work.
    with _machine_lock:
        machine = _active_machine
    if machine is None:
        raise HTTPException(status_code=503, detail="Sorter camera is not running.")
    machine.request_restart()
    _app_state.set_paused(False)
    _app_state.clear_camera_error()
    return {"ok": True, "message": "Restart requested"}


# ─────────────────────────────────────────────────────────────────────────────
# LAN URL helper
# ─────────────────────────────────────────────────────────────────────────────

def _lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


# ─────────────────────────────────────────────────────────────────────────────
# CLI + startup
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--video", metavar="PATH",
                        help="Video file path (omit for webcam)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    args = _parse_args(argv)
    cfg  = load_config()

    # Seed app state from config
    _app_state.confidence_threshold = float(cfg.get("confidence_threshold", 0.70))

    video_source: str | int = args.video if args.video else int(cfg.get("camera_index", 0))

    # Start camera thread as daemon so it dies when the server exits
    t = threading.Thread(
        target=_camera_thread,
        args=(video_source, cfg),
        daemon=True,
        name="camera-loop",
    )
    t.start()

    host = cfg.get("server", {}).get("host", "127.0.0.1")
    port = int(cfg.get("server", {}).get("port", 8000))
    lan  = _lan_ip()

    print(f"\n  Waste Sorter dashboard")
    print(f"  Local : http://127.0.0.1:{port}")
    print(f"  LAN   : http://{lan}:{port}\n")

    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
