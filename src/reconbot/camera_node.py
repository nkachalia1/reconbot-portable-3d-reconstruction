"""Laptop webcam service for remote Raspberry Pi capture coordination."""

from __future__ import annotations

import atexit
import base64
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Protocol

import numpy as np

from ._optional import require_cv2
from .active_guidance import (
    Guidance,
    guidance_code_from_stats,
    guidance_text,
)
from .features import detect_and_match, geometric_inlier_ratio
from .frame_quality import measure_frame_quality
from .io_utils import list_images
from .system_telemetry import collect_system_telemetry


class FrameSource(Protocol):
    def read(self) -> np.ndarray:
        """Return the latest BGR frame."""

    def close(self) -> None:
        """Release resources."""


class OpenCVCameraSource:
    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
    ) -> None:
        cv2 = require_cv2()
        self._cv2 = cv2
        self._capture = cv2.VideoCapture(camera_index)
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self._capture.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}")
        self._lock = threading.RLock()
        self._latest: np.ndarray | None = None
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        while self._running:
            ok, frame = self._capture.read()
            if not ok:
                time.sleep(0.05)
                continue
            with self._lock:
                self._latest = frame

    def read(self) -> np.ndarray:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            with self._lock:
                if self._latest is not None:
                    return self._latest.copy()
            time.sleep(0.02)
        raise RuntimeError("Camera did not produce a frame within five seconds")

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self._capture.release()


class DirectoryFrameSource:
    """Deterministic camera substitute used for local integration tests."""

    def __init__(self, image_dir: str | Path) -> None:
        self._cv2 = require_cv2()
        self._images = list_images(image_dir)
        if not self._images:
            raise RuntimeError(f"No simulation images found in {image_dir}")
        self._index = 0
        self._lock = threading.Lock()

    def read(self) -> np.ndarray:
        with self._lock:
            path = self._images[self._index]
            self._index = min(self._index + 1, len(self._images) - 1)
        frame = self._cv2.imread(str(path), self._cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Could not read simulation image: {path}")
        return frame

    def close(self) -> None:
        return


@dataclass
class CameraNodeState:
    previous_gray: np.ndarray | None = None
    captures: int = 0
    last_capture_at: float | None = None
    started_at: float = time.time()


class VideoRecorder:
    def __init__(self, source: FrameSource, output_dir: str | Path) -> None:
        self.source = source
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._writer = None
        self._path: Path | None = None
        self._fps = 15.0
        self._frames = 0
        self._width = 0
        self._height = 0
        self._started_at: float | None = None
        self._stopped_at: float | None = None
        self._discover_latest_recording()

    @property
    def recording(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _discover_latest_recording(self) -> None:
        candidates = [
            path
            for path in self.output_dir.glob("*.mp4")
            if not path.name.endswith("_capture.mp4")
        ]
        if not candidates:
            return
        latest = max(candidates, key=lambda path: path.stat().st_mtime)
        self._path = latest
        self._stopped_at = latest.stat().st_mtime
        timestamp_match = re.search(r"_(\d{10})\.mp4$", latest.name)
        if timestamp_match:
            self._started_at = float(timestamp_match.group(1))
        try:
            cv2 = require_cv2()
            capture = cv2.VideoCapture(str(latest))
            try:
                self._frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                self._fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 15.0
                self._width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                self._height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            finally:
                capture.release()
        except (ImportError, RuntimeError):
            return
        duration = self._frames / self._fps if self._fps > 0 else 0.0
        if self._started_at is None:
            self._started_at = self._stopped_at - duration
        self._stopped_at = self._started_at + duration

    def latest(self) -> dict[str, object] | None:
        with self._lock:
            if (
                self.recording
                or self._path is None
                or self._path.name.endswith("_capture.mp4")
                or not self._path.exists()
            ):
                return None
            duration = (
                self._frames / self._fps
                if self._frames > 0 and self._fps > 0
                else (
                    0.0
                    if self._started_at is None or self._stopped_at is None
                    else self._stopped_at - self._started_at
                )
            )
            return {
                "ok": True,
                "filename": self._path.name,
                "started_at": self._started_at,
                "stopped_at": self._stopped_at,
                "duration_s": duration,
                "frames": self._frames,
                "fps": self._fps,
                "width": self._width,
                "height": self._height,
                "bytes": self._path.stat().st_size,
            }

    def status(self) -> dict[str, object]:
        with self._lock:
            started_at = self._started_at
            end_time = time.time() if self.recording else self._stopped_at
            status = {
                "recording": self.recording,
                "frames": self._frames,
                "fps": self._fps,
                "started_at": started_at,
                "duration_s": (
                    0.0
                    if started_at is None or end_time is None
                    else end_time - started_at
                ),
            }
            latest = self.latest()
            if latest is not None:
                status.update(latest)
            return status

    def start(self, session_id: str, fps: float = 15.0) -> dict[str, object]:
        cv2 = require_cv2()
        with self._lock:
            if self.recording:
                raise RuntimeError("A video recording is already in progress")
            identifier = re.sub(r"[^A-Za-z0-9_-]+", "_", session_id).strip("_")
            if not identifier:
                identifier = time.strftime("field_%Y%m%d_%H%M%S")
            frame = self.source.read()
            self._height, self._width = frame.shape[:2]
            self._fps = max(1.0, min(float(fps), 30.0))
            self._path = self.output_dir / f"{identifier}_{int(time.time())}_capture.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                str(self._path),
                fourcc,
                self._fps,
                (self._width, self._height),
            )
            if not self._writer.isOpened():
                self._writer.release()
                self._writer = None
                raise RuntimeError("Could not open the MP4 video writer")
            self._frames = 0
            self._started_at = time.time()
            self._stopped_at = None
            self._stop_event.clear()
            self._writer.write(frame)
            self._frames = 1
            self._thread = threading.Thread(target=self._record_loop, daemon=True)
            self._thread.start()
            return self.status()

    def _record_loop(self) -> None:
        interval = 1.0 / self._fps
        deadline = time.monotonic() + interval
        while not self._stop_event.wait(max(0.0, deadline - time.monotonic())):
            frame = self.source.read()
            if frame.shape[1] != self._width or frame.shape[0] != self._height:
                cv2 = require_cv2()
                frame = cv2.resize(frame, (self._width, self._height))
            with self._lock:
                if self._writer is None:
                    break
                self._writer.write(frame)
                self._frames += 1
            deadline += interval

    def stop(self) -> dict[str, object]:
        with self._lock:
            if not self.recording:
                raise RuntimeError("No video recording is in progress")
            thread = self._thread
            self._stop_event.set()
        if thread is not None:
            thread.join(timeout=10.0)
        with self._lock:
            if self._writer is not None:
                self._writer.release()
            self._writer = None
            self._thread = None
            self._stopped_at = time.time()
            if self._path is None or not self._path.exists():
                raise RuntimeError("The video recording was not written")
            raw_path = self._path
            final_path = raw_path.with_name(raw_path.name.replace("_capture.mp4", ".mp4"))
            try:
                import imageio_ffmpeg
            except ImportError as exc:
                raise RuntimeError(
                    f"Recording saved at {raw_path}, but imageio-ffmpeg is required "
                    "to create a browser-compatible H.264 MP4"
                ) from exc
            completed = subprocess.run(
                [
                    imageio_ffmpeg.get_ffmpeg_exe(),
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(raw_path),
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "20",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(final_path),
                ],
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0 or not final_path.exists():
                raise RuntimeError(
                    f"Recording saved at {raw_path}, but H.264 conversion failed: "
                    f"{completed.stderr.strip()}"
                )
            raw_path.unlink()
            self._path = final_path
            latest = self.latest()
            if latest is None:
                raise RuntimeError("The completed video could not be indexed")
            return latest

    def path_for(self, filename: str) -> Path:
        candidate = (self.output_dir / Path(filename).name).resolve()
        if candidate.parent != self.output_dir or not candidate.exists():
            raise FileNotFoundError(filename)
        return candidate

    def close(self) -> None:
        if self.recording:
            try:
                self.stop()
            except RuntimeError:
                pass


def _analyze_guidance(
    previous_gray: np.ndarray | None,
    current_frame: np.ndarray,
    blur_score: float,
) -> Guidance:
    cv2 = require_cv2()
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    if previous_gray is None:
        return Guidance(
            match_count=0,
            geometric_inlier_ratio=0.0,
            recommendation="Reference frame captured. Begin a slow lateral arc.",
            code="reference_frame",
            median_motion_px=0.0,
        )
    result = detect_and_match(
        previous_gray,
        current_gray,
        method="ORB",
        max_features=2500,
        ratio=0.78,
    )
    inlier_ratio = geometric_inlier_ratio(result.matched_points_a, result.matched_points_b)
    if result.match_count:
        motion = result.matched_points_b - result.matched_points_a
        median_motion = float(np.median(np.linalg.norm(motion, axis=1)))
    else:
        median_motion = 0.0
    code = guidance_code_from_stats(
        result.match_count,
        inlier_ratio,
        median_motion,
        blur_score,
    )
    return Guidance(
        match_count=result.match_count,
        geometric_inlier_ratio=inlier_ratio,
        recommendation=guidance_text(code),
        code=code,
        median_motion_px=median_motion,
    )


def create_camera_app(
    source: FrameSource,
    token: str | None = None,
    recording_dir: str | Path = "data/camera_recordings",
):
    try:
        from flask import Flask, Response, jsonify, request, send_file
    except ImportError as exc:
        raise RuntimeError("Flask is required: install `python -m pip install -e .[dashboard]`.") from exc

    cv2 = require_cv2()
    app = Flask(__name__)
    state = CameraNodeState()
    recorder = VideoRecorder(source, recording_dir)
    app.extensions["reconbot_recorder"] = recorder
    atexit.register(recorder.close)
    lock = threading.Lock()

    def authorized() -> bool:
        return not token or request.headers.get("X-ReconBot-Token") == token

    @app.before_request
    def verify_token():
        if request.path == "/api/health":
            return None
        if not authorized():
            return jsonify({"error": "unauthorized"}), 401
        return None

    @app.get("/api/health")
    def health():
        return jsonify(
            {
                "ok": True,
                "role": "camera-node",
                "captures": state.captures,
                "uptime_s": time.time() - state.started_at,
                "video": recorder.status(),
            }
        )

    @app.get("/api/preview.jpg")
    def preview():
        frame = source.read()
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return jsonify({"error": "JPEG encoding failed"}), 500
        return Response(encoded.tobytes(), mimetype="image/jpeg")

    @app.post("/api/session/reset")
    def reset_session():
        if recorder.recording:
            return jsonify({"error": "Stop the active video recording first"}), 409
        with lock:
            state.previous_gray = None
            state.captures = 0
            state.last_capture_at = None
        return jsonify({"ok": True})

    @app.post("/api/video/start")
    def start_video():
        body = request.get_json(silent=True) or {}
        try:
            result = recorder.start(
                str(body.get("session_id") or ""),
                float(body.get("fps") or 15.0),
            )
        except (RuntimeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify({"ok": True, **result})

    @app.post("/api/video/stop")
    def stop_video():
        try:
            result = recorder.stop()
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify(result)

    @app.get("/api/video/latest")
    def latest_video():
        result = recorder.latest()
        if result is None:
            return jsonify({"error": "No completed video is available"}), 404
        return jsonify(result)

    @app.get("/api/video/<path:filename>")
    def video_file(filename: str):
        try:
            path = recorder.path_for(filename)
        except FileNotFoundError:
            return jsonify({"error": "Video not found"}), 404
        return send_file(path, mimetype="video/mp4", as_attachment=False)

    @app.post("/api/capture")
    def capture():
        started = time.perf_counter()
        frame = source.read()
        quality = measure_frame_quality(frame)
        accepted = (
            quality.blur_score >= 50.0
            and quality.dark_fraction < 0.50
            and quality.bright_fraction < 0.50
        )
        with lock:
            guidance = _analyze_guidance(state.previous_gray, frame, quality.blur_score)
            if accepted:
                state.previous_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            state.captures += 1
            state.last_capture_at = time.time()
            sequence = state.captures
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            return jsonify({"error": "JPEG encoding failed"}), 500
        return jsonify(
            {
                "ok": True,
                "sequence": sequence,
                "captured_at": state.last_capture_at,
                "image_base64": base64.b64encode(encoded.tobytes()).decode("ascii"),
                "width": int(frame.shape[1]),
                "height": int(frame.shape[0]),
                "accepted": accepted,
                "quality": quality.to_dict(),
                "guidance": guidance.to_dict(),
                "latency_ms": (time.perf_counter() - started) * 1000.0,
                "telemetry": collect_system_telemetry(),
            }
        )

    return app
