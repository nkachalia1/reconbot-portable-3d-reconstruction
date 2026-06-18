"""Raspberry Pi coordinator for remote capture, storage, and live dashboard APIs."""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import http.client
import json
from pathlib import Path
import re
import shutil
import threading
import time
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

from .active_guidance import guidance_text
from .io_utils import ensure_dir, write_json
from .system_telemetry import collect_system_telemetry


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_session_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_")
    if not normalized:
        normalized = datetime.now().strftime("session_%Y%m%d_%H%M%S")
    return normalized[:80]


class CameraClient(Protocol):
    def health(self) -> dict[str, object]:
        """Return camera node health."""

    def reset(self) -> dict[str, object]:
        """Reset camera guidance history."""

    def capture(self) -> dict[str, object]:
        """Capture one image and return its metadata."""

    def preview(self) -> bytes:
        """Return the latest camera JPEG."""

    def start_video(self, session_id: str, fps: float) -> dict[str, object]:
        """Begin recording on the camera node."""

    def stop_video(self) -> dict[str, object]:
        """Stop recording and return video metadata."""

    def download_video(self, filename: str, target: Path) -> None:
        """Download a completed recording to target."""


class ReconstructionClient(Protocol):
    def health(self) -> dict[str, object]:
        """Return reconstruction worker health."""

    def list_reconstructions(self) -> dict[str, object]:
        """Return the reconstruction library."""

    def submit(
        self,
        session_id: str,
        video_path: Path,
        parameters: dict[str, object],
    ) -> dict[str, object]:
        """Submit a field video for reconstruction."""

    def job(self, job_id: str) -> dict[str, object]:
        """Return one reconstruction job."""

    def activate(self, identifier: str) -> dict[str, object]:
        """Make a reconstruction active."""

    def delete(self, identifier: str) -> dict[str, object]:
        """Delete a reconstruction and its saved video."""

    def asset(
        self,
        identifier: str,
        filename: str,
        range_header: str | None = None,
    ) -> tuple[int, bytes, dict[str, str]]:
        """Download or range-read a reconstruction asset."""


class HttpCameraClient:
    def __init__(self, base_url: str, token: str | None = None, timeout_s: float = 15.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_s = timeout_s

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.token:
            headers["X-ReconBot-Token"] = self.token
        return headers

    def _request(
        self,
        path: str,
        method: str = "GET",
        payload: dict[str, object] | None = None,
    ) -> dict[str, object]:
        headers = self._headers()
        data = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload).encode("utf-8")
        request = Request(self.base_url + path, method=method, headers=headers, data=data)
        with urlopen(request, timeout=self.timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))

    def health(self) -> dict[str, object]:
        return self._request("/api/health")

    def reset(self) -> dict[str, object]:
        return self._request("/api/session/reset", "POST")

    def capture(self) -> dict[str, object]:
        return self._request("/api/capture", "POST")

    def preview(self) -> bytes:
        request = Request(self.base_url + "/api/preview.jpg", headers=self._headers())
        with urlopen(request, timeout=self.timeout_s) as response:
            return response.read()

    def start_video(self, session_id: str, fps: float) -> dict[str, object]:
        return self._request(
            "/api/video/start",
            "POST",
            {"session_id": session_id, "fps": fps},
        )

    def stop_video(self) -> dict[str, object]:
        return self._request("/api/video/stop", "POST")

    def download_video(self, filename: str, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        request = Request(
            self.base_url + "/api/video/" + quote(Path(filename).name),
            headers=self._headers(),
        )
        with urlopen(request, timeout=300.0) as response, target.open("wb") as handle:
            shutil.copyfileobj(response, handle)


class HttpReconstructionClient:
    def __init__(self, base_url: str, timeout_s: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def _json(
        self,
        path: str,
        method: str = "GET",
        payload: dict[str, object] | None = None,
    ) -> dict[str, object]:
        headers = {"Accept": "application/json"}
        data = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload).encode("utf-8")
        request = Request(self.base_url + path, method=method, headers=headers, data=data)
        with urlopen(request, timeout=self.timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))

    def health(self) -> dict[str, object]:
        return self._json("/api/health")

    def list_reconstructions(self) -> dict[str, object]:
        return self._json("/api/reconstructions")

    def submit(
        self,
        session_id: str,
        video_path: Path,
        parameters: dict[str, object],
    ) -> dict[str, object]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "video/mp4",
            "Content-Length": str(video_path.stat().st_size),
            "X-ReconBot-Session": session_id,
            "X-ReconBot-Parameters": json.dumps(parameters, separators=(",", ":")),
        }
        parsed = urlsplit(self.base_url)
        connection_type = (
            http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        )
        connection = connection_type(parsed.hostname, parsed.port, timeout=600.0)
        path = (parsed.path.rstrip("/") or "") + "/api/jobs"
        try:
            connection.putrequest("POST", path)
            for key, value in headers.items():
                connection.putheader(key, value)
            connection.endheaders()
            with video_path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    connection.send(chunk)
            response = connection.getresponse()
            body = response.read()
            if response.status >= 400:
                raise RuntimeError(
                    f"Reconstruction worker returned {response.status}: "
                    f"{body.decode('utf-8', errors='replace')}"
                )
            return json.loads(body.decode("utf-8"))
        finally:
            connection.close()

    def job(self, job_id: str) -> dict[str, object]:
        return self._json("/api/jobs/" + quote(job_id))

    def activate(self, identifier: str) -> dict[str, object]:
        return self._json(
            "/api/reconstructions/" + quote(identifier) + "/activate",
            "POST",
        )

    def delete(self, identifier: str) -> dict[str, object]:
        return self._json("/api/reconstructions/" + quote(identifier), "DELETE")

    def asset(
        self,
        identifier: str,
        filename: str,
        range_header: str | None = None,
    ) -> tuple[int, bytes, dict[str, str]]:
        headers = {}
        if range_header:
            headers["Range"] = range_header
        request = Request(
            self.base_url + f"/api/reconstructions/{quote(identifier)}/{filename}",
            headers=headers,
        )
        try:
            response = urlopen(request, timeout=300.0)
        except HTTPError as exc:
            response = exc
        with response:
            selected_headers = {
                key: value
                for key, value in response.headers.items()
                if key.lower()
                in {"content-type", "content-length", "content-range", "accept-ranges"}
            }
            return response.status, response.read(), selected_headers


@dataclass
class FieldSession:
    session_id: str
    started_at: str
    arc_direction: str = "right"
    notes: str = ""
    active: bool = True
    attempts: int = 0
    accepted: int = 0
    rejected: int = 0
    last_capture_at: str | None = None
    latest_image: str | None = None
    last_guidance: dict[str, object] | None = None
    last_quality: dict[str, object] | None = None
    last_latency_ms: float | None = None
    recording: bool = False
    video_started_at: float | None = None
    latest_video: str | None = None
    video_metadata: dict[str, object] | None = None
    reconstruction_job_id: str | None = None
    stopped_at: str | None = None
    capture_events: list[dict[str, object]] = field(default_factory=list)


class SessionStore:
    def __init__(self, root: str | Path) -> None:
        self.root = ensure_dir(root).resolve()
        self._lock = threading.Lock()
        self.current: FieldSession | None = None

    def start(self, session_id: str, arc_direction: str, notes: str = "") -> FieldSession:
        with self._lock:
            identifier = _safe_session_id(session_id)
            session_dir = ensure_dir(self.root / identifier)
            ensure_dir(session_dir / "images")
            ensure_dir(session_dir / "rejected")
            ensure_dir(session_dir / "video")
            self.current = FieldSession(
                session_id=identifier,
                started_at=_utc_now(),
                arc_direction="left" if arc_direction.lower() == "left" else "right",
                notes=notes,
            )
            self._persist_state()
            return self.current

    def stop(self) -> FieldSession:
        with self._lock:
            if self.current is None:
                raise RuntimeError("No field session is active")
            if self.current.recording:
                raise RuntimeError("Stop the video recording before finishing the session")
            self.current.active = False
            self.current.stopped_at = _utc_now()
            self._persist_state()
            return self.current

    def save_capture(self, payload: dict[str, object]) -> FieldSession:
        with self._lock:
            if self.current is None or not self.current.active:
                raise RuntimeError("Start a field session before capturing")
            session = self.current
            session.attempts += 1
            accepted = bool(payload.get("accepted"))
            target_name = f"capture_{session.attempts:04d}.jpg"
            subdir = "images" if accepted else "rejected"
            image_path = self.root / session.session_id / subdir / target_name
            image_path.write_bytes(base64.b64decode(str(payload["image_base64"])))
            relative_image = f"{subdir}/{target_name}"
            guidance = dict(payload.get("guidance") or {})
            guidance["recommendation"] = guidance_text(
                str(guidance.get("code", "unknown")),
                session.arc_direction,
            )
            event = {
                "attempt": session.attempts,
                "accepted": accepted,
                "captured_at": payload.get("captured_at"),
                "image": relative_image,
                "quality": payload.get("quality"),
                "guidance": guidance,
                "latency_ms": payload.get("latency_ms"),
                "camera_telemetry": payload.get("telemetry"),
                "pi_telemetry": collect_system_telemetry(self.root),
            }
            if accepted:
                session.accepted += 1
                session.latest_image = relative_image
            else:
                session.rejected += 1
            session.last_capture_at = _utc_now()
            session.last_guidance = guidance
            session.last_quality = dict(payload.get("quality") or {})
            session.last_latency_ms = float(payload.get("latency_ms") or 0.0)
            session.capture_events.append(event)
            session.capture_events = session.capture_events[-100:]
            metadata_path = self.root / session.session_id / "captures.jsonl"
            with metadata_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
            self._persist_state()
            return session

    def begin_video(self, payload: dict[str, object]) -> FieldSession:
        with self._lock:
            if self.current is None or not self.current.active:
                raise RuntimeError("Start a field session before recording")
            if self.current.recording:
                raise RuntimeError("A video recording is already in progress")
            self.current.recording = True
            self.current.video_started_at = float(payload.get("started_at") or time.time())
            self.current.video_metadata = dict(payload)
            self._persist_state()
            return self.current

    def video_target(self, filename: str) -> Path:
        with self._lock:
            if self.current is None or not self.current.active:
                raise RuntimeError("No active field session")
            safe_name = Path(filename).name
            if not safe_name:
                raise RuntimeError("Camera node returned an invalid video filename")
            return self.root / self.current.session_id / "video" / safe_name

    def finish_video(self, payload: dict[str, object], path: Path) -> FieldSession:
        with self._lock:
            if self.current is None:
                raise RuntimeError("No active field session")
            relative = path.relative_to(self.root / self.current.session_id).as_posix()
            metadata = dict(payload)
            metadata["stored_bytes"] = path.stat().st_size
            self.current.recording = False
            self.current.latest_video = relative
            self.current.video_metadata = metadata
            self._persist_state()
            return self.current

    def abort_video(self) -> None:
        with self._lock:
            if self.current is not None:
                self.current.recording = False
                self._persist_state()

    def set_reconstruction_job(self, job_id: str) -> FieldSession:
        with self._lock:
            if self.current is None:
                raise RuntimeError("No field session is active")
            self.current.reconstruction_job_id = job_id
            self._persist_state()
            return self.current

    def _persist_state(self) -> None:
        if self.current is None:
            return
        write_json(
            self.root / self.current.session_id / "session.json",
            asdict(self.current),
        )


def create_coordinator_app(
    camera_client: CameraClient,
    session_root: str | Path,
    dashboard_dist: str | Path | None = None,
    reconstruction_client: ReconstructionClient | None = None,
):
    try:
        from flask import Flask, Response, jsonify, request, send_file, send_from_directory
    except ImportError as exc:
        raise RuntimeError("Flask is required: install `python -m pip install -e .[dashboard]`.") from exc

    dist = Path(dashboard_dist).resolve() if dashboard_dist else None
    app = Flask(__name__, static_folder=None)
    store = SessionStore(session_root)
    started_at = time.time()

    def camera_health() -> dict[str, object]:
        try:
            health = camera_client.health()
            return {"connected": bool(health.get("ok")), **health}
        except (URLError, OSError, TimeoutError, ValueError) as exc:
            return {"connected": False, "error": str(exc)}

    def reconstruction_health() -> dict[str, object]:
        if reconstruction_client is None:
            return {"connected": False, "error": "Reconstruction worker is not configured"}
        try:
            health = reconstruction_client.health()
            return {"connected": bool(health.get("ok")), **health}
        except (URLError, OSError, TimeoutError, ValueError) as exc:
            return {"connected": False, "error": str(exc)}

    @app.get("/api/field/status")
    def status():
        return jsonify(
            {
                "ok": True,
                "role": "pi-coordinator",
                "uptime_s": time.time() - started_at,
                "camera": camera_health(),
                "reconstruction": reconstruction_health(),
                "pi": collect_system_telemetry(session_root),
                "session": asdict(store.current) if store.current else None,
            }
        )

    @app.post("/api/field/session/start")
    def start_session():
        body = request.get_json(silent=True) or {}
        try:
            camera_client.reset()
            session = store.start(
                str(body.get("session_id") or ""),
                str(body.get("arc_direction") or "right"),
                str(body.get("notes") or ""),
            )
        except (URLError, OSError, TimeoutError, ValueError) as exc:
            return jsonify({"error": f"Camera node unavailable: {exc}"}), 503
        return jsonify({"ok": True, "session": asdict(session)})

    @app.post("/api/field/capture")
    def capture():
        try:
            payload = camera_client.capture()
            session = store.save_capture(payload)
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409
        except (URLError, OSError, TimeoutError, ValueError) as exc:
            return jsonify({"error": f"Capture failed: {exc}"}), 503
        return jsonify(
            {
                "ok": True,
                "accepted": bool(payload.get("accepted")),
                "session": asdict(session),
            }
        )

    @app.get("/api/field/preview.jpg")
    def camera_preview():
        try:
            image = camera_client.preview()
        except (URLError, OSError, TimeoutError, ValueError) as exc:
            return jsonify({"error": f"Preview failed: {exc}"}), 503
        return Response(
            image,
            mimetype="image/jpeg",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @app.post("/api/field/video/start")
    def start_video():
        body = request.get_json(silent=True) or {}
        if store.current is None or not store.current.active:
            return jsonify({"error": "Start a field session before recording"}), 409
        try:
            payload = camera_client.start_video(
                store.current.session_id,
                float(body.get("fps") or 15.0),
            )
            session = store.begin_video(payload)
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409
        except (URLError, OSError, TimeoutError, ValueError) as exc:
            return jsonify({"error": f"Could not start recording: {exc}"}), 503
        return jsonify({"ok": True, "session": asdict(session)})

    @app.post("/api/field/video/stop")
    def stop_video():
        if store.current is None or not store.current.recording:
            return jsonify({"error": "No video recording is in progress"}), 409
        temporary: Path | None = None
        try:
            payload = camera_client.stop_video()
            target = store.video_target(str(payload.get("filename") or ""))
            temporary = target.with_suffix(target.suffix + ".part")
            camera_client.download_video(str(payload["filename"]), temporary)
            temporary.replace(target)
            session = store.finish_video(payload, target)
        except RuntimeError as exc:
            store.abort_video()
            return jsonify({"error": str(exc)}), 409
        except (URLError, OSError, TimeoutError, ValueError, KeyError, RuntimeError) as exc:
            store.abort_video()
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            return jsonify({"error": f"Could not save recording: {exc}"}), 503
        return jsonify({"ok": True, "session": asdict(session)})

    @app.post("/api/field/session/stop")
    def stop_session():
        try:
            session = store.stop()
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify({"ok": True, "session": asdict(session)})

    @app.post("/api/field/reconstruct")
    def reconstruct_video():
        if reconstruction_client is None:
            return jsonify({"error": "Reconstruction worker is not configured"}), 503
        if store.current is None or store.current.latest_video is None:
            return jsonify({"error": "Stop and save a field recording first"}), 409
        if store.current.recording:
            return jsonify({"error": "Stop the active recording before reconstruction"}), 409
        video_path = store.root / store.current.session_id / store.current.latest_video
        metadata = dict(store.current.video_metadata or {})
        parameters = {
            "title": store.current.session_id.replace("_", " ").title(),
            "duration_s": metadata.get("duration_s"),
            "frames": metadata.get("frames"),
            "fps": metadata.get("fps"),
            "width": metadata.get("width"),
            "height": metadata.get("height"),
            "every_n": 3,
            "min_blur": 15,
            "max_frames": 220,
            "target_faces": 300000,
        }
        try:
            payload = reconstruction_client.submit(
                store.current.session_id,
                video_path,
                parameters,
            )
            job = dict(payload.get("job") or {})
            session = store.set_reconstruction_job(str(job["id"]))
        except (URLError, OSError, TimeoutError, ValueError, KeyError, RuntimeError) as exc:
            return jsonify({"error": f"Could not start reconstruction: {exc}"}), 503
        return jsonify({"ok": True, "job": job, "session": asdict(session)}), 202

    @app.get("/api/reconstruction/jobs/<job_id>")
    def reconstruction_job(job_id: str):
        if reconstruction_client is None:
            return jsonify({"error": "Reconstruction worker is not configured"}), 503
        try:
            return jsonify(reconstruction_client.job(job_id))
        except HTTPError as exc:
            return jsonify({"error": exc.reason}), exc.code
        except (URLError, OSError, TimeoutError, ValueError) as exc:
            return jsonify({"error": f"Reconstruction worker unavailable: {exc}"}), 503

    @app.get("/api/reconstructions")
    def reconstruction_library():
        if reconstruction_client is None:
            return jsonify({"error": "Reconstruction worker is not configured"}), 503
        try:
            return jsonify(reconstruction_client.list_reconstructions())
        except (URLError, OSError, TimeoutError, ValueError) as exc:
            return jsonify({"error": f"Reconstruction worker unavailable: {exc}"}), 503

    @app.post("/api/reconstructions/<identifier>/activate")
    def activate_reconstruction(identifier: str):
        if reconstruction_client is None:
            return jsonify({"error": "Reconstruction worker is not configured"}), 503
        try:
            return jsonify(reconstruction_client.activate(identifier))
        except HTTPError as exc:
            return jsonify({"error": exc.reason}), exc.code
        except (URLError, OSError, TimeoutError, ValueError) as exc:
            return jsonify({"error": f"Reconstruction worker unavailable: {exc}"}), 503

    @app.delete("/api/reconstructions/<identifier>")
    def delete_reconstruction(identifier: str):
        if reconstruction_client is None:
            return jsonify({"error": "Reconstruction worker is not configured"}), 503
        try:
            return jsonify(reconstruction_client.delete(identifier))
        except HTTPError as exc:
            return jsonify({"error": exc.reason}), exc.code
        except (URLError, OSError, TimeoutError, ValueError) as exc:
            return jsonify({"error": f"Reconstruction worker unavailable: {exc}"}), 503

    @app.get("/api/reconstructions/<identifier>/<asset>")
    def reconstruction_asset(identifier: str, asset: str):
        if reconstruction_client is None:
            return jsonify({"error": "Reconstruction worker is not configured"}), 503
        if asset not in {"model.glb", "video.mp4"}:
            return jsonify({"error": "Asset not found"}), 404
        try:
            status_code, body, headers = reconstruction_client.asset(
                identifier,
                asset,
                request.headers.get("Range"),
            )
        except (URLError, OSError, TimeoutError, ValueError) as exc:
            return jsonify({"error": f"Reconstruction worker unavailable: {exc}"}), 503
        return Response(body, status=status_code, headers=headers)

    @app.get("/api/field/latest.jpg")
    def latest_image():
        if store.current is None or store.current.latest_image is None:
            return jsonify({"error": "No accepted capture available"}), 404
        path = store.root / store.current.session_id / store.current.latest_image
        return send_file(path, mimetype="image/jpeg", max_age=0)

    @app.get("/api/field/video.mp4")
    def latest_video():
        if store.current is None or store.current.latest_video is None:
            return jsonify({"error": "No completed video is available"}), 404
        path = store.root / store.current.session_id / store.current.latest_video
        return send_file(path, mimetype="video/mp4", conditional=True, max_age=0)

    @app.get("/api/field/events")
    def events():
        events = store.current.capture_events if store.current else []
        return jsonify({"events": events})

    @app.get("/")
    @app.get("/<path:asset>")
    def dashboard(asset: str = "index.html"):
        if dist is None or not dist.exists():
            return jsonify(
                {
                    "service": "ReconBot Pi coordinator",
                    "dashboard": "Build dashboard/ and pass --dashboard-dist.",
                }
            )
        candidate = dist / asset
        if candidate.exists() and candidate.is_file():
            return send_from_directory(dist, asset)
        return send_from_directory(dist, "index.html")

    return app
