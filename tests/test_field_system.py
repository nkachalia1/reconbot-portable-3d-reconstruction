from __future__ import annotations

import base64
from pathlib import Path

from reconbot.coordinator import SessionStore, create_coordinator_app


class FakeCameraClient:
    def health(self) -> dict[str, object]:
        return {"ok": True, "role": "camera-node"}

    def reset(self) -> dict[str, object]:
        return {"ok": True}

    def capture(self) -> dict[str, object]:
        return {
            "ok": True,
            "captured_at": 1_750_000_000.0,
            "image_base64": base64.b64encode(b"fake-jpeg").decode("ascii"),
            "accepted": True,
            "quality": {
                "blur_score": 120.0,
                "mean_intensity": 110.0,
                "dark_fraction": 0.01,
                "bright_fraction": 0.02,
            },
            "guidance": {
                "code": "reference_frame",
                "recommendation": "placeholder",
                "match_count": 0,
                "geometric_inlier_ratio": 0.0,
                "median_motion_px": 0.0,
            },
            "latency_ms": 18.0,
            "telemetry": {"hostname": "laptop"},
        }

    def preview(self) -> bytes:
        return b"preview-jpeg"

    def start_video(self, session_id: str, fps: float) -> dict[str, object]:
        return {
            "ok": True,
            "recording": True,
            "frames": 1,
            "fps": fps,
            "started_at": 1_750_000_000.0,
        }

    def stop_video(self) -> dict[str, object]:
        return {
            "ok": True,
            "filename": "field-video.mp4",
            "started_at": 1_750_000_000.0,
            "stopped_at": 1_750_000_010.0,
            "duration_s": 10.0,
            "frames": 150,
            "fps": 15.0,
            "width": 1280,
            "height": 720,
            "bytes": 10,
        }

    def latest_video(self) -> dict[str, object]:
        return {
            **self.stop_video(),
            "filename": "integration_1750000000.mp4",
        }

    def download_video(self, filename: str, target: Path) -> None:
        assert filename in {"field-video.mp4", "integration_1750000000.mp4"}
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"fake-video")


class FakeReconstructionClient:
    def health(self) -> dict[str, object]:
        return {"ok": True, "role": "reconstruction-worker"}

    def list_reconstructions(self) -> dict[str, object]:
        return {
            "active_id": "integration",
            "items": [{"id": "integration", "title": "Integration"}],
        }

    def submit(
        self,
        session_id: str,
        video_path: Path,
        parameters: dict[str, object],
    ) -> dict[str, object]:
        assert session_id == "integration"
        assert video_path.read_bytes() == b"fake-video"
        assert parameters["target_faces"] == 300000
        return {
            "job": {
                "id": "job-1",
                "reconstruction_id": "integration",
                "status": "queued",
            }
        }

    def job(self, job_id: str) -> dict[str, object]:
        assert job_id == "job-1"
        return {"job": {"id": job_id, "status": "complete"}}

    def activate(self, identifier: str) -> dict[str, object]:
        return {"active_id": identifier, "items": []}

    def delete(self, identifier: str) -> dict[str, object]:
        return {"active_id": None, "items": [], "deleted": identifier}

    def asset(
        self,
        identifier: str,
        filename: str,
        range_header: str | None = None,
    ) -> tuple[int, bytes, dict[str, str]]:
        assert identifier == "integration"
        return 200, f"{filename}:{range_header}".encode(), {"Content-Type": "application/octet-stream"}


def test_session_store_persists_capture_artifacts(tmp_path: Path):
    store = SessionStore(tmp_path)
    store.start("demo session", "left")
    session = store.save_capture(FakeCameraClient().capture())

    session_dir = tmp_path / "demo_session"
    assert session.accepted == 1
    assert (session_dir / "images" / "capture_0001.jpg").read_bytes() == b"fake-jpeg"
    assert (session_dir / "captures.jsonl").exists()
    assert (session_dir / "session.json").exists()
    assert "left" in str(session.last_guidance["recommendation"])


def test_session_store_restores_latest_active_session(tmp_path: Path):
    store = SessionStore(tmp_path)
    store.start("restored session", "right")

    restored = SessionStore(tmp_path)

    assert restored.current is not None
    assert restored.current.session_id == "restored_session"
    assert restored.current.active is True


def test_coordinator_http_workflow(tmp_path: Path):
    app = create_coordinator_app(
        FakeCameraClient(),
        tmp_path,
        reconstruction_client=FakeReconstructionClient(),
    )
    client = app.test_client()

    status = client.get("/api/field/status")
    assert status.status_code == 200
    assert status.get_json()["camera"]["connected"] is True

    started = client.post(
        "/api/field/session/start",
        json={"session_id": "integration", "arc_direction": "right"},
    )
    assert started.status_code == 200

    captured = client.post("/api/field/capture")
    assert captured.status_code == 200
    assert captured.get_json()["session"]["accepted"] == 1

    latest = client.get("/api/field/latest.jpg")
    assert latest.status_code == 200
    assert latest.data == b"fake-jpeg"

    preview = client.get("/api/field/preview.jpg")
    assert preview.status_code == 200
    assert preview.data == b"preview-jpeg"

    video_started = client.post("/api/field/video/start", json={"fps": 15})
    assert video_started.status_code == 200
    assert video_started.get_json()["session"]["recording"] is True

    video_stopped = client.post("/api/field/video/stop")
    assert video_stopped.status_code == 200
    assert video_stopped.get_json()["session"]["recording"] is False
    assert video_stopped.get_json()["session"]["latest_video"] == "video/field-video.mp4"

    video = client.get("/api/field/video.mp4")
    assert video.status_code == 200
    assert video.data == b"fake-video"

    reconstruction = client.post("/api/field/reconstruct")
    assert reconstruction.status_code == 202
    assert reconstruction.get_json()["job"]["id"] == "job-1"

    job = client.get("/api/reconstruction/jobs/job-1")
    assert job.status_code == 200
    assert job.get_json()["job"]["status"] == "complete"

    library = client.get("/api/reconstructions")
    assert library.status_code == 200
    assert library.get_json()["active_id"] == "integration"

    model = client.get("/api/reconstructions/integration/model.glb")
    assert model.status_code == 200
    assert model.data.startswith(b"model.glb")

    stopped = client.post("/api/field/session/stop")
    assert stopped.status_code == 200
    assert stopped.get_json()["session"]["active"] is False


def test_coordinator_recovers_camera_video_after_restart(tmp_path: Path):
    first_store = SessionStore(tmp_path)
    first_store.start("integration", "right")
    first_store.begin_video({"started_at": 1_750_000_000.0, "fps": 15.0})
    first_store.abort_video()

    app = create_coordinator_app(FakeCameraClient(), tmp_path)
    client = app.test_client()

    recovered = client.post("/api/field/video/recover")

    assert recovered.status_code == 200
    session = recovered.get_json()["session"]
    assert session["latest_video"] == "video/integration_1750000000.mp4"
    assert (tmp_path / "integration" / session["latest_video"]).read_bytes() == b"fake-video"
