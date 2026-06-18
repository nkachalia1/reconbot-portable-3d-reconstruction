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

    def download_video(self, filename: str, target: Path) -> None:
        assert filename == "field-video.mp4"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"fake-video")


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


def test_coordinator_http_workflow(tmp_path: Path):
    app = create_coordinator_app(FakeCameraClient(), tmp_path)
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

    stopped = client.post("/api/field/session/stop")
    assert stopped.status_code == 200
    assert stopped.get_json()["session"]["active"] is False
