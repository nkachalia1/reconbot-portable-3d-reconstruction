from __future__ import annotations

import io
import json
from pathlib import Path
import time

from reconbot.reconstruction_worker import (
    ReconstructionCatalog,
    WorkerConfig,
    create_reconstruction_app,
)


def _seed_library(root: Path) -> Path:
    seed = root / "seeds"
    run = seed / "seed-run"
    run.mkdir(parents=True, exist_ok=True)
    (run / "model.glb").write_bytes(b"seed-model")
    (run / "video.mp4").write_bytes(b"seed-video")
    (seed / "catalog.json").write_text(
        json.dumps(
            {
                "active_id": "seed-run",
                "items": [
                    {
                        "id": "seed-run",
                        "title": "Seed Run",
                        "created_at": "2026-06-17T00:00:00+00:00",
                        "status": "complete",
                        "source": "portfolio",
                        "metrics": {"registered_images": 10},
                        "viewer": {"up_axis": "Y-up"},
                        "asset_files": {
                            "model": "model.glb",
                            "video": "video.mp4",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return seed


def test_catalog_bootstraps_activates_and_deletes_seed(tmp_path: Path):
    catalog = ReconstructionCatalog(tmp_path / "library", _seed_library(tmp_path))

    payload = catalog.list_public()
    assert payload["active_id"] == "seed-run"
    assert payload["items"][0]["model_url"].endswith("/seed-run/model.glb")
    assert catalog.asset_path("seed-run", "video").read_bytes() == b"seed-video"

    seed_payload = json.loads(
        (tmp_path / "seeds" / "catalog.json").read_text(encoding="utf-8")
    )
    seed_payload["items"][0]["metrics"]["registered_images"] = 12
    (tmp_path / "seeds" / "catalog.json").write_text(
        json.dumps(seed_payload),
        encoding="utf-8",
    )
    refreshed = ReconstructionCatalog(tmp_path / "library", tmp_path / "seeds")
    assert refreshed.list_public()["items"][0]["metrics"]["registered_images"] == 12

    catalog.delete("seed-run")
    assert catalog.list_public()["items"] == []

    restarted = ReconstructionCatalog(tmp_path / "library", _seed_library(tmp_path))
    assert restarted.list_public()["items"] == []


def test_worker_api_publishes_completed_job(tmp_path: Path):
    seed = _seed_library(tmp_path)
    library = tmp_path / "library"
    config = WorkerConfig(
        project_root=tmp_path,
        library_root=library,
        seed_root=seed,
        intrinsics=tmp_path / "intrinsics.yaml",
    )

    def fake_runner(identifier, video_path, parameters, progress):
        progress("mesh", 80, "Building mesh")
        run = library / identifier
        (run / "model.glb").write_bytes(b"generated-model")
        record = {
            "id": identifier,
            "title": parameters["title"],
            "created_at": "2026-06-18T00:00:00+00:00",
            "status": "complete",
            "source": "field",
            "metrics": {"registered_images": 42},
            "viewer": {"up_axis": "Y-up"},
            "asset_files": {"model": "model.glb", "video": "video.mp4"},
        }
        return app.extensions["reconbot_catalog"].add(record)

    app = create_reconstruction_app(config, runner=fake_runner)
    client = app.test_client()
    response = client.post(
        "/api/jobs",
        data=io.BytesIO(b"recorded-video"),
        headers={
            "X-ReconBot-Session": "new run",
            "X-ReconBot-Parameters": json.dumps({"title": "New Run"}),
            "Content-Type": "video/mp4",
        },
    )
    assert response.status_code == 202
    job_id = response.get_json()["job"]["id"]

    job = None
    for _ in range(50):
        job = client.get(f"/api/jobs/{job_id}").get_json()["job"]
        if job["status"] in {"complete", "failed"}:
            break
        time.sleep(0.01)

    assert job["status"] == "complete"
    assert job["result"]["title"] == "New Run"
    assert client.get("/api/reconstructions/new_run/model.glb").data == b"generated-model"
    assert client.get("/api/reconstructions/new_run/video.mp4").data == b"recorded-video"
