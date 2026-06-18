"""Laptop reconstruction worker and persistent reconstruction library."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import threading
import time
from typing import Callable
from uuid import uuid4

from .frame_extraction import extract_frames
from .io_utils import ensure_dir, load_intrinsics_yaml, write_json


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_identifier(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_")
    return (normalized or datetime.now().strftime("field_%Y%m%d_%H%M%S"))[:80]


def _display_title(identifier: str) -> str:
    return " ".join(part.capitalize() for part in identifier.replace("-", "_").split("_") if part)


def _windows_to_wsl(path: Path) -> str:
    resolved = path.resolve()
    if os.name != "nt":
        return str(resolved)
    drive = resolved.drive.rstrip(":").lower()
    suffix = resolved.as_posix().split(":", 1)[1]
    return f"/mnt/{drive}{suffix}"


def _ply_counts(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(65536).decode("ascii", errors="ignore")
    vertices = int(re.search(r"element vertex (\d+)", header).group(1))
    face_match = re.search(r"element face (\d+)", header)
    return vertices, int(face_match.group(1)) if face_match else 0


def _parse_model_analyzer(output: str) -> dict[str, object]:
    patterns = {
        "cameras": r"Cameras:\s+(\d+)",
        "images": r"Images:\s+(\d+)",
        "registered_images": r"Registered images:\s+(\d+)",
        "sparse_points": r"Points:\s+(\d+)",
        "observations": r"Observations:\s+(\d+)",
        "mean_track_length": r"Mean track length:\s+([0-9.eE+-]+)",
        "mean_observations_per_image": r"Mean observations per image:\s+([0-9.eE+-]+)",
        "mean_reprojection_error_px": r"Mean reprojection error:\s+([0-9.eE+-]+)px",
    }
    metrics: dict[str, object] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            metrics[key] = float(match.group(1)) if "." in match.group(1) else int(match.group(1))
    return metrics


@dataclass(frozen=True)
class WorkerConfig:
    project_root: Path
    library_root: Path
    seed_root: Path
    intrinsics: Path
    wsl_distro: str = "Ubuntu-22.04"
    openmvs_bin: Path | None = None
    node_executable: str = "node"
    every_n: int = 3
    min_blur: float = 15.0
    max_frames: int = 220
    target_faces: int = 300_000
    keep_work: bool = False


class ReconstructionCatalog:
    """Thread-safe JSON catalog whose records own their video and GLB artifacts."""

    def __init__(self, root: str | Path, seed_root: str | Path | None = None) -> None:
        self.root = ensure_dir(root).resolve()
        self.state_path = self.root / "catalog.json"
        self.seed_root = Path(seed_root).resolve() if seed_root else None
        self._lock = threading.RLock()
        self._state = self._load_state()
        self._bootstrap_seeds()

    def _load_state(self) -> dict[str, object]:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {"active_id": None, "records": [], "deleted_seed_ids": []}

    def _save(self) -> None:
        write_json(self.state_path, self._state)

    def _bootstrap_seeds(self) -> None:
        if self.seed_root is None:
            return
        seed_catalog = self.seed_root / "catalog.json"
        if not seed_catalog.exists():
            return
        payload = json.loads(seed_catalog.read_text(encoding="utf-8"))
        deleted = set(self._state.get("deleted_seed_ids") or [])
        records = list(self._state.get("records") or [])
        existing = {str(record["id"]): record for record in records}
        changed = False
        for seed in payload.get("items", []):
            identifier = str(seed["id"])
            if identifier in deleted:
                continue
            if identifier in existing:
                current = existing[identifier]
                if current.get("seeded"):
                    current.update(
                        {
                            **seed,
                            "seeded": True,
                            "asset_files": {
                                key: Path(value).name
                                for key, value in dict(seed.get("asset_files") or {}).items()
                            },
                        }
                    )
                    changed = True
                continue
            source_dir = self.seed_root / identifier
            target_dir = self.root / identifier
            if source_dir.exists():
                shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
            record = dict(seed)
            record["seeded"] = True
            record["asset_files"] = {
                key: Path(value).name
                for key, value in dict(seed.get("asset_files") or {}).items()
            }
            records.append(record)
            existing[identifier] = record
            changed = True
        if changed:
            self._state["records"] = records
        if not self._state.get("active_id"):
            requested = payload.get("active_id")
            available = {str(record["id"]) for record in records}
            self._state["active_id"] = requested if requested in available else next(iter(available), None)
            changed = True
        if changed:
            self._save()

    def _record(self, identifier: str) -> dict[str, object]:
        for record in self._state.get("records", []):
            if record.get("id") == identifier:
                return record
        raise KeyError(identifier)

    def public_record(self, record: dict[str, object]) -> dict[str, object]:
        identifier = str(record["id"])
        payload = {key: value for key, value in record.items() if key != "asset_files"}
        files = dict(record.get("asset_files") or {})
        payload["model_url"] = (
            f"/api/reconstructions/{identifier}/model.glb" if files.get("model") else None
        )
        payload["video_url"] = (
            f"/api/reconstructions/{identifier}/video.mp4" if files.get("video") else None
        )
        return payload

    def list_public(self) -> dict[str, object]:
        with self._lock:
            records = sorted(
                self._state.get("records", []),
                key=lambda item: str(item.get("created_at") or ""),
                reverse=True,
            )
            return {
                "active_id": self._state.get("active_id"),
                "items": [self.public_record(record) for record in records],
            }

    def add(self, record: dict[str, object]) -> dict[str, object]:
        with self._lock:
            records = [
                item for item in self._state.get("records", []) if item.get("id") != record.get("id")
            ]
            records.append(record)
            self._state["records"] = records
            self._state["active_id"] = record["id"]
            self._save()
            return self.public_record(record)

    def activate(self, identifier: str) -> dict[str, object]:
        with self._lock:
            record = self._record(identifier)
            self._state["active_id"] = identifier
            self._save()
            return self.public_record(record)

    def delete(self, identifier: str) -> None:
        with self._lock:
            record = self._record(identifier)
            records = [item for item in self._state.get("records", []) if item.get("id") != identifier]
            self._state["records"] = records
            if record.get("seeded"):
                deleted = set(self._state.get("deleted_seed_ids") or [])
                deleted.add(identifier)
                self._state["deleted_seed_ids"] = sorted(deleted)
            shutil.rmtree(self.root / identifier, ignore_errors=True)
            if self._state.get("active_id") == identifier:
                newest = sorted(
                    records,
                    key=lambda item: str(item.get("created_at") or ""),
                    reverse=True,
                )
                self._state["active_id"] = newest[0]["id"] if newest else None
            self._save()

    def asset_path(self, identifier: str, kind: str) -> Path:
        with self._lock:
            record = self._record(identifier)
            filename = dict(record.get("asset_files") or {}).get(kind)
            if not filename:
                raise FileNotFoundError(f"{kind} is unavailable for {identifier}")
            path = (self.root / identifier / str(filename)).resolve()
            if self.root not in path.parents or not path.is_file():
                raise FileNotFoundError(path)
            return path


class ReconstructionPipeline:
    """Run frame filtering, WSL COLMAP, Windows OpenMVS, and GLB publishing."""

    def __init__(self, config: WorkerConfig, catalog: ReconstructionCatalog) -> None:
        self.config = config
        self.catalog = catalog

    def _openmvs_executable(self, name: str) -> str:
        if self.config.openmvs_bin:
            candidate = self.config.openmvs_bin / f"{name}.exe"
            if candidate.exists():
                return str(candidate)
        discovered = shutil.which(name) or shutil.which(f"{name}.exe")
        if discovered:
            return discovered
        raise RuntimeError(
            f"{name} was not found. Set RECONBOT_OPENMVS_BIN to the OpenMVS Release folder."
        )

    def _run(
        self,
        command: list[str],
        log_path: Path,
        cwd: Path | None = None,
    ) -> str:
        environment = os.environ.copy()
        for variable in ("QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR"):
            environment.pop(variable, None)
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        ensure_dir(log_path.parent)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n$ {shlex.join(command)}\n")
            handle.write(completed.stdout or "")
        if completed.returncode:
            tail = "\n".join((completed.stdout or "").splitlines()[-12:])
            raise RuntimeError(f"{Path(command[0]).name} failed:\n{tail}")
        return completed.stdout or ""

    def _run_colmap(self, arguments: list[str], log_path: Path) -> str:
        if os.name != "nt":
            return self._run(["colmap", *arguments], log_path, self.config.project_root)
        wsl_arguments = [
            _windows_to_wsl(Path(value)) if re.match(r"^[A-Za-z]:[\\/]", value) else value
            for value in arguments
        ]
        shell_command = (
            f"cd {shlex.quote(_windows_to_wsl(self.config.project_root))} && "
            "unset QT_PLUGIN_PATH QT_QPA_PLATFORM_PLUGIN_PATH QT_QPA_FONTDIR && "
            f"colmap {shlex.join(wsl_arguments)}"
        )
        return self._run(
            ["wsl.exe", "-d", self.config.wsl_distro, "--", "bash", "-lc", shell_command],
            log_path,
        )

    def __call__(
        self,
        identifier: str,
        video_path: Path,
        parameters: dict[str, object],
        progress: Callable[[str, float, str], None],
    ) -> dict[str, object]:
        config = self.config
        record_dir = ensure_dir(config.library_root / identifier)
        work = ensure_dir(record_dir / "work")
        frame_dir = work / "frames"
        colmap_dir = work / "colmap"
        sparse_dir = colmap_dir / "sparse"
        dense_dir = colmap_dir / "dense"
        openmvs_dir = ensure_dir(work / "openmvs")
        pipeline_log = record_dir / "pipeline.log"
        started = time.perf_counter()

        progress("quality_gate", 8, "Extracting sharp, overlapping frames")
        extraction = extract_frames(
            video_path,
            frame_dir,
            every_n=int(parameters.get("every_n") or config.every_n),
            min_blur=float(parameters.get("min_blur") or config.min_blur),
            max_frames=int(parameters.get("max_frames") or config.max_frames),
        )
        if extraction.written_frames < 20:
            raise RuntimeError(
                f"Only {extraction.written_frames} sharp frames survived the quality gate; "
                "record a slower orbit or lower the blur threshold."
            )

        ensure_dir(sparse_dir)
        database = colmap_dir / "database.db"
        feature_args = [
            "feature_extractor",
            "--database_path",
            str(database),
            "--image_path",
            str(frame_dir),
            "--ImageReader.camera_model",
            "OPENCV",
            "--ImageReader.single_camera",
            "1",
        ]
        if config.intrinsics.exists():
            camera_matrix, distortion, _ = load_intrinsics_yaml(config.intrinsics)
            distortion_values = distortion.reshape(-1).tolist()
            while len(distortion_values) < 4:
                distortion_values.append(0.0)
            values = [
                camera_matrix[0, 0],
                camera_matrix[1, 1],
                camera_matrix[0, 2],
                camera_matrix[1, 2],
                *distortion_values[:4],
            ]
            feature_args.extend(
                ["--ImageReader.camera_params", ",".join(str(value) for value in values)]
            )

        progress("sparse_sfm", 18, "Extracting and matching calibrated image features")
        self._run_colmap(feature_args, pipeline_log)
        self._run_colmap(
            [
                "exhaustive_matcher",
                "--database_path",
                str(database),
                "--SiftMatching.guided_matching",
                "1",
            ],
            pipeline_log,
        )
        progress("sparse_sfm", 32, "Recovering camera poses and sparse landmarks")
        self._run_colmap(
            [
                "mapper",
                "--database_path",
                str(database),
                "--image_path",
                str(frame_dir),
                "--output_path",
                str(sparse_dir),
            ],
            pipeline_log,
        )
        model = sparse_dir / "0"
        if not model.exists():
            raise RuntimeError("COLMAP did not produce a connected sparse model")
        analyzer = self._run_colmap(["model_analyzer", "--path", str(model)], pipeline_log)
        metrics = _parse_model_analyzer(analyzer)
        metrics["extracted_frames"] = extraction.written_frames
        metrics["blur_rejected_samples"] = extraction.skipped_blurry_frames
        metrics["sampled_frames"] = (
            extraction.written_frames + extraction.skipped_blurry_frames
        )
        registered = int(metrics.get("registered_images") or 0)
        metrics["registration_ratio"] = (
            registered / extraction.written_frames if extraction.written_frames else 0.0
        )

        progress("undistort", 43, "Preparing calibrated images for dense reconstruction")
        self._run_colmap(
            [
                "image_undistorter",
                "--image_path",
                str(frame_dir),
                "--input_path",
                str(model),
                "--output_path",
                str(dense_dir),
                "--output_type",
                "COLMAP",
            ],
            pipeline_log,
        )

        progress("dense_mvs", 52, "Computing multi-view depth maps on the laptop")
        self._run(
            [
                self._openmvs_executable("InterfaceCOLMAP"),
                "-i",
                str(dense_dir),
                "-o",
                "scene.mvs",
                "--image-folder",
                str(dense_dir / "images"),
            ],
            pipeline_log,
            openmvs_dir,
        )
        self._run(
            [self._openmvs_executable("DensifyPointCloud"), "scene.mvs"],
            pipeline_log,
            openmvs_dir,
        )
        dense_ply = openmvs_dir / "scene_dense.ply"
        metrics["dense_points"] = _ply_counts(dense_ply)[0]
        metrics["dense_depth_maps"] = len(list(openmvs_dir.glob("depth*.dmap")))

        progress("mesh", 78, "Reconstructing and simplifying the surface mesh")
        mesh_name = "scene_dense_mesh.ply"
        self._run(
            [
                self._openmvs_executable("ReconstructMesh"),
                "scene_dense.mvs",
                "-p",
                "scene_dense.ply",
                "-o",
                mesh_name,
                "--target-face-num",
                str(int(parameters.get("target_faces") or config.target_faces)),
            ],
            pipeline_log,
            openmvs_dir,
        )
        mesh_path = openmvs_dir / mesh_name
        metrics["mesh_vertices"], metrics["mesh_faces"] = _ply_counts(mesh_path)
        metrics["full_mesh_vertices"] = metrics["mesh_vertices"]
        metrics["full_mesh_faces"] = metrics["mesh_faces"]

        progress("texture", 89, "Projecting source images onto the presentation mesh")
        textured_name = "scene_dense_texture.ply"
        self._run(
            [
                self._openmvs_executable("TextureMesh"),
                "scene_dense.mvs",
                "-m",
                mesh_name,
                "-o",
                textured_name,
                "--resolution-level",
                "1",
                "--virtual-face-images",
                "3",
                "--patch-packing-heuristic",
                "100",
                "--max-texture-size",
                "4096",
                "--global-seam-leveling",
                "0",
                "--local-seam-leveling",
                "0",
            ],
            pipeline_log,
            openmvs_dir,
        )
        texture_candidates = sorted(openmvs_dir.glob("scene_dense_texture*.png"))
        if not texture_candidates:
            raise RuntimeError("OpenMVS did not create a texture atlas")

        progress("publish", 96, "Packaging the textured mesh for the dashboard")
        converter = config.project_root / "dashboard" / "scripts" / "convert-textured-ply.mjs"
        self._run(
            [
                config.node_executable,
                str(converter),
                "--ply",
                str(openmvs_dir / textured_name),
                "--texture",
                str(texture_candidates[0]),
                "--out-dir",
                str(record_dir),
                "--out-glb",
                str(record_dir / "model.glb"),
                "--out-texture",
                str(record_dir / "texture.png"),
            ],
            pipeline_log,
            config.project_root,
        )

        metrics["processing_time_s"] = round(time.perf_counter() - started, 2)
        metrics["video_duration_s"] = parameters.get("duration_s")
        metrics["video_frames"] = parameters.get("frames")
        write_json(record_dir / "metrics.json", metrics)
        record = {
            "id": identifier,
            "title": str(parameters.get("title") or _display_title(identifier)),
            "created_at": _utc_now(),
            "status": "complete",
            "source": "field",
            "metrics": metrics,
            "viewer": {
                "rotation_x": 0,
                "up_axis": "Y-up",
                "scale_label": "Auto-fit",
                "metric": False,
            },
            "asset_files": {
                "model": "model.glb",
                "video": "video.mp4",
                "metrics": "metrics.json",
                "log": "pipeline.log",
            },
        }
        public = self.catalog.add(record)
        if not config.keep_work:
            shutil.rmtree(work, ignore_errors=True)
            (record_dir / "texture.png").unlink(missing_ok=True)
        return public


class ReconstructionJobManager:
    def __init__(
        self,
        root: str | Path,
        runner: Callable[
            [str, Path, dict[str, object], Callable[[str, float, str], None]],
            dict[str, object],
        ],
    ) -> None:
        self.root = ensure_dir(root).resolve()
        self.runner = runner
        self._lock = threading.RLock()
        self._jobs: dict[str, dict[str, object]] = {}

    def submit(
        self,
        session_id: str,
        source_stream,
        parameters: dict[str, object],
    ) -> dict[str, object]:
        identifier = _safe_identifier(session_id)
        target_dir = ensure_dir(self.root / identifier)
        video_path = target_dir / "video.mp4"
        if video_path.exists():
            identifier = f"{identifier}_{datetime.now().strftime('%H%M%S')}"
            target_dir = ensure_dir(self.root / identifier)
            video_path = target_dir / "video.mp4"
        with video_path.open("wb") as handle:
            shutil.copyfileobj(source_stream, handle, length=1024 * 1024)
        job_id = uuid4().hex
        job = {
            "id": job_id,
            "reconstruction_id": identifier,
            "status": "queued",
            "stage": "queued",
            "progress": 0.0,
            "message": "Video saved; waiting for the reconstruction worker",
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "result": None,
            "error": None,
        }
        with self._lock:
            self._jobs[job_id] = job
        thread = threading.Thread(
            target=self._execute,
            args=(job_id, identifier, video_path, parameters),
            daemon=True,
            name=f"reconstruction-{identifier}",
        )
        thread.start()
        return dict(job)

    def _update(self, job_id: str, **changes: object) -> None:
        with self._lock:
            self._jobs[job_id].update(changes)
            self._jobs[job_id]["updated_at"] = _utc_now()

    def _execute(
        self,
        job_id: str,
        identifier: str,
        video_path: Path,
        parameters: dict[str, object],
    ) -> None:
        self._update(job_id, status="running", stage="quality_gate", progress=3.0)

        def progress(stage: str, percent: float, message: str) -> None:
            self._update(
                job_id,
                status="running",
                stage=stage,
                progress=float(percent),
                message=message,
            )

        try:
            result = self.runner(identifier, video_path, parameters, progress)
        except Exception as exc:  # noqa: BLE001 - surfaced as a durable job failure
            self._update(
                job_id,
                status="failed",
                stage="failed",
                message=str(exc),
                error=str(exc),
            )
            return
        self._update(
            job_id,
            status="complete",
            stage="complete",
            progress=100.0,
            message="Reconstruction ready",
            result=result,
        )

    def get(self, job_id: str) -> dict[str, object]:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return dict(self._jobs[job_id])


def create_reconstruction_app(
    config: WorkerConfig,
    runner: Callable[
        [str, Path, dict[str, object], Callable[[str, float, str], None]],
        dict[str, object],
    ]
    | None = None,
):
    try:
        from flask import Flask, jsonify, request, send_file
    except ImportError as exc:
        raise RuntimeError("Flask is required: install `python -m pip install -e .[dashboard]`.") from exc

    catalog = ReconstructionCatalog(config.library_root, config.seed_root)
    pipeline = runner or ReconstructionPipeline(config, catalog)
    jobs = ReconstructionJobManager(config.library_root, pipeline)
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024
    app.extensions["reconbot_catalog"] = catalog
    app.extensions["reconbot_jobs"] = jobs

    @app.get("/api/health")
    def health():
        return jsonify({"ok": True, "role": "reconstruction-worker"})

    @app.get("/api/reconstructions")
    def reconstructions():
        return jsonify(catalog.list_public())

    @app.post("/api/reconstructions/<identifier>/activate")
    def activate(identifier: str):
        try:
            record = catalog.activate(identifier)
        except KeyError:
            return jsonify({"error": "Reconstruction not found"}), 404
        return jsonify({"ok": True, "record": record, **catalog.list_public()})

    @app.delete("/api/reconstructions/<identifier>")
    def delete(identifier: str):
        try:
            catalog.delete(identifier)
        except KeyError:
            return jsonify({"error": "Reconstruction not found"}), 404
        return jsonify({"ok": True, **catalog.list_public()})

    @app.get("/api/reconstructions/<identifier>/model.glb")
    def model(identifier: str):
        try:
            path = catalog.asset_path(identifier, "model")
        except (KeyError, FileNotFoundError):
            return jsonify({"error": "Model not found"}), 404
        return send_file(path, mimetype="model/gltf-binary", conditional=True, max_age=0)

    @app.get("/api/reconstructions/<identifier>/video.mp4")
    def video(identifier: str):
        try:
            path = catalog.asset_path(identifier, "video")
        except (KeyError, FileNotFoundError):
            return jsonify({"error": "Video not found"}), 404
        return send_file(path, mimetype="video/mp4", conditional=True, max_age=0)

    @app.post("/api/jobs")
    def submit_job():
        session_id = request.headers.get("X-ReconBot-Session", "")
        if not session_id:
            return jsonify({"error": "X-ReconBot-Session is required"}), 400
        try:
            parameters = json.loads(request.headers.get("X-ReconBot-Parameters", "{}"))
        except json.JSONDecodeError:
            return jsonify({"error": "X-ReconBot-Parameters must be valid JSON"}), 400
        job = jobs.submit(session_id, request.stream, parameters)
        return jsonify({"ok": True, "job": job}), 202

    @app.get("/api/jobs/<job_id>")
    def job(job_id: str):
        try:
            payload = jobs.get(job_id)
        except KeyError:
            return jsonify({"error": "Job not found"}), 404
        return jsonify({"job": payload})

    return app
