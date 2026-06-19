"""Nerfstudio adapters that reuse ReconBot keyframes and COLMAP camera poses."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Callable, Literal

from .io_utils import ensure_dir, write_json


NeuralMethod = Literal["nerfacto", "instant-ngp", "splatfacto"]
SUPPORTED_NEURAL_METHODS: tuple[NeuralMethod, ...] = (
    "nerfacto",
    "instant-ngp",
    "splatfacto",
)


@dataclass(frozen=True)
class NeuralStage:
    """One external command in a neural reconstruction run."""

    name: str
    command: tuple[str, ...]


@dataclass(frozen=True)
class NeuralPlan:
    """Deterministic Nerfstudio command plan for inspection and execution."""

    method: NeuralMethod
    data_dir: Path
    runs_dir: Path
    export_dir: Path
    export_kind: str
    stages: tuple[NeuralStage, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "engine": "nerfstudio",
            "method": self.method,
            "data_dir": str(self.data_dir),
            "runs_dir": str(self.runs_dir),
            "export_dir": str(self.export_dir),
            "export_kind": self.export_kind,
            "stages": [
                {"name": stage.name, "command": list(stage.command)}
                for stage in self.stages
            ],
        }


@dataclass(frozen=True)
class NeuralArtifacts:
    """Published artifacts and timing from a completed neural run."""

    method: NeuralMethod
    config: Path
    primary_artifact: Path
    export_kind: str
    processing_time_s: float
    report: Path


def build_nerfstudio_plan(
    images: str | Path,
    colmap_model: str | Path,
    output_dir: str | Path,
    method: NeuralMethod = "nerfacto",
    max_iterations: int = 10_000,
    ns_process_data: str = "ns-process-data",
    ns_train: str = "ns-train",
    ns_export: str = "ns-export",
) -> NeuralPlan:
    """Build commands for a Nerfstudio method without executing external tools."""
    if method not in SUPPORTED_NEURAL_METHODS:
        raise ValueError(
            f"Unsupported neural method {method!r}; choose from "
            f"{', '.join(SUPPORTED_NEURAL_METHODS)}"
        )
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    image_path = Path(images).resolve()
    colmap_path = Path(colmap_model).resolve()
    root = Path(output_dir).resolve()
    data_dir = root / "data"
    runs_dir = root / "runs"
    export_dir = root / "export"
    config_placeholder = "{config}"

    process = NeuralStage(
        "process_data",
        (
            ns_process_data,
            "images",
            "--data",
            str(image_path),
            "--output-dir",
            str(data_dir),
            "--skip-colmap",
            "--colmap-model-path",
            str(colmap_path),
        ),
    )
    train = NeuralStage(
        "train",
        (
            ns_train,
            method,
            "--data",
            str(data_dir),
            "--output-dir",
            str(runs_dir),
            "--max-num-iterations",
            str(max_iterations),
            "--viewer.quit-on-train-completion",
            "True",
        ),
    )
    if method == "splatfacto":
        export_kind = "gaussian-splat"
        export_command = (
            ns_export,
            "gaussian-splat",
            "--load-config",
            config_placeholder,
            "--output-dir",
            str(export_dir),
        )
    else:
        export_kind = "poisson-mesh"
        export_command = (
            ns_export,
            "poisson",
            "--load-config",
            config_placeholder,
            "--output-dir",
            str(export_dir),
            "--num-points",
            "1000000",
            "--remove-outliers",
            "True",
            "--normal-method",
            "open3d",
        )

    return NeuralPlan(
        method=method,
        data_dir=data_dir,
        runs_dir=runs_dir,
        export_dir=export_dir,
        export_kind=export_kind,
        stages=(process, train, NeuralStage("export", export_command)),
    )


def _require_executables(stages: tuple[NeuralStage, ...]) -> None:
    missing = sorted(
        {
            stage.command[0]
            for stage in stages
            if not Path(stage.command[0]).is_file() and shutil.which(stage.command[0]) is None
        }
    )
    if missing:
        raise RuntimeError(
            "Neural reconstruction requires a CUDA-capable Nerfstudio environment. "
            f"Missing executables: {', '.join(missing)}"
        )


def _run_command(
    command: tuple[str, ...],
    log_path: Path,
    cwd: Path,
    environment: dict[str, str] | None,
) -> None:
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        env=environment,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n$ {' '.join(command)}\n")
        handle.write(completed.stdout or "")
    if completed.returncode:
        tail = "\n".join((completed.stdout or "").splitlines()[-20:])
        raise RuntimeError(f"{Path(command[0]).name} failed:\n{tail}")


def run_nerfstudio(
    plan: NeuralPlan,
    output_dir: str | Path,
    progress: Callable[[str, float, str], None] | None = None,
    environment: dict[str, str] | None = None,
) -> NeuralArtifacts:
    """Execute a plan in an environment where Nerfstudio and CUDA are installed."""
    _require_executables(plan.stages)
    root = ensure_dir(output_dir).resolve()
    log_path = root / "neural_pipeline.log"
    started = time.perf_counter()
    percentages = {"process_data": 10.0, "train": 25.0, "export": 90.0}
    messages = {
        "process_data": "Preparing images and imported COLMAP poses",
        "train": f"Training {plan.method} neural representation",
        "export": f"Exporting {plan.export_kind}",
    }

    config: Path | None = None
    for stage in plan.stages:
        if progress:
            progress(stage.name, percentages[stage.name], messages[stage.name])
        command = stage.command
        if stage.name == "export":
            configs = sorted(
                plan.runs_dir.rglob("config.yml"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if not configs:
                raise RuntimeError("Nerfstudio training completed without writing config.yml")
            config = configs[0]
            command = tuple(str(config) if value == "{config}" else value for value in command)
        ensure_dir(plan.export_dir if stage.name == "export" else root)
        _run_command(command, log_path, root, environment)

    assert config is not None
    candidates = sorted(plan.export_dir.rglob("*.ply"))
    if not candidates:
        raise RuntimeError(f"Nerfstudio did not export a PLY under {plan.export_dir}")
    preferred = "splat" if plan.method == "splatfacto" else "mesh"
    primary = next((path for path in candidates if preferred in path.stem.lower()), candidates[0])
    report = root / "neural_report.json"
    write_json(
        report,
        {
            **plan.to_dict(),
            "config": str(config),
            "primary_artifact": str(primary),
            "processing_time_s": round(time.perf_counter() - started, 2),
        },
    )
    return NeuralArtifacts(
        method=plan.method,
        config=config,
        primary_artifact=primary,
        export_kind=plan.export_kind,
        processing_time_s=round(time.perf_counter() - started, 2),
        report=report,
    )


def write_neural_plan(path: str | Path, plan: NeuralPlan) -> Path:
    """Persist a plan for reproducible GPU execution or interview artifacts."""
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(json.dumps(plan.to_dict(), indent=2) + "\n", encoding="utf-8")
    return target
