from __future__ import annotations

from pathlib import Path

import pytest

from reconbot.neural_reconstruction import build_nerfstudio_plan, write_neural_plan


@pytest.mark.parametrize("method", ["nerfacto", "instant-ngp"])
def test_mesh_methods_reuse_colmap_and_export_poisson(tmp_path: Path, method: str):
    plan = build_nerfstudio_plan(
        tmp_path / "images",
        tmp_path / "sparse" / "0",
        tmp_path / "neural",
        method=method,
        max_iterations=2500,
    )

    process, train, export = plan.stages
    assert process.command[:2] == ("ns-process-data", "images")
    assert "--skip-colmap" in process.command
    assert process.command[process.command.index("--colmap-model-path") + 1].endswith(
        str(Path("sparse") / "0")
    )
    assert train.command[:2] == ("ns-train", method)
    assert train.command[train.command.index("--max-num-iterations") + 1] == "2500"
    assert export.command[:2] == ("ns-export", "poisson")
    assert plan.export_kind == "poisson-mesh"


def test_splatfacto_exports_gaussian_splat(tmp_path: Path):
    plan = build_nerfstudio_plan(
        tmp_path / "images",
        tmp_path / "sparse",
        tmp_path / "neural",
        method="splatfacto",
    )

    assert plan.export_kind == "gaussian-splat"
    assert plan.stages[-1].command[:2] == ("ns-export", "gaussian-splat")
    assert "--load-config" in plan.stages[-1].command

    output = write_neural_plan(tmp_path / "plan.json", plan)
    assert '"method": "splatfacto"' in output.read_text(encoding="utf-8")


def test_neural_plan_rejects_invalid_configuration(tmp_path: Path):
    with pytest.raises(ValueError, match="Unsupported neural method"):
        build_nerfstudio_plan(tmp_path, tmp_path, tmp_path, method="unknown")
    with pytest.raises(ValueError, match="positive"):
        build_nerfstudio_plan(tmp_path, tmp_path, tmp_path, max_iterations=0)
