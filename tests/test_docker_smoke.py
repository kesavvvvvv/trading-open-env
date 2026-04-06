from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_exists():
    assert (ROOT / "Dockerfile").exists(), "Dockerfile must exist"


def test_docker_build_smoke():
    if shutil.which("docker") is None:
        pytest.skip("Docker is not installed in this environment")

    image_tag = "aitea-test-smoke:latest"
    build = subprocess.run(
        ["docker", "build", "-t", image_tag, "."],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=900,
    )
    if build.returncode != 0:
        pytest.fail(f"Docker build failed:\nSTDOUT:\n{build.stdout}\nSTDERR:\n{build.stderr}")

    run = subprocess.run(
        ["docker", "run", "--rm", image_tag, "python", "-c", "print('ok')"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if run.returncode != 0:
        pytest.fail(f"Docker run failed:\nSTDOUT:\n{run.stdout}\nSTDERR:\n{run.stderr}")

    assert "ok" in run.stdout
