from __future__ import annotations

from pathlib import Path

import yaml

from aitea import Action, AITEAEnv, Info, Observation, Reward


ROOT = Path(__file__).resolve().parents[1]


def test_openenv_yaml_exists():
    path = ROOT / "openenv.yaml"
    assert path.exists(), "openenv.yaml must exist"


def test_openenv_yaml_has_basic_metadata():
    path = ROOT / "openenv.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert any(k in data for k in ("name", "env_name", "title"))
    assert any(k in data for k in ("version", "env_version", "spec_version"))
    assert any(k in data for k in ("description", "summary", "about"))


def test_core_models_exist():
    assert Observation is not None
    assert Action is not None
    assert Reward is not None
    assert Info is not None


def test_environment_methods_exist():
    for method in ("reset", "step", "state", "close"):
        assert hasattr(AITEAEnv, method)
        assert callable(getattr(AITEAEnv, method))


def test_environment_can_initialize():
    env = AITEAEnv(task_name="execution_easy", auto_reset=True)
    try:
        assert env.state_data is not None
        transition = env.reset(task_name="execution_easy")
        assert hasattr(transition, "observation")
        assert hasattr(transition, "reward")
        assert hasattr(transition, "done")
        assert hasattr(transition, "info")
    finally:
        env.close()
