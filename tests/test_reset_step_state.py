from __future__ import annotations

from aitea.agents.baseline_rules import baseline_action
from aitea.env.aitea_env import AITEAEnv
from aitea.schemas import Observation


def _dump(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return obj


def test_reset_returns_valid_transition_and_observation():
    env = AITEAEnv(task_name="execution_easy", auto_reset=False)
    try:
        transition = env.reset(task_name="execution_easy")
        assert transition.done is False
        assert transition.reward == 0.0
        assert transition.observation is not None
        assert isinstance(transition.observation, Observation)
        assert transition.info is not None
        assert env.state_data is not None
        assert env.state_data.step == 0
    finally:
        env.close()


def test_step_returns_correct_structure():
    env = AITEAEnv(task_name="execution_easy", auto_reset=True)
    try:
        obs = env.state()
        action = baseline_action(obs)
        transition = env.step(action)
        assert hasattr(transition, "observation")
        assert hasattr(transition, "reward")
        assert hasattr(transition, "done")
        assert hasattr(transition, "info")
        assert isinstance(transition.reward, float)
        assert -1.0 <= transition.reward <= 1.0
        assert transition.observation is not None
    finally:
        env.close()


def test_state_matches_internal_environment_state():
    env = AITEAEnv(task_name="rebalance_hard", auto_reset=True)
    try:
        obs1 = env.state()
        assert isinstance(obs1, Observation)
        action = baseline_action(obs1)
        transition = env.step(action)
        obs2 = env.state()

        assert obs2.step == env.state_data.step
        assert obs2.task_name == env.task_name
        assert _dump(obs2)["step"] == _dump(transition.observation)["step"]
    finally:
        env.close()


def test_episode_progression_is_sane():
    env = AITEAEnv(task_name="liquidity_medium", auto_reset=True)
    try:
        start_step = env.state().step
        for _ in range(3):
            action = baseline_action(env.state())
            transition = env.step(action)
        assert env.state().step >= start_step + 1
        assert isinstance(transition.done, bool)
    finally:
        env.close()
