from __future__ import annotations

from aitea.agents.baseline_rules import baseline_action
from aitea.env.aitea_env import AITEAEnv
from aitea.graders import (
    ExecutionGrader,
    FXHedgeGrader,
    GraderBase,
    LiquidityGrader,
    NewsResponseGrader,
    RebalanceGrader,
    RegimeAdaptationGrader,
)


def _make_state(task_name: str):
    env = AITEAEnv(task_name=task_name, auto_reset=True)
    try:
        return env, env.state_data
    except Exception:
        env.close()
        raise


def test_graders_return_scores_in_range_and_are_deterministic():
    graders = [
        ("execution_easy", ExecutionGrader()),
        ("liquidity_medium", LiquidityGrader()),
        ("fx_hedge_medium", FXHedgeGrader()),
        ("rebalance_hard", RebalanceGrader()),
        ("news_adapt_hard", NewsResponseGrader()),
        ("regime_challenge_hard", RegimeAdaptationGrader()),
    ]

    for task_name, grader in graders:
        env, state = _make_state(task_name)
        try:
            assert isinstance(grader, GraderBase)
            s1 = grader.score(state)
            s2 = grader.score(state)
            assert isinstance(s1, float)
            assert 0.0 <= s1 <= 1.0
            assert s1 == s2
        finally:
            env.close()


def test_execution_grader_is_not_constant():
    env = AITEAEnv(task_name="execution_easy", auto_reset=True)
    try:
        grader = ExecutionGrader()
        state1 = env.state_data
        score1 = grader.score(state1)

        action = baseline_action(env.state())
        env.step(action)
        state2 = env.state_data
        score2 = grader.score(state2)

        assert 0.0 <= score1 <= 1.0
        assert 0.0 <= score2 <= 1.0
        assert score1 != score2 or state1.step != state2.step
    finally:
        env.close()
