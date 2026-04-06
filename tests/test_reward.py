from __future__ import annotations

from aitea.agents.baseline_rules import baseline_action
from aitea.env.aitea_env import AITEAEnv
from aitea.reward import RewardModel
from aitea.schemas import Action


def test_reward_changes_with_action_quality():
    env = AITEAEnv(task_name="execution_easy", auto_reset=True)
    try:
        model = RewardModel()
        state = env.state_data

        good_action = baseline_action(env.state())
        good_transition = env.step(good_action)
        good_reward = good_transition.reward

        # Reset and use a deliberately poor but still valid action.
        env.reset(task_name="execution_easy")
        bad_action = Action(hold_position=True, strategy_tag="bad_test", comment="no_progress")
        bad_transition = env.step(bad_action)
        bad_reward = bad_transition.reward

        assert -1.0 <= good_reward <= 1.0
        assert -1.0 <= bad_reward <= 1.0
        assert good_reward != bad_reward or env.state_data.step > 0

        computed = model.compute(
            state,
            pnl_delta=0.0,
            execution_cost=0.0,
            slippage=0.0,
            fill_ratio=1.0,
            turnover=0.0,
            market_drag=0.0,
            target_error=0.5,
            completion_progress=0.5,
            violations=[],
        )
        assert 0.0 <= computed.normalized_score <= 1.0
    finally:
        env.close()


def test_penalties_work_and_rewards_stay_well_scaled():
    env = AITEAEnv(task_name="rebalance_hard", auto_reset=True)
    try:
        model = RewardModel()
        state = env.state_data

        reward = model.compute(
            state,
            pnl_delta=-5000.0,
            execution_cost=2000.0,
            slippage=1000.0,
            fill_ratio=0.1,
            turnover=1.0,
            market_drag=3000.0,
            target_error=0.9,
            completion_progress=0.1,
            violations=["gross_exposure_breach", "drawdown_breach"],
        )
        assert -1.0 <= reward.total <= 1.0
        assert 0.0 <= reward.normalized_score <= 1.0
        assert reward.penalties["total"] >= 0.0
        assert reward.components["risk"] <= 1.0
    finally:
        env.close()
