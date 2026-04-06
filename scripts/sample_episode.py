"""Small manual episode runner for debugging AITEA."""

from __future__ import annotations

import json

from aitea.agents.baseline_rules import baseline_action
from aitea.env.aitea_env import AITEAEnv


def _dump(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return obj


def main() -> None:
    env = AITEAEnv(task_name="execution_easy", auto_reset=True)
    try:
        print("RESET")
        obs = env.state()
        print(json.dumps(_dump(obs), indent=2, default=str))

        for step_idx in range(10):
            action = baseline_action(env.state())
            transition = env.step(action)
            print(f"\nSTEP {step_idx}")
            print("reward:", transition.reward)
            print("done:", transition.done)
            print(json.dumps(_dump(transition.info), indent=2, default=str))

            if transition.done:
                break

        print("\nFINAL STATE")
        print(json.dumps(_dump(env.state()), indent=2, default=str))
    finally:
        env.close()


if __name__ == "__main__":
    main()
