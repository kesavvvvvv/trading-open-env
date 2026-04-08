import os
import json
import time
import requests
from typing import List, Optional
from openai import OpenAI

# =========================
# CONFIG
# =========================
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = HF_TOKEN or os.getenv("API_KEY")

if API_KEY is None:
    raise ValueError("API_KEY or HF_TOKEN must be set")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

TASK_NAME = os.getenv("TASK_NAME", "execution_easy")
BENCHMARK = "aitea_trading_env"

MAX_STEPS = 50
REQUEST_TIMEOUT = 10
SUCCESS_THRESHOLD = 0.3



# =========================
# LOGGING (STRICT FORMAT)
# =========================
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}",
        flush=True
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True
    )

# =========================
# ENV API
# =========================
def reset_env(task: str):
    try:
        res = requests.post(
            f"{API_BASE_URL}/reset",
            json={"task_name": task},
            timeout=REQUEST_TIMEOUT
        )
        res.raise_for_status()
        return res.json()
    except Exception as e:
        raise RuntimeError(f"reset failed: {e}")

def step_env(action: dict):
    try:
        res = requests.post(
            f"{API_BASE_URL}/step",
            json=action,
            timeout=REQUEST_TIMEOUT
        )
        res.raise_for_status()
        return res.json()
    except Exception as e:
        return {
            "observation": {},
            "reward": 0.0,
            "done": True,
            "error": str(e)
        }

# =========================
# PROMPT ENGINE (SMART)
# =========================
def build_system_prompt(task_name: str):
    return f"""
You are an elite institutional trading agent.

TASK: {task_name}

OBJECTIVES:
- Maximize profit (PnL)
- Minimize slippage
- Maintain risk discipline

STRATEGY:

Execution:
- Split trades into smaller orders
- Avoid aggressive large trades

Liquidity:
- Respect market depth
- Avoid high-impact trades

Rebalance:
- Move toward target allocation
- Avoid overtrading

RULES:
- If unsure → HOLD (no orders)
- Keep actions realistic
- Output STRICT JSON only

FORMAT:
{{
  "orders": [{{"symbol":"AAPL","side":"buy","quantity":10}}],
  "rebalance": {{}},
  "hedge": {{}}
}}
"""

def build_user_prompt(observation: dict):
    return f"""
Market State:
{json.dumps(observation, indent=2)}

Choose best action.
"""

# =========================
# SAFE ACTION PARSER
# =========================
def safe_parse_action(text: str) -> dict:
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        text = text[start:end]
        action = json.loads(text)
        return validate_action(action)
    except:
        return fallback_action()

def validate_action(action: dict) -> dict:
    if not isinstance(action, dict):
        return fallback_action()

    action.setdefault("orders", [])
    action.setdefault("rebalance", {})
    action.setdefault("hedge", {})

    safe_orders = []
    for o in action["orders"]:
        if (
            isinstance(o, dict)
            and "symbol" in o
            and o.get("side") in ["buy", "sell"]
            and isinstance(o.get("quantity"), (int, float))
        ):
            qty = int(max(0, o["quantity"]))
            if qty > 0:
                safe_orders.append({
                    "symbol": str(o["symbol"]),
                    "side": o["side"],
                    "quantity": qty
                })

    action["orders"] = safe_orders
    return action

def fallback_action():
    return {
        "orders": [],
        "rebalance": {},
        "hedge": {}
    }

# =========================
# LLM AGENT
# =========================
def get_action(observation: dict, task_name: str) -> dict:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": build_system_prompt(task_name)},
                {"role": "user", "content": build_user_prompt(observation)},
            ],
            temperature=0.2,
        )

        content = response.choices[0].message.content.strip()
        return safe_parse_action(content)

    except Exception:
        return fallback_action()

# =========================
# MAIN LOOP
# =========================
def run():

    rewards: List[float] = []
    steps = 0
    success = False
    score = 0.0

    log_start(TASK_NAME, BENCHMARK, MODEL_NAME)

    try:
        result = reset_env(TASK_NAME)

        observation = result.get("observation", {})
        done = result.get("done", False)

        for step in range(1, MAX_STEPS + 1):

            if done:
                break

            action_dict = get_action(observation, TASK_NAME)
            action_str = json.dumps(action_dict, separators=(",", ":"))

            result = step_env(action_dict)

            observation = result.get("observation", {})
            reward = float(result.get("reward", 0.0))
            done = result.get("done", False)
            error = result.get("error")

            rewards.append(reward)
            steps = step

            log_step(step, action_str, reward, done, error)

            if done:
                break

        # normalize score
        if steps > 0:
            score = sum(rewards) / steps

        score = max(0.0, min(1.0, score))
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        log_step(0, "{}", 0.0, True, str(e))

    finally:
        log_end(success, steps, score, rewards)

# =========================
# ENTRY
# =========================
if __name__ == "__main__":
    run()