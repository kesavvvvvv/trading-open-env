import os
import json
import requests
from typing import List, Optional
from openai import OpenAI

# =========================
# CONFIG
# =========================

# Environment (your local / HF Space)
ENV_BASE_URL = os.getenv("ENV_URL")

# LLM Proxy (MANDATORY for evaluator)
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]

MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

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
        f"[STEP] step={step} action={action} reward={reward} done={str(done).lower()} error={err}",
        flush=True
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True
    )
def warmup_llm():
    try:
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=1,
        )
    except Exception as e:
        print("Warmup failed:", e, flush=True)

# =========================
# ENVIRONMENT API
# =========================
def reset_env(task: str):
    res = requests.post(
        f"{ENV_BASE_URL}/reset",
        json={"task_name": task},
        timeout=REQUEST_TIMEOUT
    )
    res.raise_for_status()
    return res.json()


def step_env(action: dict):
    try:
        res = requests.post(
            f"{ENV_BASE_URL}/step",
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
# PROMPTS
# =========================
def build_system_prompt(task_name: str) -> str:
    return f"""
You are a professional quantitative trading agent operating in a simulated institutional execution environment.

TASK: {task_name}

=====================
OBJECTIVE
=====================
Maximize cumulative reward by:
- Executing profitable trades
- Minimizing unnecessary actions
- Managing position intelligently

=====================
ENVIRONMENT UNDERSTANDING
=====================
Each step provides:
- Market data (prices, signals, etc.)
- Portfolio state (positions, cash)

You must:
- Decide whether to BUY, SELL, or HOLD
- Maintain sensible position sizing
- Avoid invalid or empty actions

=====================
DECISION LOGIC
=====================
Follow this strictly:

1. If you have NO position:
   → Open a SMALL BUY position(something like 100-150 shares) in a promising stock

2. If you HOLD position:
   → If price trend positive → HOLD or SELL partially
   → If price drops → consider SELL

3. NEVER:
   - Return empty orders
   - Use invalid symbols
   - Use zero or negative quantity

=====================
CONSTRAINTS
=====================
- Allowed symbols: AAPL, MSFT, GOOGL. Frequently try trading different symbols.
- Quantity: integer between 1 and 1000
- Always include at least ONE order

=====================
OUTPUT FORMAT (STRICT JSON ONLY)
=====================
Return ONLY this structure:

{{
  "orders": [
    {{
      "symbol": "AAPL",
      "side": "buy",
      "quantity": 10
    }}
  ]
}}

=====================
IMPORTANT
=====================
- NO explanation
- NO text
- ONLY valid JSON
"""


def build_user_prompt(observation: dict) -> str:
    portfolio = observation.get("portfolio", {})
    positions = portfolio.get("positions", [])
    cash = portfolio.get("cash", "unknown")

    market_summary = {
        "has_position": len(positions) > 0,
        "positions": positions,
        "cash": cash
    }

    return f"""
CURRENT STATE:

Portfolio:
{json.dumps(market_summary, indent=2)}

Raw Observation:
{json.dumps(observation, indent=2)}

TASK:
Decide the next BEST trading action.

REMEMBER:
- You MUST take action (no empty orders)
- Be realistic and consistent
- Follow the strategy rules

Return ONLY JSON.
"""


# =========================
# SAFE PARSER
# =========================
def parse_action(text: str) -> dict:
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        parsed = json.loads(text[start:end])
        return validate_action(parsed)
    except Exception:
        return fallback_action()


def validate_action(action: dict) -> dict:
    if not isinstance(action, dict):
        return fallback_action()

    action.setdefault("orders", [])

    valid_orders = []
    for o in action["orders"]:
        if (
            isinstance(o, dict)
            and "symbol" in o
            and o.get("side") in ["buy", "sell"]
            and isinstance(o.get("quantity"), (int, float))
        ):
            qty = int(max(1, o["quantity"]))
            valid_orders.append({
                "symbol": str(o["symbol"]),
                "side": o["side"],
                "quantity": qty,
                "order_type": "market" 
            })

    if not valid_orders:
        return fallback_action()

    action["orders"] = valid_orders
    return action


def fallback_action():
    return {
    "orders": [
        {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 1,
            "order_type": "market"
        }
    ]
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

        content = response.choices[0].message.content or ""
        return parse_action(content)

    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", flush=True)
        return fallback_action()


# =========================
# MAIN LOOP
# =========================
def run():
    warmup_llm()   
    rewards: List[float] = []
    steps = 0
    score = 0
    success = False

    log_start(TASK_NAME, BENCHMARK, MODEL_NAME)

    try:
        result = reset_env(TASK_NAME)
        transition = result.get("transition", {})
        observation = transition.get("observation", {})
        done = transition.get("done", False)

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action = get_action(observation, TASK_NAME)
            action_str = json.dumps(action, separators=(",", ":"))

            result = step_env(action)

            transition = result.get("transition", {})

            observation = transition.get("observation", {})
            reward = float(transition.get("reward", 0.0))
            done = transition.get("done", False)
            error = transition.get("error")

            rewards.append(reward)
            steps = step

            log_step(step, action_str, reward, done, error)

            if done:
                break

        if steps > 0:
            score = sum(rewards) / steps

        score = max(0.01, min(0.99, score))
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        log_step(0, "{}", 0, True, str(e))

    finally:
        log_end(success, steps, score, rewards)


# =========================
# ENTRY
# =========================
if __name__ == "__main__":
    run()