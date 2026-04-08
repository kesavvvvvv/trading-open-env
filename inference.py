import os
import json
import time
import asyncio
import requests
from typing import List, Optional

from openai import OpenAI

# =========================
# CONFIG
# =========================
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN", "")
IMAGE_NAME = os.getenv("IMAGE_NAME")  # <-- IMPORTANT (docker mode)

TASK_NAME = os.getenv("AITEA_TASK", "execution_easy")
BENCHMARK = "aitea"
MAX_STEPS = 50
SUCCESS_SCORE_THRESHOLD = 0.6

client = OpenAI(api_key=HF_TOKEN if HF_TOKEN else "dummy-key")


# =========================
# LOGGING (STRICT FORMAT)
# =========================
def log_start():
    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def log_step(step: int, action: dict, reward: float, done: bool, error: Optional[str]):
    action_str = json.dumps(action, separators=(",", ":"))
    error_val = error if error else "null"
    done_val = str(done).lower()

    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# =========================
# SAFE BASELINE POLICY
# =========================
def fallback_policy(observation):
    portfolio = observation.get("portfolio", {})
    positions = portfolio.get("positions", [])

    total_position = sum(abs(p.get("quantity", 0)) for p in positions)

    if total_position == 0:
        return {
            "orders": [{
                "symbol": "AAPL",
                "side": "buy",
                "quantity": 5,
                "order_type": "market"
            }]
        }

    return {
        "orders": [{
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 1,
            "order_type": "market"
        }]
    }


# =========================
# LLM POLICY
# =========================
SYSTEM_PROMPT = """
You are a trading agent in a simulated financial market.

Goal:
- Maximize reward
- Minimize unnecessary trading

Rules:
- Return ONLY valid JSON
- Keep trades small and realistic
- Avoid extreme actions
"""


def get_action(observation):
    try:
        if not HF_TOKEN:
            return fallback_policy(observation)

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(observation)},
            ],
            temperature=0.1,
            max_tokens=150,
        )

        content = response.choices[0].message.content.strip()

        try:
            return json.loads(content)
        except Exception:
            return fallback_policy(observation)

    except Exception:
        return fallback_policy(observation)


# =========================
# HTTP MODE (HF SPACE)
# =========================
def post(endpoint, payload):
    return requests.post(
        f"{API_BASE_URL}{endpoint}",
        json=payload,
        timeout=10
    ).json()


def run_http_episode():
    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start()

    try:
        reset = post("/reset", {})
        observation = reset.get("transition", {}).get("observation", {})

        for step in range(1, MAX_STEPS + 1):
            action = get_action(observation)

            res = post("/step", action)
            transition = res.get("transition", {})

            observation = transition.get("observation", {})
            reward = float(transition.get("reward", 0.0))
            done = bool(transition.get("done", False))
            error = transition.get("error", None)

            rewards.append(reward)
            steps_taken = step

            log_step(step, action, reward, done, error)

            if done:
                break

            time.sleep(0.05)

        if rewards:
            score = sum(rewards) / len(rewards)

        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success, steps_taken, score, rewards)


# =========================
# DOCKER MODE (LOCAL VALIDATION)
# =========================
async def run_docker_episode():
    from openenv import EnvClient  # REQUIRED for docker mode

    env = await EnvClient.from_docker_image(IMAGE_NAME)

    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start()

    try:
        result = await env.reset()
        observation = result.observation.dict()

        for step in range(1, MAX_STEPS + 1):
            action = get_action(observation)

            result = await env.step(action)

            observation = result.observation.dict()
            reward = float(result.reward or 0.0)
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step

            log_step(step, action, reward, done, error)

            if done:
                break

        if rewards:
            score = sum(rewards) / len(rewards)

        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception:
            pass

        log_end(success, steps_taken, score, rewards)


# =========================
# ENTRYPOINT
# =========================
if __name__ == "__main__":
    if IMAGE_NAME:
        asyncio.run(run_docker_episode())  # LOCAL docker mode
    else:
        run_http_episode()  # HF Space mode