"""OpenAI-based helper agent for AITEA."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from openai import OpenAI

from ..schemas import Action, Observation
from .action_parser import parse_action


SYSTEM_PROMPT = """
You are a trading decision assistant for an institutional market simulation.

Return ONLY a JSON object matching this schema:

{
  "orders": [
    {
      "symbol": "AAPL",
      "side": "buy|sell|hold|cancel|modify",
      "quantity": 10,
      "order_type": "market|limit|twap|vwap|passive",
      "limit_price": 0.0,
      "urgency": 0.5,
      "time_in_force": "day",
      "tag": "optional"
    }
  ],
  "cancel_order_ids": [],
  "rebalance_targets": {"AAPL": 0.2},
  "hedge_targets": {"MSFT": -1000.0},
  "risk_reduction": 0.0,
  "flatten_all": false,
  "hold_position": false,
  "comment": "optional",
  "strategy_tag": "optional"
}

Rules:
- Output valid JSON only.
- Keep actions conservative.
- Prefer small, realistic steps.
- Do not include markdown fences.
"""


class LLMAgent:
    """Small wrapper around the OpenAI client for environment interaction."""

    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        temperature: float = 0.2,
        max_tokens: int = 400,
        retries: int = 2,
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = max(0, int(retries))

    def _serialize_observation(self, observation: Observation) -> str:
        try:
            if hasattr(observation, "model_dump"):
                payload = observation.model_dump()
            else:
                payload = observation.dict()
        except Exception:
            payload = {"error": "failed_to_serialize_observation"}
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def _build_messages(self, observation: Observation, previous_actions: Optional[List[str]] = None) -> List[Dict[str, str]]:
        obs_text = self._serialize_observation(observation)
        history_text = json.dumps(previous_actions or [], ensure_ascii=False)
        user_prompt = (
            f"Observation JSON:\n{obs_text}\n\n"
            f"Previous actions:\n{history_text}\n\n"
            "Produce the next best action as JSON."
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": user_prompt},
        ]

    def act(self, observation: Observation, previous_actions: Optional[List[str]] = None) -> Action:
        """
        Produce a candidate Action from an observation.

        If the model fails, fall back to a safe hold action.
        """
        fallback = Action(
            hold_position=True,
            strategy_tag="llm_fallback",
            comment="fallback_safe_action",
        )

        messages = self._build_messages(observation, previous_actions=previous_actions)

        last_text = ""
        for _ in range(self.retries + 1):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=False,
                )
                last_text = (completion.choices[0].message.content or "").strip()
                action = parse_action(last_text, fallback=fallback)
                return action
            except Exception as exc:
                last_text = str(exc)
                continue

        return fallback
