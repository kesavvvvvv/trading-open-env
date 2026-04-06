"""Parse model output into a valid AITEA Action."""

from __future__ import annotations

import json
import re
from typing import Any, Dict

from ..schemas import Action, OrderInstruction, OrderSide, OrderType


def _extract_json_block(text: str) -> str:
    """
    Extract a JSON object from raw model output.

    Handles code fences and surrounding commentary.
    """
    stripped = text.strip()

    # Common fenced JSON patterns
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()

    # First balanced-looking object
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]

    return stripped


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _normalize_order(raw: Dict[str, Any]) -> Dict[str, Any]:
    symbol = str(raw.get("symbol", "AAPL"))
    side = str(raw.get("side", "hold")).lower()
    order_type = str(raw.get("order_type", "market")).lower()

    if side not in {"buy", "sell", "hold", "cancel", "modify"}:
        side = "hold"
    if order_type not in {"market", "limit", "twap", "vwap", "passive"}:
        order_type = "market"

    qty = max(1, _safe_int(raw.get("quantity", 1), 1))
    limit_price = raw.get("limit_price", None)
    if limit_price is not None:
        limit_price = max(0.0, _safe_float(limit_price, 0.0))

    payload: Dict[str, Any] = {
        "symbol": symbol,
        "side": side,
        "quantity": qty,
        "order_type": order_type,
        "urgency": max(0.0, min(1.0, _safe_float(raw.get("urgency", 0.5), 0.5))),
        "time_in_force": str(raw.get("time_in_force", "day")),
        "tag": raw.get("tag", None),
    }
    if limit_price is not None:
        payload["limit_price"] = limit_price
    return payload


def parse_action(text: str, fallback: Action | None = None) -> Action:
    """
    Parse raw LLM output into a validated Action.

    If parsing fails, returns the provided fallback or a safe hold action.
    """
    default_action = fallback or Action(
        hold_position=True,
        strategy_tag="fallback_safe",
        comment="parser_fallback",
    )

    if not text or not text.strip():
        return default_action

    try:
        json_text = _extract_json_block(text)
        payload = json.loads(json_text)

        if not isinstance(payload, dict):
            return default_action

        orders = payload.get("orders", [])
        normalized_orders = []
        if isinstance(orders, list):
            for item in orders:
                if isinstance(item, dict):
                    normalized_orders.append(_normalize_order(item))

        rebalance_targets = payload.get("rebalance_targets", {})
        if not isinstance(rebalance_targets, dict):
            rebalance_targets = {}

        hedge_targets = payload.get("hedge_targets", {})
        if not isinstance(hedge_targets, dict):
            hedge_targets = {}

        cancel_order_ids = payload.get("cancel_order_ids", [])
        if not isinstance(cancel_order_ids, list):
            cancel_order_ids = []

        action_payload: Dict[str, Any] = {
            "orders": normalized_orders,
            "cancel_order_ids": [str(x) for x in cancel_order_ids if x is not None],
            "rebalance_targets": {str(k): max(0.0, min(1.0, _safe_float(v, 0.0))) for k, v in rebalance_targets.items()},
            "hedge_targets": {str(k): _safe_float(v, 0.0) for k, v in hedge_targets.items()},
            "risk_reduction": max(0.0, min(1.0, _safe_float(payload.get("risk_reduction", 0.0), 0.0))),
            "flatten_all": bool(payload.get("flatten_all", False)),
            "hold_position": bool(payload.get("hold_position", False)),
            "comment": payload.get("comment", None),
            "strategy_tag": payload.get("strategy_tag", None),
        }

        validator = getattr(Action, "model_validate", None)
        if callable(validator):
            return validator(action_payload)
        return Action.parse_obj(action_payload)

    except Exception:
        return default_action
