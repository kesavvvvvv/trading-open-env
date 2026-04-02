"""
State manager for AITEA.

Handles internal state snapshots and conversion to typed observations.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

from aitea.schemas.common import (
    ComplianceStatus,
    MarketRegime,
    MarketSnapshot,
    NewsEvent,
    PendingOrder,
    PortfolioSnapshot,
    RiskLevel,
    RiskSnapshot,
    TreasurySnapshot,
)
from aitea.schemas.observation import Observation


class StateManager:
    def snapshot(self, env: Any) -> Dict[str, Any]:
        """
        Return a safe deep copy of the internal state.
        """
        return deepcopy(getattr(env, "state_data", {}))

    def build_observation(self, env: Any) -> Observation:
        """
        Convert internal state into the typed Observation schema.
        """
        state = getattr(env, "state_data", {})

        market = state.get("market", {})
        portfolio = state.get("portfolio", {})
        treasury = state.get("treasury", {})
        risk = state.get("risk", {})
        news = state.get("news", [])
        pending_orders = state.get("pending_orders", [])

        obs = Observation(
            timestamp=int(state.get("timestamp", 0)),
            market=MarketSnapshot(
                prices=dict(market.get("prices", {})),
                volatility=float(market.get("volatility", 0.0)),
                spread=float(market.get("spread", 0.0)),
                regime_hint=self._coerce_regime(market.get("regime_hint")),
                liquidity=dict(market.get("liquidity", {})),
            ),
            portfolio=PortfolioSnapshot(
                cash=float(portfolio.get("cash", 0.0)),
                positions=dict(portfolio.get("positions", {})),
                unrealized_pnl=float(portfolio.get("unrealized_pnl", 0.0)),
                realized_pnl=float(portfolio.get("realized_pnl", 0.0)),
                total_value=float(portfolio.get("total_value", 0.0)),
            ),
            treasury=TreasurySnapshot(
                upcoming_outflows=dict(treasury.get("upcoming_outflows", {})),
                upcoming_inflows=dict(treasury.get("upcoming_inflows", {})),
                minimum_cash_required=float(treasury.get("minimum_cash_required", 0.0)),
                current_liquidity_buffer=float(treasury.get("current_liquidity_buffer", 0.0)),
            ),
            risk=RiskSnapshot(
                drawdown=float(risk.get("drawdown", 0.0)),
                exposure=float(risk.get("exposure", 0.0)),
                risk_level=self._coerce_risk_level(risk.get("risk_level")),
                compliance_status=self._coerce_compliance_status(risk.get("compliance_status")),
            ),
            news=[self._coerce_news_event(item) for item in news],
            pending_orders=[self._coerce_pending_order(item) for item in pending_orders],
            recent_history_summary=dict(state.get("recent_history_summary", {})),
            task_context=dict(state.get("task_context", {})) if state.get("task_context") is not None else None,
        )

        return obs

    def update_recent_history(self, env: Any, entry: Dict[str, Any], max_items: int = 20) -> None:
        history: List[Dict[str, Any]] = env.state_data.setdefault("recent_history", [])
        history.append(deepcopy(entry))
        if len(history) > max_items:
            del history[:-max_items]

    def rebuild_history_summary(self, env: Any) -> Dict[str, float]:
        history: List[Dict[str, Any]] = env.state_data.get("recent_history", [])
        if not history:
            summary = {"mean_reward": 0.0, "mean_slippage": 0.0, "mean_pnl_delta": 0.0}
            env.state_data["recent_history_summary"] = summary
            return summary

        mean_reward = sum(float(h.get("reward", 0.0)) for h in history) / len(history)
        mean_slippage = sum(float(h.get("slippage", 0.0)) for h in history) / len(history)
        mean_pnl_delta = sum(float(h.get("pnl_delta", 0.0)) for h in history) / len(history)

        summary = {
            "mean_reward": mean_reward,
            "mean_slippage": mean_slippage,
            "mean_pnl_delta": mean_pnl_delta,
        }
        env.state_data["recent_history_summary"] = summary
        return summary

    @staticmethod
    def _coerce_regime(value: Any) -> MarketRegime | None:
        if value is None:
            return None
        if isinstance(value, MarketRegime):
            return value
        try:
            return MarketRegime(str(value))
        except Exception:
            return None

    @staticmethod
    def _coerce_risk_level(value: Any) -> RiskLevel:
        if isinstance(value, RiskLevel):
            return value
        try:
            return RiskLevel(str(value))
        except Exception:
            return RiskLevel.LOW

    @staticmethod
    def _coerce_compliance_status(value: Any) -> ComplianceStatus:
        if isinstance(value, ComplianceStatus):
            return value
        try:
            return ComplianceStatus(str(value))
        except Exception:
            return ComplianceStatus.OK

    @staticmethod
    def _coerce_news_event(item: Any) -> NewsEvent:
        if isinstance(item, NewsEvent):
            return item
        if isinstance(item, dict):
            if hasattr(NewsEvent, "model_validate"):
                return NewsEvent.model_validate(item)
            return NewsEvent.parse_obj(item)
        return NewsEvent(headline=str(item), sentiment=0.0, severity=0.0, affected_assets=[])

    @staticmethod
    def _coerce_pending_order(item: Any) -> PendingOrder:
        if isinstance(item, PendingOrder):
            return item
        if isinstance(item, dict):
            if hasattr(PendingOrder, "model_validate"):
                return PendingOrder.model_validate(item)
            return PendingOrder.parse_obj(item)
        return PendingOrder(
            order_id="unknown",
            instrument="UNKNOWN",
            side="buy",
            order_type="market",
            quantity=0.0,
        )
