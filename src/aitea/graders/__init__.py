"""Grader exports and registry for AITEA."""

from .grader_base import GraderBase
from .grader_execution import ExecutionGrader
from .grader_fx_hedge import FXHedgeGrader
from .grader_liquidity import LiquidityGrader
from .grader_news_response import NewsResponseGrader
from .grader_rebalance import RebalanceGrader
from .grader_regime_adaptation import RegimeAdaptationGrader

GRADER_REGISTRY = {
    "execution_easy": ExecutionGrader,
    "liquidity_medium": LiquidityGrader,
    "fx_hedge_medium": FXHedgeGrader,
    "rebalance_hard": RebalanceGrader,
    "news_adapt_hard": NewsResponseGrader,
    "regime_challenge_hard": RegimeAdaptationGrader,
}


def get_grader_class(name: str):
    return GRADER_REGISTRY[name]


def create_grader(name: str):
    return get_grader_class(name)()


__all__ = [
    "GraderBase",
    "ExecutionGrader",
    "LiquidityGrader",
    "FXHedgeGrader",
    "RebalanceGrader",
    "NewsResponseGrader",
    "RegimeAdaptationGrader",
    "GRADER_REGISTRY",
    "get_grader_class",
    "create_grader",
]
