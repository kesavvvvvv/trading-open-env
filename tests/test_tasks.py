from __future__ import annotations

import importlib

from aitea.registry import get_task, list_tasks
from aitea.tasks import TaskBase, create_task


EXPECTED_TASKS = [
    "execution_easy",
    "liquidity_medium",
    "fx_hedge_medium",
    "rebalance_hard",
    "news_adapt_hard",
    "regime_challenge_hard",
]


def test_tasks_register_and_load():
    # Ensure decorators run and tasks are imported.
    importlib.import_module("aitea.tasks.task_execution_easy")
    importlib.import_module("aitea.tasks.task_liquidity_medium")
    importlib.import_module("aitea.tasks.task_fx_hedge_medium")
    importlib.import_module("aitea.tasks.task_rebalance_hard")
    importlib.import_module("aitea.tasks.task_news_adapt_hard")
    importlib.import_module("aitea.tasks.task_regime_challenge_hard")

    registered = list_tasks()
    for task_name in EXPECTED_TASKS:
        assert task_name in registered
        task_cls = get_task(task_name)
        task = task_cls()
        assert isinstance(task, TaskBase)
        assert task.task_name == task_name
        assert isinstance(task.profile(), dict)
        assert isinstance(task.metadata(), dict)


def test_task_factory_creates_instances():
    for task_name in EXPECTED_TASKS:
        task = create_task(task_name)
        assert isinstance(task, TaskBase)
        assert task.task_name == task_name
        assert task.profile()["kind"] in {"execution", "liquidity", "hedge", "rebalance", "news", "regime"}
