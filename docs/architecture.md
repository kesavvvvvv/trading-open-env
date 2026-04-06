# AITEA Architecture

## Overview

AITEA, the **Agentic Institutional Trading & Execution Arena**, is a realistic market simulation environment designed for agent evaluation and training. It models how an institutional desk operates under imperfect information, liquidity constraints, transaction costs, risk limits, hidden market regimes, and news shocks.

The environment is **single-agent at the API level** but **multi-agent internally**. This means the model interacting with the environment submits one action at a time through `step(action)`, while the environment itself simulates competing market participants, liquidity providers, noise traders, and regime changes.

The design goal is to make the environment useful for evaluating decision-making in realistic financial settings without turning it into a toy game.

## High-Level System Components

### 1. Environment Core
The core environment lives in `src/aitea/env/` and exposes the OpenEnv-style lifecycle:

- `reset()`
- `step(action)`
- `state()`
- `close()`

The public environment class is `AITEAEnv`. It coordinates all state transitions, validates actions, and returns typed observation, reward, done, and info objects.

### 2. Typed Schemas
The `src/aitea/schemas/` package defines strict typed data models for:

- observations
- actions
- rewards
- diagnostic info
- shared financial primitives

These models make the environment easy to validate and easier to integrate with automated tools.

### 3. Simulation Engines
The `src/aitea/engines/` package provides market realism:

- `market_engine.py` for price and volatility evolution
- `execution_engine.py` for fills, slippage, and execution cost
- `risk_engine.py` for exposure, drawdown, and constraint checks
- `treasury_engine.py` for cash and funding state
- `news_engine.py` for exogenous news shocks
- `regime_engine.py` for hidden market regimes
- `multi_agent_engine.py` for internal background order flow

These engines are deterministic under a seed, which is important for reproducibility and grading.

### 4. Tasks and Graders
The `src/aitea/tasks/` package defines six tasks with increasing difficulty. Each task configures:

- the scenario
- the market conditions
- the objective
- the success condition

The `src/aitea/graders/` package scores completed trajectories. Graders output normalized values in `[0.0, 1.0]`. They are deterministic and task-specific.

### 5. Reward Model
The `src/aitea/reward/` package computes dense trajectory-level reward. It combines:

- PnL
- execution quality
- liquidity awareness
- risk control
- compliance
- stability

It also applies penalties for invalid, destructive, or repetitive behavior.

### 6. API Layer
The `src/aitea/api/` package exposes HTTP endpoints for validation and HF Space deployment:

- `/health`
- `/reset`
- `/step`
- `/state`

These routes expose the same environment instance so a remote client can interact with the simulation.

### 7. Agent Helpers
The `src/aitea/agents/` package contains support utilities:

- deterministic baseline policy
- OpenAI client wrapper
- action parser and fallback logic

These files are not the benchmark agent itself. They exist to provide a reproducible baseline and a robust LLM interaction wrapper.

### 8. Utilities
The `src/aitea/utils/` package includes:

- math helpers
- serialization helpers
- validation helpers
- logging helpers
- constants

These helpers keep the project maintainable and reduce duplicated logic.

## Data Flow

A single step in the environment works like this:

1. The agent sends an action.
2. The action is validated against the schema.
3. The execution engine simulates fills and trading cost.
4. The market engine updates prices, spreads, and volatility.
5. The news engine may inject a shock.
6. The regime engine may switch the hidden market regime.
7. The multi-agent engine adds background market pressure.
8. The treasury and risk engines update capital and constraint state.
9. The reward model computes dense reward.
10. The grader tracks task success and score quality.
11. The environment returns the next observation, reward, done flag, and info object.

This loop is repeated until the episode ends.

## Design Principles

### Realism
The environment models a genuine institutional trading workflow:
- portfolio decisions
- execution quality
- capital usage
- market impact
- risk control

### Determinism
Every task and grader is reproducible when the seed is fixed.

### Interpretability
Rewards and grader outputs are broken into readable components. This makes debugging and evaluation easier.

### Extensibility
The architecture is modular enough to add new tasks or market regimes later without rewriting the whole system.

## Why this design is useful

AITEA is useful for evaluating agents on a real-world decision problem that requires:
- sequential planning
- risk sensitivity
- cost-aware execution
- response to uncertainty
- robustness under hidden state changes

That makes it more valuable than a toy benchmark and more interpretable than a black-box simulation.
