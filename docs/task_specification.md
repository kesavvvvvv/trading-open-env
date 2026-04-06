# AITEA Task Specification

## Overview

AITEA contains six tasks that represent progressively harder institutional trading objectives. The tasks are designed to resemble realistic workflows used by execution traders, portfolio managers, and risk teams.

Each task has:
- a clear objective
- a bounded episode horizon
- task-specific market conditions
- a deterministic grading function
- a normalized score in `[0.0, 1.0]`

The tasks are intentionally ordered from easier to harder to create a meaningful difficulty curve.

---

## 1. `execution_easy`

### Objective
Execute a target order efficiently in a relatively stable market.

### Real-world analogue
A trader wants to buy a specific quantity of a stock with minimal slippage and cost.

### Observation needs
The agent should pay attention to:
- current price
- bid/ask spread
- liquidity
- current holdings
- cash balance
- remaining target quantity
- recent fills and reward trend

### Allowed actions
- market orders
- small order slices
- hold
- cancel/modify if needed

### Success criteria
The task is considered successful when the target quantity is almost fully executed while:
- slippage remains low
- drawdown stays small
- execution is reasonably smooth

### Difficulty
Easy

---

## 2. `liquidity_medium`

### Objective
Execute or rebalance under low-liquidity conditions.

### Real-world analogue
A desk must trade in a thinner market where impact matters more.

### Observation needs
The agent should use:
- spread width
- liquidity indicators
- partial fill information
- current exposure
- turnover
- recent market drag

### Allowed actions
- small orders
- gradual execution
- conservative risk reduction
- hold

### Success criteria
The task is successful when the agent reduces the target error meaningfully without causing excessive cost or churn.

### Difficulty
Medium

---

## 3. `fx_hedge_medium`

### Objective
Reduce FX exposure efficiently.

### Real-world analogue
A portfolio manager needs to hedge currency risk without paying excessive execution cost.

### Observation needs
The agent should monitor:
- FX exposure proxy
- cash and equity
- hedge instrument availability
- slippage and transaction cost
- drawdown trend

### Allowed actions
- hedge orders
- small adjustments
- hold
- risk reduction

### Success criteria
The hedge exposure should shrink substantially while cost and volatility remain controlled.

### Difficulty
Medium

---

## 4. `rebalance_hard`

### Objective
Rebalance a multi-asset portfolio under strict constraints.

### Real-world analogue
A portfolio manager must align holdings to target weights while controlling turnover, execution cost, and risk.

### Observation needs
The agent should examine:
- target weights
- current weights
- price movement
- cash balance
- exposure levels
- risk state
- turnover history

### Allowed actions
- target-weight rebalance
- portfolio slicing
- selective buying and selling
- hold
- cancel/modify

### Success criteria
The agent should minimize tracking error while keeping drawdown, violations, and turnover under control.

### Difficulty
Hard

---

## 5. `news_adapt_hard`

### Objective
Adapt strategy during structured news shocks.

### Real-world analogue
A trading desk reacts to macro or earnings news that changes volatility and spreads suddenly.

### Observation needs
The agent should use:
- news signal strength
- sentiment
- spread widening
- volatility change
- drawdown
- exposure concentration

### Allowed actions
- reduce risk
- flatten positions
- cautious execution
- hold
- selective hedging

### Success criteria
The agent should avoid deep drawdowns and recover after shocks while maintaining stable performance.

### Difficulty
Hard

---

## 6. `regime_challenge_hard`

### Objective
Operate under hidden regime changes.

### Real-world analogue
A hedge fund must trade when the market shifts between calm, volatile, crisis, and recovery states without the regime being directly observable.

### Observation needs
The agent should infer:
- latent regime clues
- hidden volatility shifts
- realized drawdown
- price instability
- liquidity pressure
- reward stability

### Allowed actions
- defensive reallocation
- cautious orders
- risk reduction
- flattening
- hold

### Success criteria
The agent must stay robust across regime changes and maintain good return quality without large drawdowns.

### Difficulty
Hardest

---

## Difficulty Progression

The tasks are arranged to test different capabilities:

- `execution_easy`: basic execution skill
- `liquidity_medium`: cost-aware trading under limited liquidity
- `fx_hedge_medium`: structured hedging behavior
- `rebalance_hard`: portfolio optimization under constraints
- `news_adapt_hard`: shock response
- `regime_challenge_hard`: hidden-state robustness

This progression ensures the benchmark does not reward only simple overtrading or static policies.

---

## Task design notes

Each task is deterministic under the same seed and configuration. That matters because a grader should measure policy quality, not randomness.

Each task also maps cleanly to a dedicated grader:
- execution task → execution grader
- liquidity task → liquidity grader
- hedge task → FX hedge grader
- rebalance task → rebalance grader
- news task → news response grader
- regime task → regime adaptation grader
