# AITEA Reward Design

## Overview

The reward system in AITEA is designed to be:
- dense
- interpretable
- reproducible
- task-sensitive
- difficult to exploit

A good reward function is essential because the benchmark is not only checking whether the agent eventually succeeds. It is also checking whether the environment gives meaningful learning signal throughout the episode.

---

## Reward goals

The reward should encourage:
- profitable behavior
- efficient execution
- low slippage
- cost control
- risk discipline
- stable trajectories
- task-specific progress

The reward should discourage:
- random overtrading
- invalid actions
- endless holding when progress is required
- risk limit breaches
- destructive behavior
- repeated harmful patterns

---

## Reward components

The main reward model combines several normalized components.

### 1. PnL reward
This rewards positive incremental profit and penalizes losses. It is normalized against starting capital so the signal remains bounded and comparable across episodes.

### 2. Execution quality
This rewards:
- high fill ratio
- low transaction cost
- low slippage

This matters most in execution and liquidity tasks.

### 3. Liquidity awareness
This rewards small, deliberate trading in thin markets and penalizes excessive turnover.

### 4. Risk control
This rewards keeping drawdown, exposure, and violation counts low.

### 5. Compliance
This rewards using valid actions and respecting the environment rules.

### 6. Stability
This rewards a smooth reward trajectory. A policy that flips between large positive and negative outcomes is less desirable than one with consistent controlled performance.

### 7. Portfolio progress
This rewards closing the gap to the task objective, such as:
- remaining quantity in execution tasks
- tracking error in rebalance tasks
- hedge error in FX tasks

---

## Penalties

The penalty system is separate from the reward components to keep the design interpretable.

### Invalid action penalty
Applied when the agent submits malformed, impossible, or unsafe actions.

### Excessive churn penalty
Applied when the agent trades too frequently without improving task progress.

### No-op abuse penalty
Applied when the agent keeps holding or doing nothing in situations where progress is required.

### Risk breach penalty
Applied for:
- drawdown breaches
- exposure breaches
- risk limit violations

### Destructive action penalty
Applied when actions cause extreme loss of capital or destabilize the portfolio.

### Repeated harmful behavior penalty
Applied when the same bad pattern repeats over several steps.

---

## Final reward formula

The model uses a weighted sum:

`reward = weighted_components - weighted_penalties`

The result is clipped to a bounded range, then normalized to `[0.0, 1.0]` for grading convenience.

This gives:
- step-level learning signal
- clear partial progress
- penalty for bad behavior
- stable numerical range

---

## Why this design is strong

### 1. It is dense
The agent gets feedback at every step, not only at the end.

### 2. It is interpretable
You can inspect component values to understand why a policy did well or poorly.

### 3. It is hard to game
No-op loops, reckless trading, or artificial churn all get penalized.

### 4. It supports different tasks
The same reward framework works across execution, liquidity, hedge, rebalance, news, and regime tasks because the task-specific progress term changes with the task type.

### 5. It is stable
Bounded outputs and explicit clipping reduce the risk of extreme numerical instability.

---

## Practical interpretation

A strong policy should not maximize raw profit alone. It should:
- make the right trade at the right time
- trade with restraint in thin markets
- hedge when exposure grows too large
- rebalance efficiently
- respond quickly to market shocks
- avoid overreacting to noise

That is why the reward function blends return, risk, execution, and compliance together.

---

## Example behavior

A policy can receive a positive reward even if immediate profit is small, as long as it:
- reduces exposure to dangerous levels
- improves tracking error
- lowers slippage
- makes measurable task progress

Likewise, a policy with high profit but severe drawdown, violations, or churn should not score highly. This prevents brittle strategies from dominating the benchmark.
