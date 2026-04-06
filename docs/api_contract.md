# AITEA API Contract

## Overview

AITEA exposes a clean environment API centered on the standard lifecycle methods:

- `reset()`
- `step(action)`
- `state()`
- `close()`

The same environment is also exposed through an HTTP API for validation and Hugging Face Spaces deployment.

The API is designed to be strict, typed, and deterministic.

---

## Core Python contract

### `reset()`
Resets the environment into a clean episode state.

Returns a transition-like object containing:
- initial observation
- reward of `0.0`
- `done = False`
- diagnostic info

### `step(action)`
Applies a single action to the environment.

Returns:
- next observation
- reward
- done flag
- diagnostic info

### `state()`
Returns the current observation without advancing the environment.

This is useful for:
- inspection
- debugging
- wrapper code
- evaluation scripts

### `close()`
Closes the environment and marks it unavailable for further use.

---

## Observation schema

The observation returned by `state()` and `reset()` is a typed model that includes:
- step index
- timestamp
- task name
- episode id
- market snapshot
- portfolio summary
- risk summary
- recent news
- latent regime signal
- pending orders
- recent actions
- recent rewards
- short history summary
- market status

The observation is meant to be rich enough for a trading agent to reason over the environment state.

---

## Action schema

The action submitted to `step(action)` is a typed model that may include:
- order instructions
- cancel order ids
- rebalance targets
- hedge targets
- risk reduction level
- flatten-all request
- hold-position request
- free-text comment
- strategy tag

The action model is strict and validated. Unknown fields are rejected so that malformed actions do not silently pass.

---

## Reward schema

The reward object includes:
- total reward
- normalized score
- component breakdown
- penalty breakdown
- raw total
- clipped total

This allows the environment to provide both a learning signal and an inspectable explanation.

---

## Info schema

The info object includes:
- execution cost
- slippage
- fill ratio
- turnover
- PnL delta
- drawdown
- gross exposure
- net exposure
- constraint violations
- task metrics

This is a diagnostic output. It is not the learning signal itself, but it is important for debugging and automated grading.

---

## HTTP endpoints

The API layer exposes the following routes:

### `GET /health`
Returns a basic health check.

### `POST /reset`
Resets the environment. The payload may include:
- `task_name`
- `episode_id`

### `POST /step`
Advances the environment one step.

Payload:
- `action`

### `GET /state`
Returns the current state snapshot as JSON.

These endpoints allow external validators and benchmark runners to interact with the environment remotely.

---

## Expected response behavior

### `/health`
Should always return HTTP 200 when the service is up.

### `/reset`
Should initialize a fresh episode and return a valid observation and diagnostic structure.

### `/step`
Should apply the action, update the market state, and return the new transition.

### `/state`
Should return the current observation without changing the environment state.

---

## Error handling

If the API receives an invalid action or malformed payload, it should fail clearly rather than silently corrupting the environment state.

The middleware layer should:
- log requests
- normalize exceptions
- return readable error responses
- keep responses stable for automated checks

---

## JSON expectations

All payloads should be JSON serializable.

That means:
- no raw Python objects
- no unstructured return values
- no hidden state leaks

The validator depends on consistent JSON output.

---

## Contract stability

The API contract is intentionally narrow and stable. That makes it easier to:
- run automated tests
- mount the environment in HF Spaces
- reproduce results
- compare agents fairly

The contract should not be changed casually once the benchmark is published.
