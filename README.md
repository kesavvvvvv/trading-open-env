---
title: AITEA Trading Environment
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---
﻿# AITEA — Agentic Institutional Trading & Execution Arena

AITEA is a realistic institutional trading simulation environment for evaluating AI agents on portfolio construction, execution quality, liquidity-aware trading, risk management, and adaptation to market shocks. It is designed for the Meta × Scaler OpenEnv hackathon and follows the required OpenEnv-style lifecycle with typed models, `reset()`, `step()`, `state()`, `openenv.yaml`, a Dockerized runtime, and a baseline inference script.

The environment is intentionally **single-agent at the API level** while simulating **multi-agent market behavior internally**. That means the external evaluator or LLM agent interacts with one environment, but the environment itself contains background market participants, hidden regimes, news shocks, liquidity changes, and execution costs. This gives the benchmark realistic market behavior without breaking the OpenEnv interface.

---

## Why this environment exists

Most reinforcement learning and agent benchmarks are either too toy-like or too disconnected from real operational decision-making. AITEA is built to model tasks that real institutional desks actually care about:

- how to execute an order with low slippage
- how to trade under limited liquidity
- how to hedge FX exposure
- how to rebalance a multi-asset book under risk constraints
- how to respond to news shocks
- how to stay stable under regime changes

This makes the environment useful for:
- agent evaluation
- policy learning
- benchmarking LLM-based decision systems
- studying risk-aware sequential decision making
- testing how models behave under market stress

The goal is not to simulate a perfect exchange. The goal is to simulate a **good enough institutional decision environment** that rewards careful, cost-aware, robust behavior.

---



## Core design philosophy

AITEA follows five principles.

### Realism
The environment includes:
- price movement
- spread changes
- execution cost
- partial fills
- liquidity limits
- drawdown control
- hidden regimes
- news shocks
- background market participants

### Determinism
Given the same seed and task configuration, the environment should behave reproducibly. That is important for grading, debugging, and comparing agents fairly.

### Dense learning signal
The reward is not just a final success/failure flag. It gives partial progress at every step.

### Interpretability
Rewards, penalties, and grader outputs are broken into components so that users can understand why the agent performed well or poorly.

### Extensibility
The code is modular:
- schemas define typed data
- engines simulate the market
- tasks define objectives
- graders score success
- reward shapes the learning signal
- API exposes the environment
- utilities keep the system clean

---

## Repository layout

```text
aitea/
├── README.md
├── openenv.yaml
├── inference.py
├── Dockerfile
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── .env.example
├── LICENSE
│
├── src/
│   └── aitea/
│       ├── __init__.py
│       ├── config.py
│       ├── registry.py
│       ├── schemas/
│       ├── env/
│       ├── engines/
│       ├── tasks/
│       ├── graders/
│       ├── reward/
│       ├── agents/
│       ├── utils/
│       └── api/
│
├── tests/
├── scripts/
├── docs/

```
## Environment Overview

AITEA simulates a market where an agent receives structured observations and must choose actions that improve performance under real trading frictions.

This environment is designed to mimic institutional-grade trading systems, where decision-making is sequential, constrained, and influenced by multiple external factors such as liquidity, execution costs, and hidden market dynamics.

### The agent sees:

- asset prices  
- bid/ask spreads  
- liquidity indicators  
- current holdings  
- cash  
- equity  
- unrealized and realized PnL  
- drawdown  
- risk state  
- pending orders  
- recent news  
- latent regime signal  
- recent action/reward history  

These inputs are intentionally structured to allow both LLM-based agents and RL-based policies to reason effectively.

---

### The agent can do:

- buy or sell  
- set target allocations  
- hedge exposure  
- cancel or modify orders  
- flatten positions  
- reduce risk  
- hold position  

This action space reflects real-world trading capabilities available to institutional systems.

---

### The environment does:

- validate the action  
- simulate fills  
- apply slippage and cost  
- update holdings and cash  
- evolve the market  
- inject news or regime shifts  
- simulate background participants  
- compute reward  
- check task completion  
- return the next state  

This loop ensures a realistic **closed decision-feedback cycle**, critical for training intelligent agents.

---

## Observation Space

The observation is a typed structured object, not a plain blob. It is designed so that an LLM or policy model can reason over the market state safely and consistently.

### Observation contents

The exact fields are defined in `src/aitea/schemas/observation.py`, but the observation includes the following categories:

---

### Market snapshot

- symbol-level price  
- bid  
- ask  
- spread  
- last price  
- volume  

---

### Portfolio snapshot

- cash  
- equity  
- gross exposure  
- net exposure  
- leverage  
- per-asset positions  
- position market values  
- realized and unrealized PnL  

---

### Risk snapshot

- drawdown  
- risk level  
- exposure ratios  
- violation count  

---

### Event signals

- recent news  
- regime state  
- regime confidence  
- transition probability  

---

### Execution state

- pending orders  
- recent actions  
- recent rewards  
- history summary  

---

### Metadata

- step number  
- timestamp  
- task name  
- episode id  
- market status  

---

This design gives the agent enough structure to make meaningful choices without exposing internal implementation details.

---

## Action Space

The action is also typed and validated. It is designed to support realistic trading decisions.

### Action contents

The exact schema is in `src/aitea/schemas/action.py`. The action may include:

- a list of order instructions  
- cancel order ids  
- rebalance targets  
- hedge targets  
- risk reduction level  
- flatten-all flag  
- hold-position flag  
- optional comment  
- optional strategy tag  

---

### Supported trading styles

The action space supports:

- direct market orders  
- small execution slices  
- target-weight rebalancing  
- hedge adjustments  
- conservative risk-off behavior  

---

### Design intent

The action space is not meant to be a free-form natural language interface. It is meant to be:

- strict  
- reproducible  
- easy to validate  
- rich enough for real trading decisions  

---

## Reward Design

The reward system is one of the most important parts of this project.

### Why reward matters

A benchmark is only as good as its reward structure. If reward is too sparse, the agent cannot learn useful behavior. If reward is too easy to exploit, the benchmark becomes meaningless. AITEA uses a dense, interpretable, and bounded reward design.

---

### Reward components

The reward combines:

- PnL improvement  
- execution quality  
- liquidity awareness  
- risk control  
- compliance  
- stability  
- portfolio progress  

---

### Reward penalties

The reward also penalizes:

- invalid actions  
- excessive churn  
- no-op abuse  
- risk breaches  
- destructive actions  
- repeated harmful behavior  

---

### Reward behavior

The reward:

- provides partial progress at every step  
- is bounded and clipped  
- is normalized for scoring  
- is deterministic under a fixed seed  

---

## Tasks

AITEA includes six tasks. Each task represents a real institutional trading objective and is paired with a deterministic grader.

---

### execution_easy

**Objective:** execute a target order efficiently in a relatively stable market.

**Why it matters:** this is the simplest form of real execution. A trader wants to buy or sell a target quantity while minimizing slippage and cost.

**What the agent should learn:**

- how to slice execution  
- how to avoid paying unnecessary spread  
- how to close a target quantity cleanly  

**Difficulty:** easy  

---

### liquidity_medium

**Objective:** execute or rebalance under low-liquidity conditions.

**Why it matters:** liquidity constraints are a core part of real trading. Large orders can move the market and worsen fill quality.

**What the agent should learn:**

- how to trade smaller pieces  
- how to balance speed with cost  
- how to respect market impact  

**Difficulty:** medium  

---

### fx_hedge_medium

**Objective:** reduce FX exposure efficiently.

**Why it matters:** many institutional books need to hedge currency risk. Good hedging should reduce exposure without overpaying for it.

**What the agent should learn:**

- hedge sizing  
- cost control  
- exposure reduction  

**Difficulty:** medium  

---

### rebalance_hard

**Objective:** rebalance a multi-asset portfolio under strict constraints.

**Why it matters:** portfolio managers rarely trade a single asset. They need to bring a whole book toward target weights while controlling turnover and risk.

**What the agent should learn:**

- target allocation  
- transaction cost tradeoffs  
- risk-aware rebalancing  
- reduced tracking error  

**Difficulty:** hard  

---

### news_adapt_hard

**Objective:** adapt strategy during structured news shocks.

**Why it matters:** markets can change quickly around earnings, macro events, and geopolitical shocks. Good agents must react without overreacting.

**What the agent should learn:**

- shock response  
- drawdown control  
- selective de-risking  
- post-event stability  

**Difficulty:** hard  

---

### regime_challenge_hard

**Objective:** operate under hidden market regime changes.

**Why it matters:** real markets switch between calm, volatile, crisis, and recovery states, often without the agent seeing the true latent regime directly.

**What the agent should learn:**

- robustness  
- adaptation  
- resilience under uncertainty  
- stable performance across conditions  

**Difficulty:** hardest  

---

## Graders

Each task has a deterministic grader.

### Why graders matter

The graders define how the benchmark judges quality. They must be fair, reproducible, and task-specific.

---

### Grader principles

A good grader should:

- return a score in [0.0, 1.0]  
- be deterministic  
- not always return the same value  
- reflect real progress  
- penalize poor behavior fairly  

---

### Grader mapping

- execution task → execution grader  
- liquidity task → liquidity grader  
- hedge task → FX hedge grader  
- rebalance task → rebalance grader  
- news task → news response grader  
- regime task → regime adaptation grader  

---

### What graders measure

Graders measure things like:

- implementation shortfall  
- partial fill quality  
- hedge efficiency  
- tracking error  
- drawdown  
- robustness  
- stability  
- cost discipline  

---

## Multi-Agent Realism

Although AITEA exposes a single-agent interface, the environment internally simulates a multi-agent market.

### Internal actors

The environment can simulate:

- noise traders  
- liquidity providers  
- momentum traders  
- adversarial flow  
- background order flow  

---

### Why this matters

This makes the market feel alive and prevents the environment from behaving like a static toy.

---

### Benefit for agents

Agents must learn to:

- react to market pressure  
- deal with changing liquidity  
- adapt to hidden stress  
- trade efficiently in a dynamic setting  

---

## API Contract

AITEA follows a strict environment contract.

### Core methods

- reset()  
- step(action)  
- state()  
- close()  

---

### Expected behavior

- reset() returns a clean initial transition  
- step(action) advances the environment by one step  
- state() returns the current observation  
- close() shuts the environment down safely  

---

### HTTP routes

The API layer exposes:

- GET /health  
- POST /reset  
- POST /step  
- GET /state  

These routes exist so the validator and deployment environment can interact with the environment over HTTP.

---

## Local Setup

This section explains how to set up, validate, and run the AITEA environment locally. It mirrors the exact conditions used during hackathon evaluation and ensures your submission passes all validation stages.

---

### Run Test Suite

Run the full test suite to verify that the environment is correctly set up:

    pytest -q

This validates:

- OpenEnv spec compliance  
- environment lifecycle (reset / step / state)  
- task loading and grading correctness  
- reward function behavior  
- inference output format compliance  

---

### Build Docker Image

Build the Docker image locally:

    docker build -t aitea-local:latest .

This ensures:

- the environment is fully containerized  
- all dependencies are correctly installed  
- runtime behavior matches deployment conditions  

---

### Run Docker Container

Start the container and expose the API:

    docker run --rm -p 7860:7860 aitea-local:latest

After startup, the API should be accessible at:

    http://localhost:7860

---

### Verify API Endpoints

Test the core endpoints manually or using tools like curl or Postman.

Health check:

    GET /health

Reset environment:

    POST /reset

Get current state:

    GET /state

Step environment:

    POST /step

These endpoints are mandatory for:

- OpenEnv validation  
- Hugging Face Space deployment  
- automated evaluation systems  

---

### Run Sample Episode (Debugging)

Run a manual episode for debugging and inspection:

    python scripts/sample_episode.py

This script:

- initializes the environment  
- executes a short trajectory using baseline logic  
- prints observations, rewards, and transitions  

Useful for:

- debugging environment logic  
- verifying reward behavior  
- inspecting state transitions step-by-step  

---

### Full Local Validation (Recommended)

Run the complete validation pipeline:

    bash scripts/validate_local.sh

This script performs:

- test execution  
- Docker build verification  
- container startup validation  
- API health checks  
- endpoint functionality verification  
- OpenEnv compliance checks  

This step ensures your environment is fully submission-ready.

---

## Summary

A successful local setup should:

- pass all tests without failure  
- build Docker image successfully  
- start container without errors  
- respond to all API endpoints  
- run inference script without crashing  
- produce valid structured logs  

---

## Common Issues

Issue: Import errors  
Cause: Missing dependencies  
Fix: Re-run pip install -r requirements.txt  

Issue: Docker build fails  
Cause: Incorrect paths or dependency issues  
Fix: Verify Dockerfile and requirements  

Issue: API not responding  
Cause: App not bound to correct port or startup failure  
Fix: Check FastAPI configuration  

Issue: Inference fails  
Cause: Missing environment variables  
Fix: Set API_BASE_URL, MODEL_NAME, HF_TOKEN  

Issue: Invalid logs  
Cause: Incorrect output format  
Fix: Follow strict [START] → [STEP] → [END] format  

---

## Final Note

This completes the local setup.

Once all steps pass successfully, the environment is fully ready for:

- validation  
- deployment  


---

## Environment Variables

The baseline and deployment flow expect:

- API_BASE_URL  
- MODEL_NAME  
- HF_TOKEN  

---

### Optional tuning variables

- AITEA_SEED  
- AITEA_EPISODE_LENGTH  
- AITEA_STARTING_CASH  
- AITEA_TRANSACTION_COST_PCT  
- AITEA_SLIPPAGE_COEFFICIENT  
- AITEA_DEFAULT_TASK  

These allow you to control the environment without editing code.

---

## Docker Deployment

The Dockerfile is meant to:

- install dependencies  
- copy the source tree  
- expose the right port  
- start the API cleanly  

---

### Deployment goals

- build successfully  
- start successfully  
- respond to /health  
- respond to /reset  
- respond to /state  

---

## Hugging Face Spaces Deployment

The repository is intended to run as a Hugging Face Space.

### What matters for Spaces

- clean startup  
- HTTP responsiveness  
- correct port binding  
- stable routes  
- zero manual intervention after deployment  

---

### Recommended behavior

The app should be importable as a standard FastAPI application object and should use the same environment code as the local setup.

This ensures local testing and hosted deployment match as closely as possible.

---

## Limitations

AITEA is realistic, but it is still a controlled simulation. It does not attempt to replicate every detail of real markets. It focuses on the parts that matter for agent learning:

- execution  
- liquidity  
- cost  
- risk  
- adaptation  
- robustness  

That tradeoff is intentional. The environment is designed for evaluation quality, not for perfect market microstructure fidelity.

---

## Summary

AITEA is a realistic institutional trading benchmark for agents. It provides:

- a structured observation space  
- a valid action space  
- dense reward  
- deterministic graders  
- six tasks with increasing difficulty  
- internal market realism  
- clean OpenEnv-style lifecycle  
- API access  
- Docker deployment  
- Hugging Face Space compatibility  
- reproducible baseline inference  

---

