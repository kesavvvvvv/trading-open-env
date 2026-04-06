# AITEA Deployment Guide

## Overview

AITEA is designed to run in three modes:

1. local development
2. Docker container deployment
3. Hugging Face Spaces deployment

The project is intentionally structured so the same codebase supports all three without a separate rewrite. The environment logic, schema layer, reward system, task definitions, graders, API, and baseline inference script are all meant to work together in a consistent way.

The deployment goal is not only to “run”, but to start cleanly, answer HTTP requests reliably, and behave deterministically enough for automated validation.

---

## What must work before submission

Before submitting the project, the following should all succeed:

- the repository should install and import without errors
- the test suite should pass
- the Docker image should build successfully
- the container should start successfully
- the API should answer `/health`
- the API should answer `/reset`
- the API should answer `/state`
- the baseline script `inference.py` should run end-to-end
- the OpenEnv metadata file should be valid
- all tasks should be registered
- all graders should return values in `[0.0, 1.0]`
- the environment should support a stable `reset() -> step() -> state()` lifecycle

This is the minimum standard for a serious hackathon submission.

---

## Repository structure expected for deployment

The project is arranged so that deployment-related files live at the repository root, while implementation lives under `src/aitea/`.

Key root-level files:

- `Dockerfile`
- `openenv.yaml`
- `inference.py`
- `requirements.txt`
- `pyproject.toml`
- `.env.example`
- `README.md`

Key source directories:

- `src/aitea/env/`
- `src/aitea/schemas/`
- `src/aitea/engines/`
- `src/aitea/tasks/`
- `src/aitea/graders/`
- `src/aitea/reward/`
- `src/aitea/agents/`
- `src/aitea/api/`
- `src/aitea/utils/`

This layout keeps the environment logic modular and makes the deployment path predictable.

---

## Local setup

### Requirements

You need:

- Python 3.10 or newer
- pip
- Docker for container testing
- access to the OpenAI client for baseline inference
- a valid Hugging Face token or API token if your deployment path requires authentication

### Installation

From the repository root:

    pip install -r requirements.txt

For development, editable installation is also reasonable if the package metadata supports it.

### Environment variables

The baseline inference script expects the following variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

These values allow the inference script to talk to the intended model endpoint. They should be present in the runtime environment before running `inference.py`.

Example values:

    API_BASE_URL=http://127.0.0.1:7860
    MODEL_NAME=gpt-4.1-mini
    HF_TOKEN=your_token_here

For local smoke testing, the API base URL can point to the local service.

You may also keep a local `.env` file for development, but it should not be committed.

---

## Local validation workflow

A strong local workflow is:

1. run the test suite
2. build the Docker image
3. start the container
4. ping the health endpoint
5. confirm reset/state behavior
6. run the baseline inference script
7. verify the log format
8. check that the grader outputs remain in range

The helper scripts in `scripts/` are designed to support this workflow.

Typical validation order:

- `scripts/validate_local.sh`
- `scripts/build_docker.sh`
- `scripts/run_baseline.sh`
- `scripts/sample_episode.py`

The validation process should fail early if something is broken, rather than silently proceeding with a bad state.

---

## Docker deployment

### Purpose

Docker ensures the project starts cleanly in a controlled environment. The hackathon validator will check that the image builds and runs.

### Expected behavior

A valid Docker image should:

- install dependencies
- copy the project
- expose the API port
- start the FastAPI app
- respond to `/health`
- support `/reset`, `/step`, and `/state`

### Build pattern

The Dockerfile should use a slim Python base image and install only the dependencies needed at runtime.

The general pattern is:

    docker build -t aitea-local:latest .

### Run pattern

The container should start without requiring manual code edits. A typical run command is:

    docker run --rm -p 7860:7860 aitea-local:latest

If your image launches the FastAPI app correctly, the health endpoint should become available shortly after startup.

### Good practice

Keep the Docker image lightweight:

- use a slim Python base image
- avoid unnecessary build tools
- install only runtime dependencies
- keep the startup command explicit
- make sure the container binds to the correct host and port

### Common Docker failure modes

#### 1. Image builds but API does not start
Usually caused by:

- missing dependencies
- incorrect entrypoint
- import errors
- wrong module path
- not exposing or binding the correct port

#### 2. Container starts but `/health` fails
Usually caused by:

- app import errors
- startup exception
- wrong `uvicorn` target
- environment variables missing at startup

#### 3. Container starts but `/reset` fails
Usually caused by:

- environment not initialized
- missing task registration
- schema mismatch
- state manager not returning a valid transition

---

## Hugging Face Spaces deployment

### Purpose

AITEA should be deployable as a containerized Hugging Face Space tagged for OpenEnv-style evaluation.

### What the Space should expose

The Space should serve the API application and respond to HTTP checks.

The most important routes are:

- `/health`
- `/reset`
- `/step`
- `/state`

### Important deployment expectations

- the container must start automatically
- the app should bind to the expected host and port
- `/health` must return a success response
- `/reset` should create a usable episode
- `/state` should return a valid observation snapshot

### Recommended startup pattern

The HF Space entrypoint should launch the API app through the same application object used locally.

That keeps behavior consistent across:

- local tests
- container tests
- hosted evaluation

### Deployment discipline

The HF Space should not rely on manual post-start configuration. The app should be ready as soon as the container starts.

The safest pattern is:

- import the app from `src/aitea/api/app.py`
- expose it as `app`
- let the platform run the same app used in local development

This reduces surprises during automated review.

### Common HF Space failure modes

#### 1. Space runs but does not respond
Usually caused by:

- app not binding to the expected port
- incorrect startup command
- missing runtime dependency
- failure in environment initialization

#### 2. Space responds to `/health` but not `/reset`
Usually caused by:

- environment object not being attached to app state
- reset route calling the environment before initialization
- missing task registration

#### 3. Space returns malformed JSON
Usually caused by:

- using raw Python objects in responses
- incomplete serialization
- returning unsupported types from models

---

## OpenEnv compatibility and deployment

The environment is intended to follow the OpenEnv-style lifecycle:

- typed observation and action objects
- `reset()`
- `step(action)`
- `state()`
- `openenv.yaml`

That means deployment should preserve the same public contract everywhere.

The following should remain stable:

- method names
- task names
- reward range
- schema field names
- API route names
- baseline log format

If these drift, validators and graders can fail even if the project “runs”.

---

## Baseline inference deployment

The root `inference.py` file must run in the same deployed environment as the tasks.

It should:

- use the OpenAI client
- read `API_BASE_URL`
- read `MODEL_NAME`
- read `HF_TOKEN`
- connect to the environment
- step through all tasks
- emit structured logs in the required format

The inference script should be deterministic enough to reproduce a baseline score.

A safe baseline should not depend on hidden manual interaction.

---

## Validation helper scripts

### `scripts/validate_local.sh`

This script should do the following:

- run the test suite
- build the Docker image
- launch the container
- call `/health`
- call `/reset`
- call `/state`
- attempt OpenEnv validation if the CLI is available

This is your fastest pre-submission sanity check.

### `scripts/run_baseline.sh`

This script should:

- set fallback environment values if needed
- run `inference.py`
- let the user override `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN`

This makes the baseline reproducible and easy to test.

### `scripts/build_docker.sh`

This script should:

- build the Docker image
- print the resulting image tag

It is useful when iterating on the container configuration.

### `scripts/sample_episode.py`

This script should:

- create the environment
- reset it
- run a few baseline steps
- print state and diagnostic info
- show how the environment behaves step by step

It is valuable for debugging without the full inference loop.

---

## Environment variables

The following variables are important for deployment and baseline inference:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Recommended additional optional variables:

- `AITEA_SEED`
- `AITEA_EPISODE_LENGTH`
- `AITEA_STARTING_CASH`
- `AITEA_TRANSACTION_COST_PCT`
- `AITEA_SLIPPAGE_COEFFICIENT`
- `AITEA_DEFAULT_TASK`

These are useful if you want to tune the environment behavior without changing code.

---

## Logging and runtime expectations

The deployment should produce clear logs. In particular:

- startup logs should make it obvious the service is running
- API logs should show route calls and status
- baseline logs should follow the required `[START]`, `[STEP]`, `[END]` structure
- errors should be surfaced clearly instead of failing silently

For judge compatibility, the runtime should stay stable and not produce wildly inconsistent output.

---

## Suggested container behavior

A good container lifecycle is:

1. install dependencies
2. import the package successfully
3. initialize the FastAPI app
4. initialize the environment lazily or at startup
5. begin serving requests
6. keep the same environment instance reachable to the routes

This is simple and reliable.

---

## What not to do

Do not:

- hardcode secrets into the repository
- depend on manual intervention after startup
- return unstructured Python objects from API endpoints
- let reward values escape the expected range
- use nondeterministic grader logic
- make task names drift from the registry
- change the baseline log format

Those are the kinds of issues that break validation even if the simulation itself looks fine.

---

## Submission checklist

Before submitting:

- confirm the Dockerfile builds
- confirm the API answers `/health`
- confirm `/reset` returns a valid transition
- confirm `/state` returns an observation
- confirm the baseline script runs end-to-end
- confirm the test suite passes
- confirm all tasks are registered
- confirm the graders produce scores in `[0.0, 1.0]`
- confirm the OpenEnv metadata is valid
- confirm the environment responds consistently across local and container runs

---

## Deployment principle

The deployment should behave identically whether it runs:

- on your laptop
- inside Docker
- on a Hugging Face Space

That consistency is what makes the environment dependable for automated judging.

A strong submission is not just a clever simulation. It is a simulation that starts cleanly, responds predictably, and preserves the same contract everywhere it runs.