# Mechanism Lab

An original, standalone experimental workbench for computational decision-making. Inspired by mechanistic neuroscience modelling, **not dependent on Neuroblox or Julia**.

## Features

- Reproducible two-armed bandit with reward reversal.
- Four competing models: random baseline, Q-learning, prediction-error-adaptive learning, and a leaky competing-trace model.
- Maximum-likelihood fitting, AIC/BIC model ranking, multi-start optimization.
- Synthetic parameter recovery and model-recovery counts.
- Matched-seed intervention comparisons.
- Interactive zero-build HTML dashboard with FastAPI backend.
- Tests and CLI for scripting experiments.

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e '.[dev]'
pytest -q
```

## Run the browser lab

In terminal 1:

```bash
uvicorn mechanism_lab.api:app --reload
```

In terminal 2:

```bash
python -m http.server 8080 -d web
```

Open http://localhost:8080.

## CLI

```bash
mechanism-lab simulate --output simulation.json
mechanism-lab compare --model adaptive --output comparison.json
mechanism-lab recover --model qlearning --output recovery.json
mechanism-lab intervene --model qlearning --output intervention.json
```

## Interpretation and limitations

All agents are **simplified computational models**. The adaptive model is a heuristic, not Bayesian inference. The leaky model is a coarse dynamical analogue, not a biophysical neural circuit. Model comparison uses in-sample AIC/BIC, not held-out prediction or formal evidence. Recovery estimates can be noisy or non-identifiable, especially for short tasks. Intervention comparisons modify model parameters only, and do not imply causal effects in real brains. The UI uses a local API and is intended for development, not public deployment.

## Project structure

- `mechanism_lab/core.py`: task, agents, simulation, fitting, comparison, recovery, interventions
- `mechanism_lab/api.py`: HTTP API
- `mechanism_lab/cli.py`: command-line experiments
- `web/index.html`: browser workbench
- `tests/test_core.py`: automated tests

## Next research improvements

Implement held-out predictive evaluation, posterior uncertainty, parameter identifiability maps, genuinely Bayesian change-point learners, and validated spiking neural dynamics. These are deliberately not claimed as implemented.
