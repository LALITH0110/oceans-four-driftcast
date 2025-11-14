# File Summary:
# - Make targets for environment setup, data generation, simulations, and docs.
# - Wraps Typer CLI commands to provide a reproducible developer workflow.
# - Invoked by CI and contributors for quickstart automation.

.PHONY: env precommit data run sweep preview final perf judge test docs

PYTHON ?= python

env:
	conda env update -f environment.yml --prune

precommit:
	pre-commit install

data:
	$(PYTHON) scripts/generate_synthetic_inputs.py
	$(PYTHON) -m driftcast.cli ingest normalize data/raw/mock_crowd.json --out-dir data/crowd/processed

run:
	$(PYTHON) -m driftcast.cli run --config configs/natl_subtropical_gyre.yaml --seed 42

sweep:
	$(PYTHON) -m driftcast.cli sweep --config configs/natl_shipping.yaml --param physics.diffusivity_m2s=5,10

preview:
	$(PYTHON) -m driftcast.cli animate preview --config configs/natl_coastal.yaml --seed 42

final:
	$(PYTHON) -m driftcast.cli animate final --config configs/natl_coastal.yaml --seed 42

perf:
	$(PYTHON) -m driftcast.cli perf check --config configs/natl_subtropical_gyre.yaml --seed 123

judge:
	$(PYTHON) -m driftcast.cli judge --config configs/natl_subtropical_gyre.yaml --seed 42

test:
	pytest --cov=driftcast --cov-report=term-missing

docs:
	$(PYTHON) scripts/build_docs.py
