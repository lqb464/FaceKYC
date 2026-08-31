.PHONY: install install-vision validate-data train-pad evaluate-pad evaluate-verification promote batch-verify monitor api frontend test lint format compile docker-check audit-notebooks

PYTHON ?= python

install: ## Core, API, frontend, notebook and development dependencies
	$(PYTHON) -m pip install -e ".[dev,api,frontend,notebooks]"

install-vision: ## PyTorch, facenet-pytorch and image runtime
	$(PYTHON) -m pip install -e ".[vision]"

validate-data: ## Validate CelebA-Spoof manifest and subject isolation
	$(PYTHON) scripts/validate_data.py --manifest data/processed/celeba_spoof_manifest.csv

train-pad: ## Train PAD using train/validation only; requires --dataset-root when invoked directly
	@echo "Run: $(PYTHON) scripts/train_liveness.py --dataset-root PATH_TO_CELEBA_SPOOF"

evaluate-verification: ## Show the leakage-safe LFW stages
	@echo "Run compare, calibrate, then holdout exactly once; see notebooks/README.md"

promote: ## Show artifact promotion command
	@echo "Run: $(PYTHON) scripts/promote_artifact.py --model-version VERSION"

batch-verify: ## Show bounded batch command (candidate use must be explicit)
	@echo "Run: $(PYTHON) scripts/batch_verify.py --manifest manifest.csv --output outputs/results.jsonl [--allow-candidate]"

monitor: ## Show privacy-safe monitoring command
	@echo "Run: $(PYTHON) scripts/monitor_results.py --input outputs/results.jsonl --output reports/monitoring.json"

api: ## Run API locally
	$(PYTHON) -m uvicorn backend.main:app --host 0.0.0.0 --port 8004

frontend: ## Run analyst review UI locally
	FACEKYC_API_URL=http://localhost:8004 $(PYTHON) -m streamlit run frontend/app.py --server.port 8504

test: ## Unit and integration tests without model downloads
	$(PYTHON) -m pytest

lint: ## Static quality checks
	$(PYTHON) -m ruff check src/facekyc backend frontend scripts tests

format: ## Format and safe lint fixes
	$(PYTHON) -m ruff format src/facekyc backend frontend scripts tests
	$(PYTHON) -m ruff check --fix src/facekyc backend frontend scripts tests

compile: ## Import-independent bytecode check
	$(PYTHON) -m compileall -q src/facekyc backend scripts

docker-check: ## Validate Compose syntax
	docker compose config --quiet

audit-notebooks: ## Require execution counts and reject error outputs
	$(PYTHON) scripts/audit_notebooks.py
