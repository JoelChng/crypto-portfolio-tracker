PYTHON = $(HOME)/miniforge3/envs/mlenv/bin/python
CONFIG = configs/pipeline.yaml

.PHONY: install data features train explain api all test

install:
	pip install -r requirements.txt

data:
	$(PYTHON) run_pipeline.py --config $(CONFIG) --stage data

features:
	$(PYTHON) run_pipeline.py --config $(CONFIG) --stage features

train:
	$(PYTHON) run_pipeline.py --config $(CONFIG) --stage train

explain:
	$(PYTHON) run_pipeline.py --config $(CONFIG) --stage explain

api:
	$(PYTHON) run_pipeline.py --config $(CONFIG) --stage api

all:
	$(PYTHON) run_pipeline.py --config $(CONFIG) --stage all

test:
	pytest tests/ -v

lint:
	ruff check . --fix
