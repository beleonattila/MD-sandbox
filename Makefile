# mdbench. Run targets inside the environment, e.g.
#   ~/.local/bin/micromamba run -r ~/micromamba -n mdbench-md make test
# `make env` only needs the micromamba binary; on a fresh machine without make, run the
# `micromamba create ... -f env/mdbench-md.lock.txt` line from the env target by hand.

SHELL       := /bin/bash
PYTHON      ?= python
MICROMAMBA  ?= $(HOME)/.local/bin/micromamba
MAMBA_ROOT  ?= $(HOME)/micromamba
ENV_NAME    ?= mdbench-md
CONFIG      ?=
BRANCH      ?=
N_ATOMS     ?=
DAYS        ?= 0

export PYTHONPATH := $(CURDIR)/src$(if $(PYTHONPATH),:$(PYTHONPATH))

.PHONY: help env test lint run score report clean-runs bench

help:
	@echo "targets: env test lint run CONFIG=... score BRANCH=... report clean-runs [DAYS=n] bench [N_ATOMS=n]"

# Creates mdbench-md from the explicit lock file when present, else from the YAML solve.
# mdbench-fe and mdbench-ml are deferred (see CLAUDE.md) and not created here.
env:
	@if $(MICROMAMBA) env list -r $(MAMBA_ROOT) | awk '{print $$1}' | grep -qx '$(ENV_NAME)'; then \
		echo "env $(ENV_NAME) already exists under $(MAMBA_ROOT); remove it first to rebuild"; \
	elif [ -f env/$(ENV_NAME).lock.txt ]; then \
		$(MICROMAMBA) create -y -r $(MAMBA_ROOT) -n $(ENV_NAME) -f env/$(ENV_NAME).lock.txt; \
	else \
		$(MICROMAMBA) create -y -r $(MAMBA_ROOT) -f env/$(ENV_NAME).yml; \
	fi
	@echo "mdbench-fe / mdbench-ml: deferred, not created"

test:
	$(PYTHON) -m pytest -q

lint:
	ruff check .
	ruff format --check .

run:
	@test -n "$(CONFIG)" || { echo "usage: make run CONFIG=configs/<name>.yml"; exit 2; }
	$(PYTHON) -m mdbench run $(CONFIG)

score:
	@test -n "$(BRANCH)" || { echo "usage: make score BRANCH=<a|b|c|d>"; exit 2; }
	$(PYTHON) -m mdbench score $(BRANCH)

report:
	$(PYTHON) -m mdbench report

# DAYS=0 (default) deletes everything under runs/; DAYS=n deletes only trajectory and
# checkpoint files older than n days. results/ is never touched.
clean-runs:
	@mkdir -p runs
	@if [ "$(DAYS)" = "0" ]; then \
		find runs -mindepth 1 -maxdepth 1 -exec rm -rf {} +; \
		echo "runs/ emptied; results/ kept"; \
	else \
		find runs -type f \( -name '*.xtc' -o -name '*.dcd' -o -name '*.nc' -o -name '*.chk' \) \
			-mtime +$(DAYS) -print -delete; \
	fi

bench:
	$(PYTHON) -m mdbench bench configs/bench.yml $(if $(N_ATOMS),--n-atoms $(N_ATOMS))
