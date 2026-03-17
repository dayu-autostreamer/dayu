SHELL=/bin/bash

REGISTRY := $(or $(REG),docker.io)
REPOSITORY := $(or $(REPO),dayuhub)
IMAGE_REPO ?= $(REGISTRY)/$(REPOSITORY)
IMAGE_TAG ?= $(or $(TAG),v1.3)
PYTHON ?= python3
NPM ?= npm
FRONTEND_DIR ?= frontend
PYTHONPATH_VALUE := $(CURDIR)/backend:$(CURDIR)/dependency
PYTHONPYCACHEPREFIX ?= $(CURDIR)/.cache/pycache

NOCACHE ?= $(or $(NO_CACHE),0)
BUILD_NO_CACHE_FLAG := $(if $(filter 1 true TRUE yes YES,$(NOCACHE)),--no-cache,)

.EXPORT_ALL_VARIABLES:

define HELP_INFO
# Dayu developer entry points.
#
# Build:
#   make build WHAT=component
#   make all
#
# Components:
#   backend, frontend, datasource, generator, distributor, controller, monitor, scheduler, car-detection, etc.
#
# Quality:
#   make install-python-dev
#   make python-syntax
#   make test-unit-integration
#   make test-component
#   make test-e2e
#   make frontend-install
#   make frontend-lint
#   make frontend-format-check
#   make frontend-build
#   make check
#
# Examples:
#   make build WHAT=monitor,generator
#   make test-unit-integration
#   make frontend-lint
endef

.PHONY: help build all install-python-dev lint-python python-syntax test-unit-integration test-component test-e2e frontend-install frontend-lint frontend-format-check frontend-build check

help:
	@echo "$${HELP_INFO}"

# Build images
build:
	@echo "Running build images of $(WHAT)"
	@echo "Current registry is: $(REGISTRY)"
	@echo "Current repository is: $(REPOSITORY)"
	@echo "Current image tag is: $(IMAGE_TAG)"
	bash hack/make-rules/cross-build.sh --files $(WHAT) --tag $(IMAGE_TAG) --repo $(REPOSITORY) --registry $(REGISTRY) $(BUILD_NO_CACHE_FLAG)

# Build all images
all:
	@echo "Current registry is: $(REGISTRY)"
	@echo "Current repository is: $(REPOSITORY)"
	@echo "Current image tag is: $(IMAGE_TAG)"
	bash hack/make-rules/cross-build.sh --tag $(IMAGE_TAG) --repo $(REPOSITORY) --registry $(REGISTRY) $(BUILD_NO_CACHE_FLAG)

install-python-dev:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements-dev.txt

lint-python:
	PYTHONPATH="$(PYTHONPATH_VALUE)" $(PYTHON) -m ruff check \
		backend datasource components tools tests dependency/core/lib dependency/core/controller

python-syntax:
	PYTHONPYCACHEPREFIX="$(PYTHONPYCACHEPREFIX)" PYTHONPATH="$(PYTHONPATH_VALUE)" \
		$(PYTHON) -m compileall -q backend datasource components tools tests

test-unit-integration:
	PYTHONPATH="$(PYTHONPATH_VALUE)" $(PYTHON) -m pytest -m "unit or integration"

test-component:
	PYTHONPATH="$(PYTHONPATH_VALUE)" $(PYTHON) -m pytest -m component

test-e2e:
	PYTHONPATH="$(PYTHONPATH_VALUE)" $(PYTHON) -m pytest -m e2e

frontend-install:
	cd $(FRONTEND_DIR) && $(NPM) install --legacy-peer-deps --no-audit --no-fund --no-package-lock

frontend-lint:
	cd $(FRONTEND_DIR) && $(NPM) run lint

frontend-format-check:
	cd $(FRONTEND_DIR) && $(NPM) run format:check

frontend-build:
	cd $(FRONTEND_DIR) && $(NPM) run build

check: python-syntax test-unit-integration frontend-lint frontend-format-check
