SHELL=/bin/bash

REGISTRY := $(or $(REG),docker.io)
REPOSITORY := $(or $(REPO),dayuhub)
IMAGE_REPO ?= $(REGISTRY)/$(REPOSITORY)
IMAGE_TAG ?= $(or $(TAG),v1.1-dev)

#$(info "REGISTRY = $(REGISTRY)")
#$(info "REPOSITORY = $(REPOSITORY)")
#$(info "IMAGE_TAG = $(IMAGE_TAG)")

.EXPORT_ALL_VARIABLES:

define BUILD_HELP_INFO
# Build Docker images using the build.sh script in the "hack" folder.
#
# Usage:
#   make build WHAT=component
#   make all
#   make help
#
# Components:
#   backend, frontend, datasource, generator, distributor, controller, monitor, scheduler, car-detection, etc.
#
# Example:
#   make build WHAT=monitor,generator
#   make all
#   make help
endef

.PHONY: build all help

help:
	@echo "$${BUILD_HELP_INFO}"

# Build images
build:
	@echo "Running build images of $(WHAT)"
	@echo "Current registry is: $(REGISTRY)"
	@echo "Current repository is: $(REPOSITORY)"
	@echo "Current image tag is: $(IMAGE_TAG)"
	bash hack/make-rules/cross-build.sh --files $(WHAT) --tag $(IMAGE_TAG) --repo $(REPOSITORY) --registry $(REGISTRY)

# Build all images
all:
	@echo "Running build images of $(WHAT)"
	@echo "Current registry is: $(REGISTRY)"
	@echo "Current repository is: $(REPOSITORY)"
	@echo "Current image tag is: $(IMAGE_TAG)"
	bash hack/make-rules/cross-build.sh --tag $(IMAGE_TAG) --repo $(REPOSITORY) --registry $(REGISTRY)


