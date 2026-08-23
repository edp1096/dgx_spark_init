SHELL := /bin/bash
CONTAINER_COMPOSE ?= docker compose -f compose.converter.yaml
CONTAINER_DATA_DIR ?= $(HOME)/.local/share/gemma4-litert-build
CONTAINER_LOG_DIR = $(CONTAINER_DATA_DIR)/logs
CONTAINER_ENV = HOST_UID="$(shell id -u)" HOST_GID="$(shell id -g)" GEMMA4_LITERT_CONTAINER_DATA_DIR="$(CONTAINER_DATA_DIR)"

.PHONY: converter env runtime download convert verify install-service uninstall-service upload clean-build clean-runtime clean-all all container-help container-image container-build container-convert container-verify container-all container-paths container-ps container-logs clean-container-build

converter:
	./scripts/build_converter_arm64.sh

env: converter
	./scripts/setup_conversion_env.sh

runtime:
	./scripts/setup_runtime_env.sh

download: env
	./scripts/download_model.sh

convert: download
	./scripts/convert_multimodal.sh

verify:
	./scripts/verify_bundle.sh

install-service: runtime
	./scripts/install_user_service.sh

uninstall-service:
	./scripts/uninstall_user_service.sh

upload:
	./scripts/upload_huggingface.sh

clean-build:
	rm -rf -- ./builder ./models ./output ./artifacts ./*.log

clean-runtime: uninstall-service
	./scripts/clean_runtime.sh

clean-all: clean-runtime clean-build

all: convert verify

container-help:
	@echo "Use make container-build, container-convert, container-verify, or container-all on the host."
	@echo "Use make container-ps or make container-logs to inspect an active converter."

container-paths:
	@echo "build root: $(CONTAINER_DATA_DIR)"
	@echo "wheel:      $(CONTAINER_DATA_DIR)/artifacts/wheels"
	@echo "model:      $(CONTAINER_DATA_DIR)/output/Huihui-gemma-4-E2B-it-abliterated-litert-lm/model.litertlm"
	@echo "logs:       $(CONTAINER_LOG_DIR)"

container-image:
	mkdir -p "$(CONTAINER_DATA_DIR)/home" "$(CONTAINER_DATA_DIR)/cache" "$(CONTAINER_LOG_DIR)"
	touch "$(CONTAINER_DATA_DIR)/.gemma4-litert-build-root"
	$(CONTAINER_ENV) $(CONTAINER_COMPOSE) build converter

container-build: container-image
	set -o pipefail; $(CONTAINER_ENV) $(CONTAINER_COMPOSE) run --rm converter make converter 2>&1 | tee "$(CONTAINER_LOG_DIR)/container-build.log"

container-convert: container-image
	set -o pipefail; $(CONTAINER_ENV) $(CONTAINER_COMPOSE) run --rm converter make convert 2>&1 | tee "$(CONTAINER_LOG_DIR)/container-convert.log"

container-verify: container-image
	set -o pipefail; $(CONTAINER_ENV) $(CONTAINER_COMPOSE) run --rm converter make verify 2>&1 | tee "$(CONTAINER_LOG_DIR)/container-verify.log"

container-all: container-convert
	set -o pipefail; $(CONTAINER_ENV) $(CONTAINER_COMPOSE) run --rm converter make verify 2>&1 | tee "$(CONTAINER_LOG_DIR)/container-verify.log"

container-ps:
	@docker ps --filter label=com.docker.compose.service=converter --format 'table {{.ID}}\t{{.Names}}\t{{.Status}}'

container-logs:
	@container_id="$$(docker ps --filter label=com.docker.compose.service=converter --format '{{.ID}}' | head -n 1)"; \
	if [[ -z "$$container_id" ]]; then \
		echo "No converter container is currently running." >&2; \
		echo "Completed run logs: $(CONTAINER_LOG_DIR)" >&2; \
		exit 1; \
	fi; \
	docker logs -f "$$container_id"

clean-container-build:
	GEMMA4_LITERT_CONTAINER_DATA_DIR="$(CONTAINER_DATA_DIR)" ./scripts/clean_container_build.sh
