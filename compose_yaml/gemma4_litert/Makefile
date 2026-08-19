SHELL := /bin/bash

.PHONY: converter env download convert verify install-service upload all

converter:
	./scripts/build_converter_arm64.sh

env: converter
	./scripts/setup_conversion_env.sh

download: env
	./scripts/download_model.sh

convert: download
	./scripts/convert_multimodal.sh

verify:
	./scripts/verify_bundle.sh

install-service:
	./scripts/install_user_service.sh

upload:
	./scripts/upload_huggingface.sh

all: convert verify
