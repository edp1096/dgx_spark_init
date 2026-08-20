SHELL := /bin/bash

.PHONY: converter env runtime download convert verify install-service upload clean-build all

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

upload:
	./scripts/upload_huggingface.sh

clean-build:
	rm -rf -- ./builder ./models ./output ./artifacts ./*.log

all: convert verify
