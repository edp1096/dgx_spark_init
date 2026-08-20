SHELL := /bin/bash

.PHONY: converter env runtime download convert verify install-service uninstall-service upload clean-build clean-runtime clean-all all

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
