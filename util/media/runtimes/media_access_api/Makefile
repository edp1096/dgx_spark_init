PYTHON ?= python3
COMPOSE := docker compose --env-file versions.env

.PHONY: show-versions check-updates set-version build up rebuild verify snapshot rollback

show-versions:
	$(PYTHON) scripts/versions.py show

check-updates:
	$(PYTHON) scripts/versions.py check

set-version:
	$(PYTHON) scripts/versions.py set "$(COMPONENT)" "$(VERSION)"

build:
	$(COMPOSE) build

up:
	$(COMPOSE) up -d --remove-orphans

rebuild: build up

verify:
	$(PYTHON) scripts/versions.py verify

snapshot:
	$(PYTHON) scripts/versions.py snapshot "$(TAG)"

rollback:
	$(PYTHON) scripts/versions.py rollback "$(TAG)"
