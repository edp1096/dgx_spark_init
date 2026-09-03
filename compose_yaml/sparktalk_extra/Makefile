COMPOSE := docker compose --env-file versions.env
YTDLP_UPDATE_TO ?= nightly
SSH_DATA_DIR ?= /home/edp1096/.local/share/sparktalk/extra/ssh
SSH_KEY ?=
SSH_KEY_ID ?=

.PHONY: init test build up down logs ps clean purge collector-logs media-logs ssh-logs ytdlp-version ytdlp-update ytdlp-reset ssh-key-import

init:
	mkdir -p "$(SSH_DATA_DIR)/keys" "$(SSH_DATA_DIR)/state"
	touch "$(SSH_DATA_DIR)/state/known_hosts"
	chmod 0700 "$(SSH_DATA_DIR)" "$(SSH_DATA_DIR)/keys" "$(SSH_DATA_DIR)/state"
	chmod 0600 "$(SSH_DATA_DIR)/state/known_hosts"

test:
	go test ./...

build:
	$(COMPOSE) build

up: init
	SPARKTALK_EXTRA_SSH_DATA_DIR="$(SSH_DATA_DIR)" $(COMPOSE) up -d --remove-orphans

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f collector media ssh

collector-logs:
	$(COMPOSE) logs -f collector

media-logs:
	$(COMPOSE) logs -f media

ssh-logs:
	$(COMPOSE) logs -f ssh

ps:
	$(COMPOSE) ps

clean:
	$(COMPOSE) down --rmi local --remove-orphans

purge:
	$(COMPOSE) down --rmi local --remove-orphans --volumes

ssh-key-import: init
	test -n "$(SSH_KEY)" -a -f "$(SSH_KEY)"
	test -n "$(SSH_KEY_ID)"
	install -m 0600 "$(SSH_KEY)" "$(SSH_DATA_DIR)/keys/$(SSH_KEY_ID)"

ytdlp-version:
	$(COMPOSE) run --rm --no-deps --entrypoint /bin/sh media -c 'if test -x /var/lib/sparktalk-extra/media/bin/yt-dlp; then /var/lib/sparktalk-extra/media/bin/yt-dlp --version; else /usr/local/bin/yt-dlp --version; fi'

ytdlp-update:
	$(COMPOSE) run --rm --no-deps --user 0:0 --entrypoint /bin/sh -e YTDLP_UPDATE_TO=$(YTDLP_UPDATE_TO) media -c 'set -eu; mkdir -p /var/lib/sparktalk-extra/media/bin; if test ! -x /var/lib/sparktalk-extra/media/bin/yt-dlp; then cp /usr/local/bin/yt-dlp /var/lib/sparktalk-extra/media/bin/yt-dlp; fi; chmod u+w /var/lib/sparktalk-extra/media/bin/yt-dlp; /var/lib/sparktalk-extra/media/bin/yt-dlp --update-to "$$YTDLP_UPDATE_TO"; chmod 0555 /var/lib/sparktalk-extra/media/bin/yt-dlp'
	$(COMPOSE) restart media

ytdlp-reset:
	$(COMPOSE) run --rm --no-deps --user 0:0 --entrypoint /bin/sh media -c 'rm -f /var/lib/sparktalk-extra/media/bin/yt-dlp'
	$(COMPOSE) restart media
