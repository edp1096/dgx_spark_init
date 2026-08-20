COMPOSE := docker compose --env-file versions.env
YTDLP_UPDATE_TO ?= nightly

.PHONY: test build up down logs ps clean purge ytdlp-version ytdlp-update ytdlp-reset

test:
	go test ./...

build:
	$(COMPOSE) build

up:
	$(COMPOSE) up -d --remove-orphans

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f api

ps:
	$(COMPOSE) ps

clean:
	$(COMPOSE) down --rmi local --remove-orphans

purge:
	$(COMPOSE) down --rmi local --remove-orphans --volumes

ytdlp-version:
	$(COMPOSE) run --rm --no-deps --entrypoint /bin/sh api -c 'if test -x /var/lib/sparktalk-media-api/bin/yt-dlp; then /var/lib/sparktalk-media-api/bin/yt-dlp --version; else /usr/local/bin/yt-dlp --version; fi'

ytdlp-update:
	$(COMPOSE) run --rm --no-deps --user 0:0 --entrypoint /bin/sh -e YTDLP_UPDATE_TO=$(YTDLP_UPDATE_TO) api -c 'set -eu; mkdir -p /var/lib/sparktalk-media-api/bin; if test ! -x /var/lib/sparktalk-media-api/bin/yt-dlp; then cp /usr/local/bin/yt-dlp /var/lib/sparktalk-media-api/bin/yt-dlp; fi; chmod u+w /var/lib/sparktalk-media-api/bin/yt-dlp; /var/lib/sparktalk-media-api/bin/yt-dlp --update-to "$$YTDLP_UPDATE_TO"; chmod 0555 /var/lib/sparktalk-media-api/bin/yt-dlp'
	$(COMPOSE) restart api

ytdlp-reset:
	$(COMPOSE) run --rm --no-deps --user 0:0 --entrypoint /bin/sh api -c 'rm -f /var/lib/sparktalk-media-api/bin/yt-dlp'
	$(COMPOSE) restart api
