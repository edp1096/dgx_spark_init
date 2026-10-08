#!/usr/bin/env bash

source "$(dirname "$0")/common.sh"

unit_source="${PROJECT_DIR}/systemd/media-prompt-enhancer.service"
unit_target="${HOME}/.config/systemd/user/media-prompt-enhancer.service"
install -Dm644 "${unit_source}" "${unit_target}"
systemctl --user daemon-reload
systemctl --user enable --now media-prompt-enhancer.service
