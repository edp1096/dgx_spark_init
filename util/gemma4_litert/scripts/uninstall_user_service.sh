#!/usr/bin/env bash

set -euo pipefail

unit_name="media-prompt-enhancer.service"
unit_path="${HOME}/.config/systemd/user/${unit_name}"

systemctl --user disable --now "${unit_name}" >/dev/null 2>&1 || true
if [[ -e "${unit_path}" || -L "${unit_path}" ]]; then
  rm -f -- "${unit_path}"
fi
systemctl --user daemon-reload
systemctl --user reset-failed "${unit_name}" >/dev/null 2>&1 || true

echo "Removed ${unit_name}. LiteRT-LM runtime and registered models were preserved."
