#!/usr/bin/env bash

set -euo pipefail

unit_dir="${XDG_CONFIG_HOME:-${HOME}/.config}/systemd/user"
unit_target="${unit_dir}/llama-cpp-spark.service"

systemctl --user disable --now llama-cpp-spark.service 2>/dev/null || true
if [[ -f "${unit_target}" ]]; then
  rm -- "${unit_target}"
fi
systemctl --user daemon-reload
echo "Removed llama-cpp-spark.service; host artifacts and models were preserved."
