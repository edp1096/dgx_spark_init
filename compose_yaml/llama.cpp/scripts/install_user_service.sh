#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
unit_source="${project_dir}/systemd/llama-cpp-spark.service"
unit_dir="${XDG_CONFIG_HOME:-${HOME}/.config}/systemd/user"
unit_target="${unit_dir}/llama-cpp-spark.service"

test -x "${project_dir}/artifacts/current/llama-server" || {
  echo "Build the host runtime first: ./scripts/build_host.sh" >&2
  exit 1
}

mkdir -p "${unit_dir}"
install -m 0644 "${unit_source}" "${unit_target}"
systemctl --user daemon-reload
systemctl --user enable --now llama-cpp-spark.service
echo "Installed ${unit_target}"
