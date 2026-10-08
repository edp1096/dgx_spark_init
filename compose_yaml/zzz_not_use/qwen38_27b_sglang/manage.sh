#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
usage() {
 cat <<'EOF'
Usage: ./manage.sh COMMAND [OPTIONS]
Commands: setup image model start stop restart status logs validate
  setup   Prepare runtime image and download model + DFlash2; do not start.
  model   Download model + DFlash2 (builds download runtime if missing).
  image   Build runtime image if missing.
Options for setup/model: --official | --abliterated, --ask-token
Configuration: .env (created from env.sample). Token: HF_TOKEN environment.
Abliterated downloads the complete pre-quantized checkpoint; no patch or PTQ.
EOF
}
action="${1:-status}"
case "$action" in help|-h|--help) usage; exit 0;; esac
[[ $# == 0 ]] || shift
case "$action" in build) action=image;; prepare) action=model;; esac
case "$action" in setup|image|model|start|stop|restart|status|logs|validate) ;; *) usage >&2; exit 2;; esac
variant=''; ask=0
while (( $# )); do
 case "$1" in
 --official|--abliterated)
  next="${1#--}"
  [[ -z "$variant" || "$variant" == "$next" ]] || { echo 'Conflicting variants' >&2; exit 2; }
  variant="$next";;
 --ask-token) ask=1;;
 -h|--help) usage; exit 0;;
 *) echo 'Unknown option; pass token with HF_TOKEN or --ask-token' >&2; exit 2;;
 esac
 shift
done
if [[ -n "$variant" || "$ask" == 1 ]]; then
 [[ "$action" == model || "$action" == setup ]] || { echo 'Options require setup/model' >&2; exit 2; }
fi
[[ -f "$script_dir/.env" ]] || (umask 077; cp "$script_dir/env.sample" "$script_dir/.env")
caller_token="${HF_TOKEN-${HUGGING_FACE_HUB_TOKEN-}}"
set -a
source "$script_dir/env.sample"
source "$script_dir/.env"
set +a
export HF_TOKEN="$caller_token"
unset caller_token
if [[ "$ask" == 1 ]]; then
 read -r -s -p 'Hugging Face token: ' HF_TOKEN < /dev/tty
 printf '\n' > /dev/tty
 export HF_TOKEN
fi
if [[ -n "$variant" ]]; then
 export MODEL_VARIANT="$variant"
 python3 - "$script_dir/.env" "$variant" <<'ENV'
import os,sys,tempfile
from pathlib import Path
p=Path(sys.argv[1]); lines=[x for x in p.read_text().splitlines() if not x.startswith('MODEL_VARIANT=')]
lines.append('MODEL_VARIANT='+sys.argv[2])
fd,name=tempfile.mkstemp(dir=p.parent,prefix='.env-')
with os.fdopen(fd,'w') as f: f.write('\n'.join(lines)+'\n')
os.replace(name,p)
ENV
fi
case "$MODEL_VARIANT" in official|abliterated) ;; *) echo 'Invalid MODEL_VARIANT' >&2; exit 2;; esac
exec bash "$script_dir/runtime.sh" "$action"
