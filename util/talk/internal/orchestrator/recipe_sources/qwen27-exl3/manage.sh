#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
usage() {
 cat <<'EOF'
Usage: ./manage.sh COMMAND [OPTIONS]
Commands: setup image model start stop restart status logs validate
  setup       Prepare image, model, and worker files; does not start the server.
  image       Build or pull the runtime image (legacy alias: build).
  model       Prepare model weights and worker copy (legacy alias: prepare).
Options for setup/model:
  --official       Select original weights / disable the runtime ablation.
  --abliterated    Select abliterated weights / enable the runtime ablation.
  --ask-token      Read a Hugging Face token without echoing it.
Environment: HF_TOKEN (download only; never saved by this command).
Configuration: .env (copy of env.sample); MODEL_VARIANT=official|abliterated.
Logs: logs [head|worker] (worker is available on cluster recipes).
EOF
}
action="${1:-status}"
case "$action" in -h|--help|help) usage; exit 0;; esac
shift "$(( $# > 0 ? 1 : 0 ))"
case "$action" in build) action=image;; prepare) action=model;; esac
case "$action" in setup|image|model|start|stop|restart|status|logs|validate) ;; *) echo "Unknown command: $action" >&2; exit 2;; esac
variant=''; ask=0; log_host=''
while (( $# )); do
 case "$1" in
  --official|--abliterated)
   next="${1#--}"; [[ -z "$variant" || "$variant" == "$next" ]] || { echo 'Conflicting model options' >&2; exit 2; }; variant="$next";;
  --ask-token) ask=1;;
  head|worker) [[ "$action" == logs && -z "$log_host" ]] || { echo 'Unexpected argument' >&2; exit 2; }; log_host="$1";;
  -h|--help) usage; exit 0;;
  *) echo 'Unknown option (token values must use HF_TOKEN or --ask-token)' >&2; exit 2;;
 esac
 shift
done
if [[ -n "$variant" || "$ask" == 1 ]]; then
 [[ "$action" == setup || "$action" == model ]] || { echo 'Model/token options require setup or model' >&2; exit 2; }
fi
# Preserve the caller token across .env loading, without logging or persisting it.
caller_token="${HF_TOKEN-${HUGGING_FACE_HUB_TOKEN-}}"
[[ -f "$script_dir/.env" ]] || { cp "$script_dir/env.sample" "$script_dir/.env"; chmod 600 "$script_dir/.env"; }
set -a
source "$script_dir/.env"
set +a
[[ -z "$caller_token" ]] || export HF_TOKEN="$caller_token"
unset caller_token
if [[ "$ask" == 1 ]]; then
 read -r -s -p 'Hugging Face token: ' HF_TOKEN < /dev/tty
 printf '\n' > /dev/tty
 [[ -n "$HF_TOKEN" ]] || { echo 'Token is empty' >&2; exit 2; }
 export HF_TOKEN
fi
if [[ -n "$variant" ]]; then
 if [[ "$MODEL_KIND" == qwen27-exl3 && "$variant" == official ]]; then
  echo 'This recipe currently provides only the Uncensored checkpoint; no official checkpoint is configured.' >&2; exit 2
 fi
 # Fixed keys/values only; update model selection, never the credential.
 python3 - "$script_dir/.env" "$MODEL_KIND" "$variant" <<'PYENV'
import sys,os,tempfile
from pathlib import Path
p=Path(sys.argv[1]);kind,variant=sys.argv[2:];updates={'MODEL_VARIANT':variant}
if kind=='glm53': updates['ABLIT']='1' if variant=='abliterated' else '0'
if kind=='ds4fve': updates['ABLITERATED']='1' if variant=='abliterated' else '0'
if kind=='flash-next-exl3': updates['ABLIT_LAMBDA']='1.5' if variant=='abliterated' else '0'
lines=p.read_text().splitlines()
for k,v in updates.items():
 lines=[x for x in lines if not x.startswith(k+'=')];lines.append(k+'='+v)
fd,name=tempfile.mkstemp(dir=p.parent,prefix='.env-')
with os.fdopen(fd,'w') as f:f.write('\n'.join(lines)+'\n')
os.replace(name,p)
PYENV
fi
if [[ "$log_host" == worker && "$MODEL_KIND" != glm53 && "$MODEL_KIND" != ds4fve ]]; then echo 'This recipe has no worker' >&2; exit 2; fi
export RUNTIME_HF_TOKEN="${HF_TOKEN-}"
args=("$action")
[[ -z "$log_host" ]] || args+=("$log_host")
exec bash "$script_dir/runtime.sh" "${args[@]}"
