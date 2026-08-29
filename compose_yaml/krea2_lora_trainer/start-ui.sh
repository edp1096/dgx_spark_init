#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${HF_TOKEN:-}" ]]; then
    unset HF_TOKEN
fi

state_dir="${AI_TOOLKIT_STATE_ROOT:-/data/official-ui}"
mkdir -p "${state_dir}" /opt/ai-toolkit/datasets /opt/ai-toolkit/output

# Prisma keeps this path fixed relative to the repository. Point it at the
# bind-mounted state directory so jobs and settings survive image rebuilds.
rm -f /opt/ai-toolkit/aitk_db.db
ln -s "${state_dir}/aitk_db.db" /opt/ai-toolkit/aitk_db.db

cd /opt/ai-toolkit/ui
npx prisma db push
exec npm run start
