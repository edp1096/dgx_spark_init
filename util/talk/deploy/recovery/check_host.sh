#!/usr/bin/env bash
# Read-only prerequisites. Does not install software or start models.
set -u
failed=0
check() {
    local description="$1"
    shift
    if "$@" >/dev/null 2>&1; then
        printf 'PASS %s\n' "$description"
    else
        printf 'FAIL %s\n' "$description"
        failed=1
    fi
}
check 'Linux ARM64' bash -c 'test "$(uname -s)" = Linux && test "$(uname -m)" = aarch64'
check 'Python 3.11+' python3 -c 'import sys; sys.exit(sys.version_info < (3, 11))'
check 'NVIDIA driver' nvidia-smi
check 'Docker access without sudo' docker info
check 'Docker Compose plugin' docker compose version
check 'NVIDIA Container Toolkit' nvidia-ctk --version
check 'rsync' rsync --version
check 'OpenSSH client' ssh -V
if [[ $# -gt 0 ]]; then
    check 'Worker SSH without password' ssh -o BatchMode=yes -o ConnectTimeout=5 "$1" true
    check 'Worker Docker without sudo' ssh -o BatchMode=yes -o ConnectTimeout=5 "$1" docker info
    check 'Worker Python' ssh -o BatchMode=yes -o ConnectTimeout=5 "$1" python3 --version
    check 'Worker rsync' ssh -o BatchMode=yes -o ConnectTimeout=5 "$1" rsync --version
fi
printf 'These checks do not prove GPU container execution or a clean model installation.\n'
exit "$failed"
