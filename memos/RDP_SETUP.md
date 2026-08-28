## ssh로 원격 데스크톱 켜기

* 아래 항목의 스크립트에 id/pw 맞춰 넣고 저장하고 실행권한 주고 실행.

* setup-gnome-rdp.sh
```sh
#!/usr/bin/env bash

# Run: sudo apt update; sudo ./setup-gnome-rdp.sh

set -Eeuo pipefail

# Edit these values before running the script.
readonly RDP_USERNAME="enter_the_usename"
readonly RDP_PASSWORD="enter_the_password"
readonly RDP_PORT="3389"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

[[ -n "$RDP_USERNAME" ]] || die "RDP_USERNAME must not be empty"
[[ "$RDP_USERNAME" != *$'\n'* ]] || die "RDP_USERNAME must not contain a newline"
[[ -n "$RDP_PASSWORD" ]] || die "RDP_PASSWORD must not be empty"
[[ "$RDP_PORT" =~ ^[0-9]+$ ]] || die "RDP_PORT must be a number"
((RDP_PORT >= 1 && RDP_PORT <= 65535)) || die "RDP_PORT must be between 1 and 65535"

if ((EUID != 0)); then
    command -v sudo >/dev/null 2>&1 || die "sudo is required"
    exec sudo -- "$0"
fi

command -v systemctl >/dev/null 2>&1 || die "systemd is required"
[[ -r /etc/os-release ]] || die "cannot identify the operating system"

# shellcheck source=/dev/null
. /etc/os-release
[[ "${ID:-}" == "ubuntu" ]] || die "this script currently supports Ubuntu only"

printf 'RDP username: %s\n' "$RDP_USERNAME"

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y gnome-remote-desktop openssl

getent passwd gnome-remote-desktop >/dev/null || die "gnome-remote-desktop service user is missing"
command -v grdctl >/dev/null 2>&1 || die "grdctl was not installed"

readonly cert_dir="/var/lib/gnome-remote-desktop/certificates"
readonly cert_file="$cert_dir/rdp.crt"
readonly key_file="$cert_dir/rdp.key"

install -d -m 0700 -o gnome-remote-desktop -g gnome-remote-desktop "$cert_dir"

if [[ ! -s "$cert_file" || ! -s "$key_file" ]]; then
    cert_name="$(hostname -f 2>/dev/null || hostname)"
    cert_name="${cert_name//\//-}"
    primary_ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
    san="DNS:$cert_name"
    [[ -z "$primary_ip" ]] || san+=",IP:$primary_ip"

    umask 077
    openssl req -newkey rsa:2048 -nodes -x509 -sha256 -days 3650 \
        -subj "/CN=$cert_name" \
        -addext "subjectAltName=$san" \
        -keyout "$key_file" \
        -out "$cert_file"
    chown gnome-remote-desktop:gnome-remote-desktop "$key_file" "$cert_file"
    chmod 0600 "$key_file"
    chmod 0644 "$cert_file"
else
    printf 'Reusing existing TLS certificate: %s\n' "$cert_file"
fi

grdctl --system rdp set-tls-key "$key_file"
grdctl --system rdp set-tls-cert "$cert_file"
grdctl --system rdp set-credentials "$RDP_USERNAME" "$RDP_PASSWORD"
grdctl --system rdp set-port "$RDP_PORT"
grdctl --system rdp disable-view-only
grdctl --system rdp enable

systemctl enable gnome-remote-desktop.service
systemctl restart gnome-remote-desktop.service

if command -v ufw >/dev/null 2>&1 && ufw status | head -n 1 | grep -q '^Status: active'; then
    ufw allow "$RDP_PORT/tcp"
fi

printf '\nGNOME Remote Desktop status:\n'
grdctl --system status

if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)$RDP_PORT$"; then
    primary_ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
    printf '\nSUCCESS: RDP is listening on port %s.\n' "$RDP_PORT"
    [[ -z "$primary_ip" ]] || printf 'Connect to: %s:%s\n' "$primary_ip" "$RDP_PORT"
    printf 'RDP username: %s\n' "$RDP_USERNAME"
else
    printf '\nRDP did not start listening on port %s. Recent service logs:\n' "$RDP_PORT" >&2
    journalctl -u gnome-remote-desktop.service -b --no-pager -n 50 >&2
    exit 1
fi
```

