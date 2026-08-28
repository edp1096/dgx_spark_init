#!/bin/sh

set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
prefix=${PREFIX:-${HOME}/.local}

if [ "$#" -gt 0 ]; then
    if [ "$#" -ne 2 ] || [ "$1" != "--prefix" ]; then
        echo "Usage: $0 [--prefix /absolute/path]" >&2
        exit 2
    fi
    prefix=$2
fi

case "$prefix" in
    /*) ;;
    *) echo "The install prefix must be an absolute path." >&2; exit 2 ;;
esac

if [ "$prefix" = "/" ]; then
    echo "Refusing to install with / as the prefix." >&2
    exit 2
fi

for command_name in make pkg-config ibus glib-compile-schemas; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "Missing required command: $command_name" >&2
        exit 1
    fi
done

if ! pkg-config --exists ibus-1.0 libhangul; then
    echo "Missing development packages for ibus-1.0 or libhangul." >&2
    exit 1
fi

regenerate_build_system=false
for generated_file in \
    configure config.guess config.sub compile missing install-sh depcomp \
    py-compile test-driver Makefile.in src/Makefile.in
do
    if [ ! -e "$script_dir/$generated_file" ]; then
        regenerate_build_system=true
        break
    fi
done

if [ "$regenerate_build_system" = true ]; then
    for command_name in autoconf automake aclocal autoreconf autopoint; do
        if ! command -v "$command_name" >/dev/null 2>&1; then
            echo "Missing build-system command: $command_name" >&2
            echo "Install the packages listed in README.md and try again." >&2
            exit 1
        fi
    done
    (cd "$script_dir" && NOCONFIGURE=1 ./autogen.sh)
fi

build_dir=$(mktemp -d "${TMPDIR:-/tmp}/hangul-backspace-install.XXXXXX")
cleanup()
{
    rm -rf -- "$build_dir"
}
trap cleanup EXIT HUP INT TERM

jobs=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)

cd "$build_dir"
"$script_dir/configure" --prefix="$prefix"
make -j"$jobs"
make check DISABLE_GUI_TESTS=hangul-backspace
make install

ibus_datadir=$(pkg-config --variable=datadir ibus-1.0)
if [ -z "$ibus_datadir" ]; then
    ibus_datadir=/usr/share
fi

IBUS_COMPONENT_PATH="$prefix/share/ibus/component:$ibus_datadir/ibus/component" \
    ibus write-cache

if command -v systemctl >/dev/null 2>&1 &&
   systemctl --user is-active --quiet org.freedesktop.IBus.session.GNOME.service; then
    systemctl --user restart org.freedesktop.IBus.session.GNOME.service
else
    ibus restart
fi

attempt=0
while [ "$attempt" -lt 50 ]; do
    if ibus list-engine 2>/dev/null | grep -q 'hangul-backspace'; then
        break
    fi
    attempt=$((attempt + 1))
    sleep 0.2
done

if ! ibus list-engine 2>/dev/null | grep -q 'hangul-backspace'; then
    echo "Installed, but IBus did not discover hangul-backspace." >&2
    echo "Log out and back in, then select hangul-backspace." >&2
    exit 1
fi

ibus engine hangul-backspace
echo "Installed and selected hangul-backspace."
