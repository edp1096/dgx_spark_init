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
    echo "Refusing to uninstall with / as the prefix." >&2
    exit 2
fi

if command -v ibus >/dev/null 2>&1; then
    current_engine=$(ibus engine 2>/dev/null || true)
    if [ "$current_engine" = "hangul-backspace" ]; then
        ibus engine hangul
    fi
fi

regenerate_build_system=false
for generated_file in \
    configure config.guess config.sub compile missing install-sh depcomp \
    py-compile test-driver Makefile.in src/Makefile.in setup/Makefile.in \
    icons/Makefile.in data/Makefile.in m4/Makefile.in
do
    if [ ! -e "$script_dir/$generated_file" ]; then
        regenerate_build_system=true
        break
    fi
done

if [ "$regenerate_build_system" = false ]; then
    for source_file in \
        configure.ac Makefile.am src/Makefile.am setup/Makefile.am \
        icons/Makefile.am data/Makefile.am m4/Makefile.am
    do
        if [ "$source_file" = configure.ac ]; then
            generated_file=configure
        else
            generated_file=${source_file%.am}.in
        fi
        if [ "$script_dir/$source_file" -nt "$script_dir/$generated_file" ]; then
            regenerate_build_system=true
            break
        fi
    done
fi

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

build_dir=$(mktemp -d "${TMPDIR:-/tmp}/hangul-backspace-uninstall.XXXXXX")
cleanup()
{
    rm -rf -- "$build_dir"
}
trap cleanup EXIT HUP INT TERM

cd "$build_dir"
"$script_dir/configure" --prefix="$prefix"
make uninstall

setup_dir="$prefix/share/hangul-backspace/setup"
if [ -d "$setup_dir" ]; then
    find "$setup_dir" -type f -name '*.pyc' -delete
    find "$setup_dir" -type d -name __pycache__ -empty -delete
fi

schema_dir="$prefix/share/glib-2.0/schemas"
if command -v glib-compile-schemas >/dev/null 2>&1 && [ -d "$schema_dir" ]; then
    glib-compile-schemas "$schema_dir"
fi

if command -v ibus >/dev/null 2>&1; then
    ibus write-cache
    if command -v systemctl >/dev/null 2>&1 &&
       systemctl --user is-active --quiet org.freedesktop.IBus.session.GNOME.service; then
        systemctl --user restart org.freedesktop.IBus.session.GNOME.service
    else
        ibus restart
    fi
fi

echo "Uninstalled hangul-backspace."
