IBus Hangul engine with Backspace repeat handling.

## Install

```sh
sudo apt install autopoint gettext libtool libibus-1.0-dev libhangul-dev
./install-user.sh
```

## Uninstall

```sh
./uninstall-user.sh
```

## Note
Installation path: `~/.local`

Automatic jamo reordering is disabled by default: `ls` produces `ㅣㄴ`, while
`sl` composes `니`. The engine's “Automatic reordering” setting can enable the
alternative behavior explicitly. Existing saved values take precedence over
the default; turn that setting off if an earlier installation enabled it.

### Backspace diagnostics and tests

Native Wayland terminals can consume the first Backspace press while Hangul
is still composing. On Mutter 50, forwarding subsequent presses loses the
input device and stops deletion. For terminal input contexts without IBus's
synchronous key-processing capability, the engine commits the standard DEL
control character (`0x7f`) for repeats after composition becomes empty.
Releasing Backspace ends this handling. Fresh presses and Latin input use the
terminal's normal Backspace mapping. Terminals configured to require `0x08`
instead of DEL are not supported by this fallback.

Set `HANGUL_BACKSPACE_TRACE=1` when starting the engine to log Backspace
press/release events, capabilities, and composition state to stderr. The trace
does not include input text.

`make check` includes headless Backspace tests. They use a private D-Bus and
in-memory settings, and do not send keys to the desktop or change user settings.

An optional integration test runs a private, headless GNOME Wayland desktop
with a VTE terminal and virtual keyboard. It requires GNOME Shell with headless
support, `dbus-run-session`, dconf, and Python GI bindings for GTK 3, VTE 2.91
and IBus. From the source directory, after building and running `make check`:

```sh
/usr/bin/python3 tests/native-backspace.py \
  --binary /path/to/build/src/hangul-backspace \
  --schema-dir /path/to/build/src/test-schemas
```

Add `--im-module ibus` to test the direct GTK IBus path. Both variants use
temporary settings, their own D-Bus and Wayland socket, and a virtual keyboard
inside that desktop. They do not change Tilix launchers or desktop settings.

## Installed files

- `~/.local/libexec/hangul-backspace`
- `~/.local/libexec/hangul-backspace-setup`
- `~/.local/share/ibus/component/hangul-backspace.xml`
- `~/.local/share/hangul-backspace/`
- `~/.local/share/applications/hangul-backspace.desktop`
- `~/.local/share/icons/hicolor/{64x64,scalable}/apps/hangul-backspace.*`
- `~/.local/share/glib-2.0/schemas/{org.freedesktop.ibus.engine.hangul-backspace.gschema.xml,gschemas.compiled}`
- `~/.local/share/locale/{ka,ko,zh_CN}/LC_MESSAGES/hangul-backspace.mo`
- `~/.local/share/metainfo/org.freedesktop.ibus.engine.hangul-backspace.metainfo.xml`
