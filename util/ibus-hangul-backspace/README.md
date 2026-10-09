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

For native Wayland text fields that provide surrounding text, held repeats
after composition becomes empty delete the preceding character through the
input-method protocol. This also works with GTK 3 clients whose key repeat
cannot start after the initial press was consumed. The engine checks support,
UTF-8 validity and cursor bounds before every deletion, and stops requesting
deletion at the beginning of the buffer. Other native events remain unhandled
so the compositor can preserve their source device and repeat metadata.
Direct IBus clients continue to use their input module's key forwarding.

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

Add `--gtk-entry 3` or `--gtk-entry 4` to test an ordinary GTK text field
instead of VTE (the corresponding Python GI binding is required). Add
`--browser /path/to/chrome` to test Chrome/Chromium against a local textarea
using a temporary profile. These modes check composition, committed text,
jamo order, Latin input, and that deletion stops after release.

Add `--vscode /usr/bin/code` to reproduce the VS Code integrated-terminal
issue in an isolated profile. A temporary test extension records terminal
input without launching a user shell; nothing is installed in the host VS Code.
Terminal tests compare every Backspace press after composition ends with
the actual DEL bytes received, so partial deletion loss fails the test.

Known limitation on GNOME 50 Wayland: VS Code's integrated terminal can lose
held Backspace events after Hangul composition becomes empty. This remains
unresolved; the VS Code editor was reported to work normally. The optional
VS Code test currently reproduces this failure. No GNOME extension is included
or required by this package.

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
