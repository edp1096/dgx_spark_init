ibus hangul which solved issue backspace.

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
