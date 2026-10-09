#!/usr/bin/python3
"""Optional GNOME Wayland/VTE regression test in a private desktop session.

Requires gnome-shell with --headless, dbus-run-session, dconf, and Python GI
bindings for Gtk 3, Vte 2.91 and IBus. No physical input devices are opened.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET


def wait_until(check, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = check()
        if result:
            return result
        time.sleep(0.1)
    raise RuntimeError("Timed out waiting for private session")


def terminal(root):
    import gi
    gi.require_version("Gtk", "3.0")
    gi.require_version("Vte", "2.91")
    from gi.repository import Gtk, Vte, GLib

    window = Gtk.Window(title="Isolated Backspace Test")
    window.set_default_size(900, 650)
    widget = Vte.Terminal()
    window.add(widget)
    widget.spawn_sync(Vte.PtyFlags.DEFAULT, None,
                      [sys.executable, str(Path(__file__).resolve()), "--reader", str(root)],
                      None, GLib.SpawnFlags.DEFAULT, None, None, None)
    window.show_all()
    window.present()
    widget.grab_focus()
    (root / "ready").touch()
    Gtk.main()


def reader(root):
    import tty
    tty.setraw(0)
    with (root / "bytes").open("ab", buffering=0) as output:
        while True:
            data = os.read(0, 1024)
            if not data:
                return
            output.write(data)


def session(root, binary, im_module):
    import gi
    gi.require_version("IBus", "1.0")
    from gi.repository import Gio, GLib, IBus

    processes = []
    logs = []

    def start(args, name):
        log = (root / (name + ".log")).open("w")
        logs.append(log)
        process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
        processes.append(process)
        return process

    def setting(schema, key, value):
        subprocess.run(["gsettings", "set", schema, key, value], check=True)

    def received():
        path = root / "bytes"
        return path.read_bytes() if path.exists() else b""

    try:
        # Registry and settings belong exclusively to this dbus-run-session.
        components = root / "share/ibus/component"
        components.mkdir(parents=True)
        component = ET.Element("component")
        for name, value in {"name": "org.freedesktop.IBus.HangulBackspace",
                            "exec": str(binary) + " --ibus",
                            "description": "Backspace regression test"}.items():
            ET.SubElement(component, name).text = value
        engine = ET.SubElement(ET.SubElement(component, "engines"), "engine")
        for name, value in {"name": "hangul-backspace", "longname": "Backspace test",
                            "language": "ko", "layout": "us"}.items():
            ET.SubElement(engine, name).text = value
        ET.ElementTree(component).write(components / "test.xml", encoding="utf-8")
        os.environ["IBUS_COMPONENT_PATH"] = str(components) + ":/usr/share/ibus/component"
        os.environ["HANGUL_BACKSPACE_TRACE"] = "1"
        setting("org.freedesktop.ibus.engine.hangul-backspace", "initial-input-mode", "hangul")
        setting("org.gnome.desktop.input-sources", "sources", "[('ibus', 'hangul-backspace')]")
        setting("org.gnome.desktop.peripherals.keyboard", "delay", "200")
        setting("org.gnome.desktop.peripherals.keyboard", "repeat-interval", "30")
        shell = start(["gnome-shell", "--headless", "--wayland", "--no-x11",
                       "--virtual-monitor", "1024x768", "--wayland-display", "hangul-test"], "shell")
        wait_until(lambda: (root / "run/hangul-test").exists() or
                   (shell.poll() is not None and sys.exit("Private compositor exited")))
        os.environ["WAYLAND_DISPLAY"] = "hangul-test"
        os.environ["GTK_IM_MODULE"] = im_module
        start([sys.executable, str(Path(__file__).resolve()), "--terminal", str(root)], "terminal")
        wait_until(lambda: (root / "ready").exists())
        bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)

        def call(path, interface, method, args=None):
            return bus.call_sync("org.gnome.Mutter.RemoteDesktop", path, interface,
                                 method, args, None, Gio.DBusCallFlags.NONE, 3000, None)

        def create_remote():
            try:
                return call("/org/gnome/Mutter/RemoteDesktop",
                            "org.gnome.Mutter.RemoteDesktop", "CreateSession")
            except GLib.Error:
                return None

        remote_path = wait_until(create_remote).unpack()[0]

        def remote(method, args=None):
            return call(remote_path, "org.gnome.Mutter.RemoteDesktop.Session", method, args)

        remote("Start")

        def key(code, pressed):
            remote("NotifyKeyboardKeycode", GLib.Variant("(ub)", (code, pressed)))

        def tap(code):
            key(code, True)
            time.sleep(0.08)
            key(code, False)
            time.sleep(0.08)

        def hold():
            key(14, True)
            time.sleep(1.2)
            key(14, False)
            time.sleep(0.3)
            snapshot = received()
            time.sleep(0.3)
            assert received() == snapshot, "Deletion continued after release"

        time.sleep(3)
        tap(1)  # Dismiss the private desktop's initial overview.
        IBus.init()
        ibus = IBus.Bus.new()
        assert ibus.is_connected(), "Private IBus did not start"
        # Register this exact binary directly, regardless of any registry cache.
        start([str(binary)], "engine")
        time.sleep(0.5)
        assert ibus.set_global_engine("hangul-backspace")
        time.sleep(0.5)

        for label, keys, prefix in [
                ("composing", [19, 37] * 3, "가가".encode()),
                ("committed", [19, 37] * 3 + [57], "가가가 ".encode())]:
            offset = len(received())
            for code in keys:
                tap(code)
            hold()
            data = received()[offset:]
            assert data.startswith(prefix), (label, data)
            suffix = data[len(prefix):]
            assert len(suffix) >= 5 and suffix == b"\x7f" * len(suffix), (label, data)
            print(f"PASS {im_module}/{label}: {len(suffix)} deletions", flush=True)
            offset = len(received())
            tap(14)
            time.sleep(0.2)
            assert received()[offset:] == b"\x7f", "Fresh press must delete once"

        for label, keys, expected in [
                ("jamo-order", [38, 31, 57], "ㅣㄴ ".encode()),
                ("normal-composition", [31, 38, 57], "니 ".encode())]:
            offset = len(received())
            for code in keys:
                tap(code)
            time.sleep(0.2)
            assert received()[offset:] == expected, (label, received()[offset:])
            print(f"PASS {im_module}/{label}", flush=True)

        # Latin mode must retain the terminal's normal key handling as well.
        key(42, True)
        tap(57)
        key(42, False)
        time.sleep(0.2)
        offset = len(received())
        for code in [19, 37] * 2:
            tap(code)
        hold()
        data = received()[offset:]
        assert data.startswith(b"rkrk"), data
        suffix = data[4:]
        assert len(suffix) >= 5 and suffix == b"\x7f" * len(suffix), data
        print(f"PASS {im_module}/latin: {len(suffix)} deletions", flush=True)
        remote("Stop")
        shell_log = (root / "shell.log").read_text()
        assert "CLUTTER_IS_INPUT_DEVICE" not in shell_log, "Compositor rejected forwarded keys"
    finally:
        for process in reversed(processes):
            process.terminate()
        for process in reversed(processes):
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in logs:
            log.close()


def main():
    if len(sys.argv) == 3 and sys.argv[1] in ("--terminal", "--reader"):
        {"--terminal": terminal, "--reader": reader}[sys.argv[1]](Path(sys.argv[2]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--schema-dir", type=Path, required=True)
    parser.add_argument("--im-module", choices=("wayland", "ibus"), default="wayland")
    parser.add_argument("--session", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.session:
        session(args.session, args.binary, args.im_module)
        return
    binary = args.binary.resolve(strict=True)
    schemas = args.schema_dir.resolve(strict=True)
    if not (schemas / "gschemas.compiled").is_file():
        parser.error("--schema-dir must contain gschemas.compiled (run make check first)")
    for command in ("gnome-shell", "dbus-run-session", "gsettings"):
        if not shutil.which(command):
            parser.error("Missing dependency: " + command)
    root = Path(tempfile.mkdtemp(prefix="hangul-backspace-native."))
    env = os.environ.copy()
    for name, directory in [("XDG_RUNTIME_DIR", "run"), ("XDG_CONFIG_HOME", "config"),
                            ("XDG_CACHE_HOME", "cache"), ("XDG_DATA_HOME", "share")]:
        path = root / directory
        path.mkdir(mode=0o700)
        env[name] = str(path)
    for name in ("DISPLAY", "WAYLAND_DISPLAY", "IBUS_ADDRESS"):
        env.pop(name, None)
    # Set these BEFORE starting D-Bus, so activated dconf services are isolated.
    env.update(GSETTINGS_BACKEND="dconf", GSETTINGS_SCHEMA_DIR=str(schemas),
               IBUS_ADDRESS_FILE=str(root / "ibus.address"),
               LIBGL_ALWAYS_SOFTWARE="1", XDG_SESSION_TYPE="wayland",
               XDG_CURRENT_DESKTOP="GNOME")
    print("Private test session and logs:", root, flush=True)
    with (root / "session.log").open("w") as log:
        result = subprocess.run([
            "dbus-run-session", "--", sys.executable, str(Path(__file__).resolve()),
            "--session", str(root), "--binary", str(binary), "--schema-dir", str(schemas),
            "--im-module", args.im_module], env=env, stdout=log, stderr=log, timeout=60)
    output = (root / "session.log").read_text()
    if result.returncode:
        print(output[-4000:])
        raise SystemExit(result.returncode)
    for line in output.splitlines():
        if line.startswith("PASS "):
            print(line)


if __name__ == "__main__":
    main()
