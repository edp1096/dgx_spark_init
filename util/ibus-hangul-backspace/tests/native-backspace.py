#!/usr/bin/python3
"""Optional GNOME Wayland input regression test in a private desktop session.

Requires gnome-shell with --headless, dbus-run-session, dconf, and Python GI
bindings for Gtk 3, Vte 2.91 and IBus. No physical input devices are opened.
GTK entry and Chrome/Chromium modes exercise ordinary graphical text fields.
"""
import argparse
import json
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


def entry(root, version):
    import gi
    gi.require_version("Gtk", version + ".0")
    from gi.repository import Gtk, GLib
    window = Gtk.Window(title="Isolated GTK Entry Test")
    window.set_default_size(900, 300)
    widget = Gtk.Entry()
    preedit = [""]

    def report(*args):
        value = widget.get_text()
        cursor = widget.get_position()
        value = value[:cursor] + preedit[0] + value[cursor:]
        path = root / "browser-state.next"
        path.write_text(json.dumps({"value": value}))
        path.replace(root / "browser-state.json")

    def composing(widget, text):
        preedit[0] = text
        report()

    widget.connect("changed", report)
    delegate = widget.get_delegate() if version == "4" else widget
    delegate.connect("preedit-changed", composing)
    if version == "4":
        window.set_child(widget)
    else:
        window.add(widget)
        window.show_all()
    window.present()
    widget.grab_focus()
    report()
    (root / "ready").touch()
    GLib.MainLoop().run()


def browser_page(root):
    from http.server import BaseHTTPRequestHandler, HTTPServer

    page = """<!doctype html><meta charset=utf-8>
    <textarea id=field autofocus style='width:90vw;height:70vh;font-size:30px'></textarea>
    <script>
    let seq=0; const field=document.querySelector('#field');
    function report(event) {
      fetch('/state', {method:'POST', body:JSON.stringify({seq:++seq,
        value:field.value, event:event.type, composing:event.isComposing || false})});
    }
    ['input','compositionend','focus'].forEach(name => field.addEventListener(name,report));
    field.focus(); report({type:'ready'});
    </script>""".encode()

    class Handler(BaseHTTPRequestHandler):
        latest = 0

        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(page)

        def do_POST(self):
            state = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if state['seq'] > Handler.latest:
                Handler.latest = state['seq']
                path = root / 'browser-state.next'
                path.write_text(json.dumps(state))
                path.replace(root / 'browser-state.json')
            self.send_response(204)
            self.end_headers()

    server = HTTPServer(('127.0.0.1', 0), Handler)
    (root / 'browser.url').write_text('http://127.0.0.1:%d/' % server.server_port)
    server.serve_forever()


def vscode_extension(root):
    """Record actual integrated-terminal input without running a user shell."""
    extension = root / "test-extension"
    extension.mkdir()
    (extension / "package.json").write_text(json.dumps({
        "name": "backspace-test", "publisher": "local", "version": "0.0.1",
        "engines": {"vscode": "^1.80.0"}, "main": "extension.js",
        "activationEvents": ["onStartupFinished"]}))
    (extension / "extension.js").write_text("""
const vscode = require('vscode');
const fs = require('fs');
const path = require('path');
exports.activate = async function(context) {
  const root = path.dirname(context.extensionPath);
  const write = new vscode.EventEmitter();
  const terminal = vscode.window.createTerminal({name: 'Isolated Backspace Test', pty: {
    onDidWrite: write.event,
    open() { write.fire('Private input test\\r\\n'); },
    close() {},
    handleInput(data) { fs.appendFileSync(path.join(root, 'bytes'), data); }
  }});
  context.subscriptions.push(terminal, write);
  terminal.show();
  await vscode.commands.executeCommand('workbench.action.terminal.focus');
  fs.writeFileSync(path.join(root, 'ready'), '');
};
""")
    profile = root / "code-profile/User"
    profile.mkdir(parents=True)
    (profile / "settings.json").write_text(json.dumps({
        "workbench.startupEditor": "none", "update.mode": "none",
        "telemetry.telemetryLevel": "off", "security.workspace.trust.enabled": False,
        "terminal.integrated.gpuAcceleration": "off"}))
    return extension


def session(root, binary, im_module, browser=None, entry_version=None, vscode=None):
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
        if browser or entry_version:
            path = root / "browser-state.json"
            return json.loads(path.read_text())["value"].encode() if path.exists() else b""
        path = root / "bytes"
        return path.read_bytes() if path.exists() else b""

    def check_compositor():
        log = (root / "shell.log").read_text()
        for marker in ("CLUTTER_IS_INPUT_DEVICE",
                       "meta_wayland_text_input_focus_delete_surrounding",
                       "clutter_input_focus_delete_surrounding"):
            assert marker not in log, "Compositor rejected input: " + marker

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
        if browser:
            start([sys.executable, str(Path(__file__).resolve()), "--browser-page", str(root)], "page")
            wait_until(lambda: (root / "browser.url").exists())
            start([str(browser), "--ozone-platform=wayland", "--no-first-run",
                   "--no-default-browser-check", "--disable-background-networking",
                   "--disable-extensions", "--password-store=basic",
                   "--user-data-dir=" + str(root / "browser-profile"),
                   (root / "browser.url").read_text()], "browser")
            wait_until(lambda: (root / "browser-state.json").exists())
        elif vscode:
            extension = vscode_extension(root)
            start([str(vscode), "--new-window", "--ozone-platform=wayland", "--password-store=basic",
                   "--user-data-dir=" + str(root / "code-profile"),
                   "--extensions-dir=" + str(root / "code-extensions"),
                   "--extensionDevelopmentPath=" + str(extension),
                   "--skip-welcome", "--skip-release-notes", "--disable-workspace-trust"], "code")
            wait_until(lambda: (root / "ready").exists(), timeout=40)
        elif entry_version:
            start([sys.executable, str(Path(__file__).resolve()), "--entry" + entry_version,
                   str(root)], "entry")
            wait_until(lambda: (root / "ready").exists())
        else:
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

        def hold(code=14):
            key(code, True)
            time.sleep(1.2)
            key(code, False)
            time.sleep(0.3)
            snapshot = received()
            time.sleep(0.3)
            assert received() == snapshot, "Input continued after release"

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

        if browser or entry_version:
            client = "browser" if browser else "gtk" + entry_version
            for label, keys, expected in [
                    ("composing", [19, 37] * 8, "가" * 8),
                    ("committed", [19, 37] * 8 + [57], "가" * 8 + " ")]:
                for code in keys:
                    tap(code)
                time.sleep(0.3)
                assert received().decode() == expected, (label, received())
                hold()
                assert received() == b"", (label, received())
                print(f"PASS {client}/{label}: held Backspace emptied the field", flush=True)
            for label, keys, expected in [
                    ("jamo-order", [38, 31, 57], "ㅣㄴ "),
                    ("normal-composition", [31, 38, 57], "니 ")]:
                for code in keys:
                    tap(code)
                time.sleep(0.2)
                assert received().decode() == expected, (label, received())
                hold()
                assert received() == b"", (label, received())
                print(f"PASS {client}/{label}", flush=True)
            key(42, True)
            tap(57)
            key(42, False)
            for code in [19, 37] * 8:
                tap(code)
            time.sleep(0.2)
            assert received() == b"rk" * 8, received()
            hold()
            assert received() == b"", received()
            print(f"PASS {client}/latin", flush=True)
            remote("Stop")
            check_compositor()
            return

        client = "vscode" if vscode else im_module
        for label, keys, prefix in [
                ("composing", [19, 37] * 3, "가가".encode()),
                ("committed", [19, 37] * 3 + [57], "가가가 ".encode())]:
            offset = len(received())
            for code in keys:
                tap(code)
            trace_offset = (root / "engine.log").stat().st_size
            hold()
            data = received()[offset:]
            assert data.startswith(prefix), (label, data)
            suffix = data[len(prefix):]
            assert len(suffix) >= 5 and suffix == b"\x7f" * len(suffix), (label, data)
            events = (root / "engine.log").read_text()[trace_offset:].splitlines()
            presses = [event for event in events if "backspace event " in event and "release=0 " in event]
            composing = sum("composing=1 " in event for event in presses)
            assert len(suffix) == len(presses) - composing, (label, len(suffix), len(presses), composing)
            print(f"PASS {client}/{label}: {len(suffix)} deletions", flush=True)
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
            print(f"PASS {client}/{label}", flush=True)

        if vscode:
            for label, taps in [("held-consonant", 0), ("tapped-then-held-consonant", 2)]:
                offset = len(received())
                for _ in range(taps):
                    tap(30)  # a -> ㅁ
                hold(30)
                tap(57)  # Commit the last preedit before counting PTY input.
                data = received()[offset:].decode()
                assert data.endswith(" ") and set(data[:-1]) == {"ㅁ"}, (label, data)
                count = len(data) - 1 - taps
                assert 25 <= count <= 40, (label, count)
                print(f"PASS vscode/{label}: {count} repeated consonants", flush=True)
            for code in [19, 37] * 3:
                tap(code)
            tap(14)
            tap(14)  # Empty composition with two distinct presses first.
            offset = len(received())
            trace_offset = (root / "engine.log").stat().st_size
            hold()
            presses = [line for line in (root / "engine.log").read_text()[trace_offset:].splitlines()
                       if "backspace event " in line and "release=0 " in line]
            assert received()[offset:] == b"\x7f" * len(presses), received()[offset:]
            print(f"PASS vscode/tapped-then-held-backspace: {len(presses)} deletions", flush=True)
            offset = len(received())
            tap(19)
            tap(37)
            key(29, True)
            tap(30)  # Ctrl+A after an active composition.
            key(29, False)
            time.sleep(0.2)
            assert received()[offset:] == "가".encode() + b"\x01", received()[offset:]
            print("PASS vscode/control-after-composition", flush=True)

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
        print(f"PASS {client}/latin: {len(suffix)} deletions", flush=True)
        remote("Stop")
        check_compositor()
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
    if len(sys.argv) == 3 and sys.argv[1] in ("--entry3", "--entry4"):
        entry(Path(sys.argv[2]), sys.argv[1][-1])
        return
    if len(sys.argv) == 3 and sys.argv[1] in ("--terminal", "--reader", "--browser-page"):
        {"--terminal": terminal, "--reader": reader,
         "--browser-page": browser_page}[sys.argv[1]](Path(sys.argv[2]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--schema-dir", type=Path, required=True)
    parser.add_argument("--im-module", choices=("wayland", "ibus"), default="wayland")
    parser.add_argument("--browser", type=Path,
                        help="Test Chrome/Chromium instead of VTE, using a temporary profile")
    parser.add_argument("--gtk-entry", choices=("3", "4"),
                        help="Test a GTK entry instead of VTE")
    parser.add_argument("--vscode", type=Path,
                        help="Test VS Code's integrated terminal with an isolated test extension")
    parser.add_argument("--session", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if sum(bool(mode) for mode in (args.browser, args.gtk_entry, args.vscode)) > 1:
        parser.error("Choose only one of --browser, --gtk-entry or --vscode")
    if args.session:
        session(args.session, args.binary, args.im_module, args.browser, args.gtk_entry,
                args.vscode)
        return
    binary = args.binary.resolve(strict=True)
    schemas = args.schema_dir.resolve(strict=True)
    if not (schemas / "gschemas.compiled").is_file():
        parser.error("--schema-dir must contain gschemas.compiled (run make check first)")
    for command in ("gnome-shell", "dbus-run-session", "gsettings"):
        if not shutil.which(command):
            parser.error("Missing dependency: " + command)
    browser = args.browser.resolve(strict=True) if args.browser else None
    vscode = args.vscode.resolve(strict=True) if args.vscode else None
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
    extra = ["--browser", str(browser)] if browser else []
    if args.gtk_entry:
        extra += ["--gtk-entry", args.gtk_entry]
    if vscode:
        extra += ["--vscode", str(vscode)]
    with (root / "session.log").open("w") as log:
        result = subprocess.run([
            "dbus-run-session", "--", sys.executable, str(Path(__file__).resolve()),
            "--session", str(root), "--binary", str(binary), "--schema-dir", str(schemas),
            "--im-module", args.im_module] + extra, env=env, stdout=log, stderr=log, timeout=90)
    output = (root / "session.log").read_text()
    if result.returncode:
        print(output[-4000:])
        raise SystemExit(result.returncode)
    for line in output.splitlines():
        if line.startswith("PASS "):
            print(line)


if __name__ == "__main__":
    main()
