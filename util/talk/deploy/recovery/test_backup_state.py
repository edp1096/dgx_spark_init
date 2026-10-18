import argparse
import contextlib
import io
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

import backup_state as recovery


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def fixture(self, role="head"):
        root = self.root / role
        app, data, ssh = root / "app", root / "data", root / "ssh"
        for directory in [app, data / "extra/ssh/keys/.objects", ssh]:
            directory.mkdir(parents=True)
        (data / "extra/ssh/keys/.node-id").write_text(role + "-node")
        (data / "extra/ssh/keys/.objects/test").write_text("synthetic-private-key")
        (ssh / "id_ed25519").write_text("synthetic-bootstrap-key")
        (app / "sparktalk.yaml").write_text("synthetic-config-token")
        (app / "sparktalk-linux-arm64").write_text("binary-must-not-be-in-private-state")
        (app / "sparktalk.db.media").mkdir()
        (app / "sparktalk.db.media/attachment").write_text("attachment")
        (app / "sparktalk.db.browser-key").write_text("pairing-secret")
        (app / "sparktalk.db.plugins/plugin").mkdir(parents=True)
        program = app / "sparktalk.db.plugins/plugin/run"
        program.write_text("#!/bin/sh\nexit 0\n")
        program.chmod(0o755)
        connection = sqlite3.connect(app / "sparktalk.db")
        self.addCleanup(connection.close)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE messages (content TEXT)")
        connection.execute("INSERT INTO messages VALUES ('committed WAL message')")
        connection.commit()
        return argparse.Namespace(role=role, app_dir=app, database=None, data_dir=data,
                                  ssh_dir=ssh, backup=self.root / (role + "-backup"), apps_stopped=True)

    def quiet(self, action, args):
        with contextlib.redirect_stdout(io.StringIO()):
            return action(args)

    def destination(self, original):
        args = argparse.Namespace(**vars(original))
        root = self.root / (original.role + "-restored")
        args.app_dir, args.data_dir, args.ssh_dir = root / "app", root / "data", root / "ssh"
        return args

    def test_head_roundtrip_includes_wal_media_browser_and_plugin(self):
        args = self.fixture()
        self.quiet(recovery.create, args)
        recovery.verify(args.backup, "head")
        dest = self.destination(args)
        self.quiet(recovery.restore, dest)
        with sqlite3.connect(dest.app_dir / "sparktalk.db") as database:
            self.assertEqual(database.execute("SELECT content FROM messages").fetchone()[0], "committed WAL message")
        self.assertEqual((dest.app_dir / "sparktalk.db.media/attachment").read_text(), "attachment")
        self.assertEqual((dest.app_dir / "sparktalk.db.browser-key").read_text(), "pairing-secret")
        self.assertEqual((dest.app_dir / "sparktalk.db.plugins/plugin/run").stat().st_mode & 0o777, 0o700)
        self.assertFalse((args.backup / "sparktalk-linux-arm64").exists())
        self.assertEqual((args.backup / "config").stat().st_mode & 0o777, 0o600)

    def test_two_hosts_keep_distinct_identity_and_reject_wrong_role(self):
        for role in ("head", "worker"):
            args = self.fixture(role)
            self.quiet(recovery.create, args)
            dest = self.destination(args)
            self.quiet(recovery.restore, dest)
            self.assertEqual((dest.data_dir / "extra/ssh/keys/.node-id").read_text(), role + "-node")
        with self.assertRaisesRegex(ValueError, "Wrong host role"):
            recovery.verify(self.root / "head-backup", "worker")

    def test_tampering_rejected_before_restore(self):
        args = self.fixture()
        self.quiet(recovery.create, args)
        (args.backup / "extra/ssh/keys/.objects/test").write_text("damaged")
        dest = self.destination(args)
        with self.assertRaisesRegex(ValueError, "checksum"):
            recovery.restore(dest)
        self.assertFalse(dest.data_dir.exists())

    def test_preexisting_destination_prevents_all_writes(self):
        args = self.fixture()
        self.quiet(recovery.create, args)
        dest = self.destination(args)
        dest.ssh_dir.mkdir(parents=True)
        (dest.ssh_dir / "existing-key").write_text("keep me")
        with self.assertRaisesRegex(ValueError, "overwrite"):
            recovery.restore(dest)
        self.assertFalse(dest.data_dir.exists())
        self.assertEqual((dest.ssh_dir / "existing-key").read_text(), "keep me")

    def test_symlink_failure_leaves_no_complete_backup(self):
        args = self.fixture()
        (args.ssh_dir / "linked-key").symlink_to(args.ssh_dir / "id_ed25519")
        with self.assertRaisesRegex(ValueError, "link"):
            recovery.create(args)
        self.assertFalse(args.backup.exists())

    def test_requires_acknowledgement_and_cannot_backup_into_inputs(self):
        args = self.fixture()
        args.apps_stopped = False
        with self.assertRaisesRegex(ValueError, "Stop all Talk"):
            recovery.create(args)
        args.apps_stopped = True
        args.backup = args.ssh_dir / "backup"
        with self.assertRaisesRegex(ValueError, "outside"):
            recovery.create(args)

    def test_unexpected_files_and_parent_symlinks_are_rejected(self):
        args = self.fixture()
        self.quiet(recovery.create, args)
        (args.backup / "unexpected").write_text("extra")
        with self.assertRaisesRegex(ValueError, "unexpected"):
            recovery.verify(args.backup)
        (args.backup / "unexpected").unlink()
        dest = self.destination(args)
        dest.data_dir.parent.mkdir(parents=True)
        elsewhere = self.root / "elsewhere"
        elsewhere.mkdir()
        dest.data_dir.symlink_to(elsewhere, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink parent"):
            recovery.restore(dest)
        self.assertEqual(list(elsewhere.iterdir()), [])

    def test_documented_cli_backup_verify_restore(self):
        args = self.fixture()
        dest = self.destination(args)
        script = str(Path(recovery.__file__))
        for action, selected in [("backup", args), ("verify", args), ("restore", dest)]:
            command = [sys.executable, "-B", script, action, "--backup", str(args.backup), "--role", "head"]
            if action != "verify":
                command += ["--app-dir", str(selected.app_dir), "--data-dir", str(selected.data_dir),
                            "--ssh-dir", str(selected.ssh_dir), "--apps-stopped"]
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("synthetic-private-key", result.stdout + result.stderr)
        self.assertTrue((dest.app_dir / "sparktalk.db").is_file())


if __name__ == "__main__":
    unittest.main()
