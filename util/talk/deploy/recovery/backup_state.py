#!/usr/bin/env python3
"""Separate, private SparkTalk state backups. Requires only Python 3 stdlib."""

import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def files(root):
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError(f"Unsupported link or special file: {path}")
        if path.is_file():
            yield path


def copy_private(source, target):
    if source.is_symlink():
        raise ValueError(f"Symlinks require a separate backup: {source}")
    if source.is_dir():
        list(files(source))  # Validate before copying; do not follow symlinks.
        shutil.copytree(source, target)
        for directory in [target, *(p for p in target.rglob("*") if p.is_dir())]:
            directory.chmod(0o700)
        for path in files(target):
            path.chmod(0o700 if path.stat().st_mode & 0o111 else 0o600)
    elif source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        shutil.copyfile(source, target)
        target.chmod(0o700 if source.stat().st_mode & 0o111 else 0o600)
    else:
        raise ValueError(f"Missing required input: {source}")


def paths(args):
    app = args.app_dir.expanduser().absolute() if args.app_dir else None
    database = args.database.expanduser().absolute() if args.database else (app / "sparktalk.db" if app else None)
    return app, database, args.data_dir.expanduser().absolute(), args.ssh_dir.expanduser().absolute()


def create(args):
    if not args.apps_stopped:
        raise ValueError("Stop all Talk instances and Extra SSH writers first; pass --apps-stopped after doing so")
    app, database, data, ssh = paths(args)
    if args.role == "head" and app is None:
        raise ValueError("Head backup requires --app-dir")
    output = args.backup.expanduser().absolute()
    if output.exists():
        raise ValueError("Backup destination already exists; use a new directory")
    sources = {"extra": data / "extra"}
    if ssh.exists():
        sources["system-ssh"] = ssh
    if (data / "credentials").exists():
        sources["credentials"] = data / "credentials"
    if args.role == "head":
        sources["config"] = app / "sparktalk.yaml"
        if not database.is_file():
            raise ValueError("Database is missing")
        # Include every DB companion, including media, knowledge, plugins and browser key.
        for path in database.parent.glob(database.name + ".*"):
            sources["db" + path.name[len(database.name):]] = path
    protected = [*sources.values(), *([database] if database else [])]
    if any(output == p or p in output.parents for p in protected):
        raise ValueError("Backup output must be outside the input trees")
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(prefix=".sparktalk-backup-", dir=output.parent) as temporary:
        stage = Path(temporary)
        for name, source in sources.items():
            copy_private(source, stage / name)
        if args.role == "head":
            # SQLite backup incorporates committed WAL content; raw copying the DB does not.
            with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as source:
                with closing(sqlite3.connect(stage / "db")) as target:
                    source.backup(target)
                    target.execute("PRAGMA journal_mode=DELETE")
                    if target.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                        raise ValueError("SQLite integrity check failed")
            (stage / "db").chmod(0o600)
        manifest = {
            "schema": 1, "role": args.role, "hostname": socket.gethostname(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "database_name": database.name if args.role == "head" else None,
            "entries": sorted([*sources, *(["db"] if args.role == "head" else [])]),
            "sha256": {str(p.relative_to(stage)): digest(p) for p in files(stage)},
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (stage / "manifest.json").chmod(0o600)
        # Rename only a complete backup. Partial failures leave no apparently valid backup.
        stage.rename(output)
    print(f"Private {args.role} backup created: {output}")
    print("Keep it on separate encrypted storage. No executable or source code was included.")


def verify(root, expected_role=None):
    if root.is_symlink():
        raise ValueError("Backup root must not be a symlink")
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != 1 or manifest.get("role") not in ("head", "worker"):
        raise ValueError("Unsupported backup manifest")
    if expected_role and manifest["role"] != expected_role:
        raise ValueError("Wrong host role: never restore the head backup onto the worker")
    entries = manifest["entries"]
    if not isinstance(entries, list) or any(name not in {"extra", "credentials", "system-ssh", "config", "db"} and not (name.startswith("db.") and "/" not in name and "\\" not in name) for name in entries):
        raise ValueError("Invalid backup entries")
    if set(p.name for p in root.iterdir()) != set(entries) | {"manifest.json"}:
        raise ValueError("Backup entries are missing or unexpected")
    actual = {str(p.relative_to(root)): digest(p) for p in files(root) if p.name != "manifest.json" or p.parent != root}
    if actual != manifest["sha256"]:
        raise ValueError("Backup checksum verification failed")
    if manifest["role"] == "head":
        with closing(sqlite3.connect((root / "db").as_uri() + "?mode=ro&immutable=1", uri=True)) as database:
            if database.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("Backup database is corrupt")
    return manifest


def restore(args):
    if not args.apps_stopped:
        raise ValueError("Restore before starting Talk or Extra SSH; pass --apps-stopped")
    root = args.backup.expanduser().absolute()
    manifest = verify(root, args.role)
    app, database, data, ssh = paths(args)
    if args.role == "head" and app is None:
        raise ValueError("Head restore requires --app-dir")
    if args.role == "head" and database.name != manifest["database_name"]:
        raise ValueError("Keep the original database filename when restoring")
    targets = {}
    for name in manifest["entries"]:
        if name in ("extra", "credentials"):
            target = data / name
        elif name == "system-ssh":
            target = ssh
        elif name == "config":
            target = app / "sparktalk.yaml"
        else:
            target = Path(str(database) + name[2:])
        if target.exists() or target.is_symlink():
            raise ValueError(f"Restore refuses to overwrite an existing path: {target}")
        if any(parent.is_symlink() for parent in target.parents):
            raise ValueError(f"Restore target has a symlink parent: {target}")
        targets[name] = target
    # Validate every destination before writing anything.
    for name, target in targets.items():
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        copy_private(root / name, target)
    print(f"Restored {args.role} state. No service was started and no executable was changed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["backup", "verify", "restore"])
    parser.add_argument("--backup", type=Path, required=True, help="Private backup directory; never a repository directory")
    parser.add_argument("--role", choices=["head", "worker"])
    parser.add_argument("--app-dir", type=Path)
    parser.add_argument("--database", type=Path, help="Defaults to APP_DIR/sparktalk.db")
    parser.add_argument("--data-dir", type=Path, default=Path.home() / ".local/share/sparktalk")
    parser.add_argument("--ssh-dir", type=Path, default=Path.home() / ".ssh")
    parser.add_argument("--apps-stopped", action="store_true", help="Confirm all Talk instances and Extra SSH writers have been stopped")
    args = parser.parse_args()
    os.umask(0o077)
    try:
        if args.action == "verify":
            manifest = verify(args.backup.expanduser().absolute(), args.role)
            print(f"Verified {manifest['role']} backup: checksums and SQLite integrity passed")
        else:
            if not args.role:
                raise ValueError("--role is required")
            (create if args.action == "backup" else restore)(args)
    except (OSError, ValueError, sqlite3.Error, KeyError, TypeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
