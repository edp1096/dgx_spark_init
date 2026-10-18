# Two-Spark recovery

Start with [RUNBOOK.md](RUNBOOK.md). It covers the current head (`192.168.100.61`)
and worker (`192.168.100.60`), both using account `edp1096`.

1. Before formatting, save the working executable and separate **head** and
   **worker** state backups on storage outside both Sparks.
2. Reinstall DGX OS and check Docker, GPU access and the host network.
3. Restore each host's own private backup, then restore SSH connectivity.
4. Load saved Docker images and model files, and start Talk on the head.
5. Confirm key synchronization and test the required model set.

`check_host.sh` checks prerequisites. `backup_state.py` creates, verifies and
restores private state using Python 3.11+ only. Run its `--help` for options.
Test the tool with `python3 -B -m unittest discover -s . -p 'test_*.py' -v` from
this directory. If using the standalone kit, verify its files with
`sha256sum -c CHECKSUMS.sha256` before recovery.

**Keys stay outside Talk.** This directory and the executable contain no private
keys. Private backups contain keys, credentials and conversation data; keep
them separately on encrypted storage. The tool does not encrypt them.

Backup/restore is tested in temporary directories, including SQLite WAL data,
wrong-host rejection and overwrite protection. A complete two-machine format
and GPU reinstall has not been performed as part of this work.
