# Restore two Sparks without an assistant

Reference layout, checked on 2026-10-18:

| Item | Head | Worker |
| --- | --- | --- |
| Host address | `192.168.100.61` | `192.168.100.60` |
| Backup role | `head` | `worker` |
| Linux account | `edp1096` | `edp1096` |
| Talk app directory | `~/workspace/dgx_spark_init/util/talk/dist` | No production Talk app required |
| Runtime data | `~/.local/share/sparktalk` | `~/.local/share/sparktalk` |
| Model cache | `~/.cache/huggingface` | `~/.cache/huggingface` |

Keep these usernames, home paths and host roles when following the commands
unchanged. Different users, paths or addresses require updating `sparktalk.yaml`
before starting Talk. Model-specific cluster/RDMA networking is additional to
the ordinary SSH link; see the [NVIDIA networking guide](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html).

## 1. Save everything before formatting

Use an external **Linux filesystem** on encrypted storage. Both Sparks' internal
disks will be erased. Replication between the Sparks is not an external backup.
In the examples the external filesystem is mounted at `/mnt/recovery` on each
machine. Connect it to one machine at a time if using a single drive.

On each machine, set these variables in that machine's terminal:

```bash
export RECOVERY=/mnt/recovery/sparktalk-20261018
export REPO="$HOME/workspace/dgx_spark_init"
export TOOL="$REPO/util/talk/deploy/recovery"
export APP="$REPO/util/talk/dist"
mountpoint -q /mnt/recovery || { echo 'Mount the external drive first'; exit 1; }
umask 077
mkdir -p "$RECOVERY/public" "$RECOVERY/private" "$RECOVERY/images"
chmod 700 "$RECOVERY/private"
```

Before state backup, finish active conversations and document jobs. Stop model
sets in Talk, then stop **every Talk process using this data**, including dev
instances. For a user-service installation run:

```bash
systemctl --user stop sparktalk.service
```

For manual launches, identify the processes with `pgrep -af sparktalk`, then use
`kill -TERM PID` for each relevant Talk process. Replace `PID` with its numeric
process ID. Do not run the backup while these apps are writing. On both machines,
stop the Extra SSH container if it exists:

```bash
if docker container inspect sparktalk-extra-ssh >/dev/null 2>&1; then
  docker stop sparktalk-extra-ssh
fi
```

The backup tool requires `--apps-stopped` as an acknowledgement; it does not stop
or detect all applications for you. SQLite uses its backup API to include
committed WAL data. Files such as attachments and key objects still require
stopping writers for a consistent backup.

On **61/head**:

```bash
cp -a "$TOOL" "$RECOVERY/public/recovery"
install -m 755 "$APP/sparktalk-linux-arm64" "$RECOVERY/public/sparktalk-linux-arm64"
python3 "$TOOL/backup_state.py" backup --role head --app-dir "$APP" \
  --backup "$RECOVERY/private/head" --apps-stopped
python3 "$TOOL/backup_state.py" verify --role head --backup "$RECOVERY/private/head"
```

With the standalone recovery kit, extract it and run `sha256sum -c CHECKSUMS.sha256`
inside the extracted directory. Use that directory as `TOOL` when copying the
public recovery files, and use its supplied executable instead of `$APP`'s
executable. The private backup still comes from your installed app/data paths.
The kit itself does not contain your state or keys.

On **60/worker**, copy this public `recovery` directory from the external drive
if the repository is absent, and run:

```bash
python3 "$RECOVERY/public/recovery/backup_state.py" backup --role worker \
  --backup "$RECOVERY/private/worker" --apps-stopped
python3 "$RECOVERY/public/recovery/backup_state.py" verify --role worker \
  --backup "$RECOVERY/private/worker"
```

The head backup contains YAML settings, a checked SQLite snapshot and **all
`sparktalk.db.*` companions**, including media, knowledge, plugin packages and
the browser pairing key. Both host backups contain `data/extra`, optional
`data/credentials` and each host's own `~/.ssh`. No executable or source code is
inside these private backups. The exact preserved directory names are `extra`,
`credentials` and `system-ssh`; they map back to the paths in the table.
Missing optional credentials/SSH directories are omitted. Symlinks fail the
backup instead of silently leaving their targets unprotected. Custom databases
need `--database /absolute/path/to/database` on backup and restore.

The helper preserves each host's `.node-id`, `.manifest.json`, `.objects` and
known-hosts state. Restore **head to head, worker to worker**. Never clone one
host's whole key directory onto the other: duplicate node IDs break replication.

### Save runtime images and model files

Images and model caches are **not** included in `backup_state.py`. Save at least
the images used by your required sets on **each** machine before formatting.
For the SSH helper on either host:

```bash
docker image inspect sparktalk-extra-ssh:0.3.0 >/dev/null
docker save -o "$RECOVERY/images/extra-ssh-0.3.0.tar" sparktalk-extra-ssh:0.3.0
```

Use separate host subdirectories if saving other images that might differ.
List tags with `docker image ls`; save each required tag using `docker save -o
/external/path/image.tar REPOSITORY:TAG`. Do not assume an installed local image
can be pulled from Docker Hub. The working FP4 image is
`local/qwen38-fp4-local:csf-yarn-20261016`.

On each machine, save the full model cache, including direct `edp1096/...`
directories as well as `hub/`. Store it separately for each host:

```bash
# Use head on 61; use worker on 60.
export ROLE=head
mkdir -p "$RECOVERY/models/$ROLE"
rsync -a --info=progress2 "$HOME/.cache/huggingface/" "$RECOVERY/models/$ROLE/"
rsync -acn --itemize-changes "$HOME/.cache/huggingface/" "$RECOVERY/models/$ROLE/"
```

The final comparison should show no differences and exit successfully. Check
that models at custom mount paths are also backed up. An inaccessible file or
broken link is a failed backup, not permission to proceed with formatting.
Re-downloading large public checkpoints is possible but requires network access,
access approvals/tokens where applicable, and the model's documented preparation
steps. It is not a substitute for a verified backup of local-only artifacts.

Preserve local source changes if you intend to rebuild the executable later.
From the repository, the following includes tracked files and untracked files
that are not ignored, without archiving `.git/config` or its remote credentials:

```bash
cd "$REPO"
git ls-files --cached --others --exclude-standard -z > "$RECOVERY/source-files.list"
tar --null -T "$RECOVERY/source-files.list" -czf "$RECOVERY/public/source.tar.gz"
```

Review the file list for your own secrets before sharing the source archive.
Keep the private state directories entirely outside the source checkout. The
working executable remains the simplest recovery path while source fixes have
not yet been committed and published.

## 2. Reinstall and validate both hosts

For a Founders Edition unit, restore DGX OS using the [NVIDIA system recovery procedure](https://docs.nvidia.com/dgx/dgx-spark/system-recovery.html).
Use the manufacturer's recovery media for an OEM Spark. Disconnect the backup
drive while flashing; reconnect it only after OS installation.
Re-create account `edp1096` and assign `.61` to the head and `.60` to the worker
through the OS network settings. Do not reuse these addresses if they belong to
other devices. Keep Talk and Extra SSH stopped until both state restores finish.

The [DGX Spark container runtime guide](https://docs.nvidia.com/dgx/dgx-spark/nvidia-container-runtime-for-docker.html)
describes the preconfigured GPU container stack. If components are missing,
follow that guide and the [Docker Ubuntu installation guide](https://docs.docker.com/engine/install/ubuntu/).
Do not run an unrelated x86 GPU driver installer on Spark. Install missing basic
utilities and allow your user to access Docker:

```bash
sudo apt-get update
sudo apt-get install -y openssh-server rsync python3
sudo systemctl enable --now ssh
sudo usermod -aG docker "$USER"
```

Log out and back in after the Docker group change. Then on **each** machine run:

```bash
bash /mnt/recovery/sparktalk-20261018/public/recovery/check_host.sh
```

Every prerequisite should print `PASS`. Test an actual GPU container too, for
example with the CUDA image used in the NVIDIA runtime guide:

```bash
docker run --rm --gpus all nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04 nvidia-smi
```

## 3. Restore each host's private state

Mount the external drive. Set `RECOVERY`, `REPO` and `APP` as in step 1. Restore
before starting Talk, creating new keys, or configuring key synchronization.
The tool refuses every pre-existing destination, even an empty directory. It
checks all destinations before copying. If the fresh OS has already made
`~/.ssh`, preserve it under a new name first:

```bash
if [ -e "$HOME/.ssh" ]; then
  test ! -e "$HOME/.ssh.after-reinstall" || { echo 'Preserve the existing staging directory first'; exit 1; }
  mv "$HOME/.ssh" "$HOME/.ssh.after-reinstall"
fi
```

On **61/head**:

```bash
python3 "$RECOVERY/public/recovery/backup_state.py" restore --role head \
  --app-dir "$APP" --backup "$RECOVERY/private/head" --apps-stopped
install -m 755 "$RECOVERY/public/sparktalk-linux-arm64" "$APP/sparktalk-linux-arm64"
```

On **60/worker**:

```bash
python3 "$RECOVERY/public/recovery/backup_state.py" restore --role worker \
  --backup "$RECOVERY/private/worker" --apps-stopped
```

Run as `edp1096`, without `sudo`, so restored files belong to the correct account.
The tool verifies per-file SHA-256 and SQLite integrity before restoring. These
checks detect accidental changes, not malicious replacement of a backup and
its manifest. It preserves executable bits for plugin programs and restricts
all restored file access to the current user.

OS SSH **server** host keys are newly generated by the OS and are not in these
backups. At each machine's console, display its new fingerprint:

```bash
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
```

From the head, compare the worker's fingerprint when connecting. Once checked,
remove only the stale entry and reconnect:

```bash
ssh-keygen -R 192.168.100.60
ssh edp1096@192.168.100.60 true
ssh -o BatchMode=yes edp1096@192.168.100.60 true
```

Repeat in the reverse direction if your deployment uses worker-to-head SSH.
If password-free authentication is still missing, use `ssh-copy-id
edp1096@192.168.100.60` from the head after verifying the host fingerprint.
Never disable host-key checking to conceal the mismatch. Talk's separate
trusted-host records may also need to be refreshed using its SSH test/trust UI.

## 4. Load images and model files, then start Talk

On **both** hosts, load the saved SSH image:

```bash
docker load -i "$RECOVERY/images/extra-ssh-0.3.0.tar"
docker image inspect sparktalk-extra-ssh:0.3.0 >/dev/null
```

Load the remaining saved images with `docker load -i IMAGE.tar`. Restore the
matching host's model cache:

```bash
# Use head on 61; use worker on 60.
export ROLE=head
mkdir -p "$HOME/.cache/huggingface"
rsync -a --info=progress2 "$RECOVERY/models/$ROLE/" "$HOME/.cache/huggingface/"
rsync -acn --itemize-changes "$RECOVERY/models/$ROLE/" "$HOME/.cache/huggingface/"
```

From the head, verify the ordinary cluster link:

```bash
bash "$RECOVERY/public/recovery/check_host.sh" edp1096@192.168.100.60
```

Check `sparktalk.yaml` locally before startup: addresses, data/cache paths and
database path must match the restored machine. Set `runtime.auto_start: false`
while validating a new OS. Treat this YAML as private; it may contain tokens.
Start Talk **on the head only**, from its app directory:

```bash
cd "$APP"
./sparktalk-linux-arm64
```

Open `http://192.168.100.61:8585`. For future user-service launches, install
`util/talk/deploy/systemd/sparktalk.service` into `~/.config/systemd/user/`, check
its paths, then use `systemctl --user daemon-reload` and `systemctl --user enable
--now sparktalk.service`. Stop the manual process first to avoid a port conflict.

## 5. Verify recovery in Talk

1. Open **Settings → Features → SSH/keys**. Both replicas should be connected,
   have distinct node IDs, and show matching epoch/version and key counts.
   Use **Sync now**; keep the restored authority unless intentionally handing it off.
2. Check **Settings → System → Support services**. Prepare any missing image
   on the host where the service is placed; start the services you need.
3. Check **Settings → System → Model preparation** for each required set. Restored
   images and weights are reused. See the [FP4-Local preparation notes](../../../../compose_yaml/qwen38_fn_fp4_local_vllm/README.md)
   in the repository for its local checkpoint path and pinned revision.
4. Start the required set. Send a short chat and test semantic search, voice
   and any other features you actually use. A chat-only check does not validate
   a two-GPU model or every auxiliary service.
5. Confirm old conversations, attachments and knowledge documents are present.
   Re-pair the browser extension if its connection was not restored.

| Symptom | Check / recovery |
| --- | --- |
| `No route to host` | Worker power, assigned address, network link and SSH service. |
| Missing `sparktalk-extra-ssh:0.3.0` | Load the saved image on **that host**; do not look for `:latest`. |
| Duplicate node ID | The same private backup was restored twice. Stop both apps and recover each host's own backup; do not delete IDs to bypass the error. |
| Offline authority / conflicting repositories | Do not initialize a new repository or overwrite replicas. Check the correct backups, restore the authority host, then retry. |
| Existing restore destination | Preserve the fresh data elsewhere first; the tool deliberately does not overwrite it. |

Without private backups you can create new keys and register their public keys
again, but erased keys, tokens and conversation data cannot be recovered from
the Talk executable or the public repository.

## Validation limits and sources

The tools are tested with synthetic state in temporary directories. The host
prerequisite check runs on the current machines. No live key store or production
database is overwritten by those tests. This is not evidence of a complete
two-host OS wipe, offline recovery, every model download, or every GPU workload.

Source: [Talk paths and requirements](../../README.md),
[key-store authority rules](../../internal/orchestrator/keystore.go),
[SSH store identity and objects](../../../../compose_yaml/sparktalk_extra/cmd/ssh/keystore.go),
[SSH image/volume definition](../../internal/orchestrator/assets/compose.extra-ssh.yaml).
