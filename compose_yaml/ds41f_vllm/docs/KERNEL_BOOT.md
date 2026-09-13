# Persistent kernel selection — 2026-09-13

Both nodes (192.168.100.60 and .61) now default to
`6.17.0-1032-nvidia`. The user authorized keeping 7.0 out of use after the
one-shot rollback test. No additional reboot was performed.

Each host has `/etc/default/grub.d/zz-ds41-kernel.cfg`:

```sh
GRUB_DEFAULT="Advanced options for DGX OS GNU/Linux>DGX OS GNU/Linux, with Linux 6.17.0-1032-nvidia"
```

`update-grub` regenerated both menus. The generated default selects the existing
normal 6.17 entry, `grub-script-check` passed, and the one-shot `next_entry`
override was cleared. Current kernels remain 6.17. Talk, DS41 and ASR health
checks passed after the change.

Pre-change GRUB files and regeneration logs are retained on each host under
`/var/backups/ds41-kernel-20260913/`.

To restore normal automatic kernel selection later, remove only this override
file and run `update-grub` as root on both hosts. A future validated kernel can
instead be selected by updating the exact menu entry in this file.
