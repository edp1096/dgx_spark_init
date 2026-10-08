# ko16k trial suspended after host reboots

The ko16k experiment did not produce any benchmark responses. The last saved worker progress was120/206 weight shards; last valid memory sample was2026-09-09 02:46:42 KST with about31.21GiB available, no additional swap growth and all cgroup OOM counters zero. Last GPU sample02:46:25 was49C and1% utilization. Logging may have stopped before the actual failure; these are not measurements at the exact crash instant.

Journal boot records show fresh boots at02:47:16,02:47:48,02:48:14 and02:48:45. Current uptime reports boot start02:48:41. These records do not establish whether each reset was automatic or manual. The model container is stopped, Exit255, OOMKilled=false, restart=no. No automatic retry performed. Existing completed ko32k/ko64k results remain preserved.

Current boot prints the same corrected, vendor-specific BERT GUID and payload seen previously. Repeated output does not prove a new distinct event or identify the reset cause. No near-failure OOM/panic/Xid/storage-error cause was established from retained trial-boot logs. Original logs are preserved including damaged trailing bytes; recovered valid telemetry is saved separately.

See [diagnostic status](reboot-2026-09-09-0248/status.json), [boot list](reboot-2026-09-09-0248/boot-list.txt), [trial boot kernel](reboot-2026-09-09-0248/trial-boot-kernel.log), [current kernel](reboot-2026-09-09-0248/current-kernel.log). The previous-boot files in that folder refer to the brief02:48 boot; trial-boot files explicitly target the session during ko16k loading.
