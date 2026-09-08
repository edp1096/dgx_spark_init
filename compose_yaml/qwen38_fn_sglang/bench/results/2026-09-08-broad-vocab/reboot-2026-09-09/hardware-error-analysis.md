# 2026-09-09 BERT hardware-error investigation

## Finding
The boot at approximately 00:40:56 KST reports one previous-boot firmware hardware error at 00:41:02. This is separate from systemd-journald's unclean system.journal warning. That warning concerns the logging service's journal file, not an explicit SSD hardware failure or EXT4 filesystem-journal failure.

The BERT record is classified `corrected` at both event and section level. The section GUID is `3c1e3f4b-1e1a-43df-af28-59820e958e3c`, length 62 bytes, with an `MTKID` marker. Linux reports the section as unknown; no publicly documented decoder was established. The record does not identify an NVMe device, a DIMM/ECC address, or a GPU Xid. Corrected classification does not establish that this record caused the reboot, or prove the platform healthy.

## Evidence and limits
- Raw evidence: current-kernel.log, lines 837 onward. Previous boot kernel log has no corresponding BERT error report.
- Current and previous saved kernel logs have no matching NVMe timeout/reset/failure, I/O error, EXT4-fs error, PCIe AER error, or DOE failure in the targeted scan. AER-enabled and NVMe initialization messages are normal discovery messages.
- SSD: Oyen Digital PCIe SSD, Phison controller, firmware EVFM01.1. SMART and NVMe error-log reads were denied by device permissions, so SSD health is not certified.
- Firmware from fwupdmgr: ASUS EC 0x02000007, SoC firmware 0x03000008, USB PD 0x00000516. See firmware-summary.json. No firmware changes performed.
- Access to raw BERT tables and root pstore remains unavailable under current account permissions. Kernel-printed payload is preserved.
- Available pre-crash samples showed GPU 63 C and approximately 25.94 GiB sampled memory usage; these are not continuous measurements and do not rule out a transient fault.

## Source interpretation
Linux BERT driver explicitly reads errors left from a prior boot. Linux CPER code prints unrecognized section GUIDs and raw payloads; `unknown` is a missing decoder, not a component diagnosis.
- https://github.com/torvalds/linux/blob/master/drivers/acpi/apei/bert.c
- https://github.com/torvalds/linux/blob/master/drivers/firmware/efi/cper.c

A NVIDIA forum user reports the same GUID on another machine that rebooted, and interprets an MTKID record in a different ASUS GX10 case as a USB PD/EC issue. These are third-party interpretations hosted on NVIDIA's forum, not a verified manufacturer decoding specification. Bytes rendering as ASCII `PD` alone cannot identify a Power Delivery fault. Power/EC remains a hypothesis, not the diagnosed cause.
- https://forums.developer.nvidia.com/t/dgx-spark-reboots-every-20-minutes/367448/8
- https://forums.developer.nvidia.com/t/another-asus-gx10-problem/371201/9

The defensible diagnosis is a firmware-reported, corrected, manufacturer-specific platform hardware event of unresolved component origin. Decoding this GUID and payload requires ASUS/NVIDIA's format knowledge; pstore and privileged NVMe logs could independently narrow the reboot cause. Benchmark remains stopped.
