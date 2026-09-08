#!/usr/bin/env python3
"""Sample host unified-memory pressure. Values are KiB, not per-process GPU allocations."""
import argparse
import csv
import datetime
import signal
import socket
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", required=True)
parser.add_argument("--phase-file", required=True)
parser.add_argument("--interval", type=float, default=1)
parser.add_argument("--seconds", type=float, default=14400)
args = parser.parse_args()
if args.interval <= 0 or args.seconds <= 0:
    parser.error("interval and seconds must be positive")
running = True
def stop(*_):
    global running
    running = False
signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
keys = ["MemTotal", "MemAvailable", "MemFree", "Buffers", "Cached", "SReclaimable", "Shmem", "SwapTotal", "SwapFree", "AnonPages"]
with open(args.output, "x", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["utc", "host", "phase", *keys, "oom_kill", "pswpin", "pswpout"])
    writer.writeheader()
    deadline = time.monotonic() + args.seconds
    while running and time.monotonic() < deadline:
        mem = {k: int(v.split()[0]) for k, v in (line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())}
        vm = dict(line.split() for line in Path("/proc/vmstat").read_text().splitlines())
        phase = Path(args.phase_file).read_text().strip()
        row = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "host": socket.gethostname(), "phase": phase}
        row.update({k: mem[k] for k in keys})
        row.update({k: vm.get(k, "") for k in ["oom_kill", "pswpin", "pswpout"]})
        writer.writerow(row)
        f.flush()
        time.sleep(args.interval)
