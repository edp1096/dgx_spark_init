# Monitored 4096-token prefill retry

The 4096-token scheduler retry completed successfully twice and is now the
normal serving default. Expert kernels remain bounded at 2048 tokens, with
shared graph I/O, 384 MiB scratch, unchanged 224/128 expert slots and original
weight precision. Both nodes still run `dgx-ds41-stream:b12x8`; routed staging,
predicted prefetch and whole-model graphs remain disabled.

## What the retry establishes

The earlier head-host reboot occurred during startup after the routed-staging
experiment. It left no usable kernel crash record. The new trials started from
the qualified non-staging configuration and added allocation tracing and an
external memory watchdog. Both 4096 boots completed without a host reset,
OOM event or watchdog trip. This demonstrates successful operation under the
new trials; it does not identify the cause of the earlier reset or prove that
its underlying cause has been fixed.

`probe_watchdog.py` samples host availability, memory pressure and the labelled
container's cgroup every half second. It kills only that immutable container ID
if host availability falls below 8 GiB, cgroup use exceeds 98 GiB or a cgroup
OOM event occurs. Normal containers with a different label are protected. A
self-test verified both protection of the unlabelled live service and forced
termination of an owned test container.

The head watcher streams telemetry and logs over management SSH to the worker,
where `probe_peer_recorder.py` fsyncs them to persistent storage. It stops only
the matching experimental worker if the head monitoring connection fails.
`probe_trace.py` records expert allocation and execution-plan boundaries without
printing tensors. `launch_probe.py` checks both ranks are stopped and brings up
observers before launching a labelled experiment. Raw telemetry survives in
`~/.local/state/ds41-probes/20260912-retry4096` and
`~/.local/state/ds41-probes/20260912-final4096` on the two hosts. A watchdog cannot
guarantee recovery from an instantaneous kernel, firmware or storage failure.

## Fresh paired measurement

The earlier `/tmp` fixtures disappeared on reboot. New fixtures were captured
through a temporary SparkTalk instance using an isolated SQLite backup and a
local capture endpoint. The live database and configuration were not changed.
The capture instance was removed afterward; raw fixtures remain mode 0600 in
persistent private user state, outside the repo. Each request contains the real
14-tool registry. Their prompt sizes are 6308 and 12934 tokens.

4096 was measured first, then the 2048 control. Each fixture has an excluded
warmup and two timed trials, with expert slots reset on both ranks and a unique
prefix-cache namespace per request. Every recorded reply had zero cached prompt
tokens, the exact marker and normal stop. Both variants use identical frozen
request bodies, native 2048 kernels, shared buffers and startup tracing. The
4096 trial additionally ran external watchdogs; their overhead is included.

The first retry warmup encountered a measurement-harness error: a worker's cache
reset log was earlier than the head's absolute `--since` timestamp. The reset
itself occurred on both ranks. Log extraction now selects the exact request
reset epoch within node-relative logs, avoiding cross-host clock skew. That
incomplete warmup is retained in `pp-round5-log-window-pilot.log` and excluded.
The full comparison was rerun after the fix.

| Scheduler | 6308-token pp | TTFT | 12934-token pp | TTFT |
| --- | ---: | ---: | ---: | ---: |
| 2048 | 159.38 tok/s | 39.58 s | 190.88 tok/s | 67.76 s |
| **4096** | **192.86 tok/s** | **32.71 s** | **241.29 tok/s** | **53.60 s** |

PP improves 21.0% / 26.4%; TTFT falls 17.4% / 20.9%. The last periodic expert
read snapshots decline from about 201.9 to 160.5 GiB per rank for the short
input and 330.0 to 242.7 GiB for the document. These are periodic snapshots,
not exact final request totals. Timed 4096 ranges were 192.45–193.26 and
241.17–241.41 tok/s. Results are `pp-round5-batch{2048,4096}.json` and
`pp-round5-comparison.json`.

During the first monitored 4096 run, minimum available RAM was 11.12 / 10.97 GiB.
Peak cgroup allocations were 97,962,868,736 / 97,392,644,096 bytes, below the
107,374,182,400-byte limits. Neither rank reported an OOM or guard trip.
See `pp-round5-probe-memory.json`.

## Second boot and actual application checks

The second 4096 boot disabled benchmark control and removed its control files.
Its first inference after health was a real SparkTalk request with the normal
reasoning setting `max`, 65536 context, 8192 output allowance and enabled tools.
It returned the correct marker; the next turn correctly replaced it. Stored
metrics matched SSE and only the validator's temporary session was deleted.

| Actual Talk turn | Input / cached | pp | tg | TTFT |
| --- | ---: | ---: | ---: | ---: |
| First after restart | 6333 / 0 | 179.52 | 12.52 | 35.28 s |
| Follow-up | 6362 / 6144 | 58.84 | 18.00 | 8.87 s |

The previous 2048 first turn was 43.10 seconds. These reasoning-enabled runs
have different generated reasoning lengths and cache states, so their tg
values are not a controlled decode comparison. Follow-up latency is not
uniformly improved by a larger prefill scheduler.

A 54,988-token input with an 8192-token output allowance returned its exact
middle marker with HTTP 200 and normal stop. A following short request also
succeeded. That repetitive input took 39.72 seconds to first content; it checks
capacity and does not establish a general long-input throughput improvement.
After it, available RAM was about 13.16 / 13.06 GiB, with zero cgroup OOM events.
Expert tensors remain 87,836,590,080 bytes per rank; graph I/O is 75,689,160 bytes
and scratch 402,653,184 bytes. Source hashes matched between hosts.

Four deterministic code/prose responses remain byte-identical to the earlier
validated responses and all stop normally. Weighted tg was 18.44 tok/s versus
18.61 previously (-0.9%). This small test with different initial cache states
is a regression check, not a precise causal tg measurement.

Evidence: `pp-round5-sparktalk`, `pp-round5-context`, `pp-round5-runtime-final`,
`pp-round5-decode` and `pp-round5-decode-comparison` in `results/`.

## Serving and further architecture work

`./manage.sh start` now selects scheduler 4096 and expert kernels 2048. For the
previous setting, use `DSV41_MAX_BATCHED_TOKENS=2048`. Scheduler values above
4096 are rejected. Explicit probe tokens still require a live watcher for a
4096 experiment; the standard launch does not enable diagnostic tracing.
The memory guard retains the previous base allowance and adds 2 GiB above the
2048 scheduler allowance, supported by the monitored runs and capacity check.

Antirez's phase-aware/layer-major/fully-resident-encoder proposal is a separate
architecture project. It was reviewed during this retry but is not the source
of these measurements; see [ENCODER_PREFILL_RESEARCH.md](ENCODER_PREFILL_RESEARCH.md).
