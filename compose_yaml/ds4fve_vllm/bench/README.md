# DeepSeek cache/tool-call update trial

The pinned runtime already contains the opt-in SWA prefix and DSML recovery patches.
This trial changes only those two switches; C128A caching stays enabled in both arms.
The local production `.env` is not edited by the trial.

```sh
python3 run_update_trial.py results/baseline --mode baseline
python3 run_update_trial.py results/candidate --mode candidate
```

Both arms use 128K maximum context, 2 sequences, batch 4096, utilization 0.78,
temperature 0 and 18 requests (six tasks repeated three times). Identical repeated
prompts intentionally exercise prefix caching. Tool calls are inspected, not executed.
Every response is saved immediately. Do not run both arms or other model servers together.
The process stops both model ranks after the trial; support services remain running.

Host boot IDs, available memory, swap and memory-pressure counters are sampled on
both nodes. The run stops below 8 GiB available or 2 GiB free swap, or if generation
adds more than 512 MiB swap relative to the ready state. Loading swap is recorded;
it is not alone evidence of memory exhaustion. Interrupted attempts are retained.
A successful 128K trial does not qualify the production 1M / six-sequence profile.

CPU regression suites from the pinned upstream: `test-dspark-swa-prefix.py` (16 tests)
and `test-dsml-recovery.py` (37 tests). Malformed DSML is covered by those fixtures;
live generations are not guaranteed to produce the malformed wrapper being repaired.
