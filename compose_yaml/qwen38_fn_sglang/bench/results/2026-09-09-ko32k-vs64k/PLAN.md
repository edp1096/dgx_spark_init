# ko32k vs saved ko64k actual comparison

User authorized running ko32k on 2026-09-09 after reviewing the partial crash-affected trial.
Run the same frozen 24 primary tasks twice, reverse order in repeat 2, same prepared input IDs, token limits, temperature zero and seeds. Use the same image, worker instrumentation, 3 MTP steps/4 draft tokens, context and KV 65536, Mamba slots 18. Only the shortlist changes to 32768. Compare all 48 same (task, repeat) keys against the preserved ko64k primary responses. Do not combine supplementary longer-budget responses.

Shortlist uses original independent calibration corpus and exact existing policy; special/byte/Hangul protection retained. Evaluation prompts and outputs were not used to rank candidates. Offline rebuilding verified the same policy reproduces the original ko64k set exactly.

Original 64k results were collected before the host reboot and include a process restart. New 32k results are from a different session: timing comparisons retain order, thermal and background-load limitations. The prior reboot cause is unresolved.

Save and fsync every completed response; monitor memory every 250 ms and GPU temperature every 10 s. Existing guard aborts below 16 GiB available or over 512 MiB additional swap, unexpected containers or monitor failure. No automatic retry after crash. Run one GPU server only; production remains stopped. Use frozen 3-criterion rubrics and distinguish token-limit/missing final answers from corrupt or interrupted records. CPU grading only after model unload. Report paired quality, speed, missing answers, truncation, acceptance and token coverage; no default setting changes.
