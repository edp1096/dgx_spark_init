# Broad draft-vocabulary evaluation — preregistered plan

Started 2026-09-08 22:18 KST. Initial ETA 90–150 minutes (23:48–00:48 KST).

Question: should the Korean 64K or 128K draft shortlist be the general-purpose default?
This measures draft vocabulary, not context length or the target model's knowledge.

1. Freeze 24 held-out scenarios, prompts, source references and three binary rubric
   items per scenario before either model run. Include five medical topics requested
   by the user, everyday tasks, science/research, repository coding, economics,
   document/contract reading, humanities/language and extended conversation.
2. Keep the existing tokenizer, independently prepared shortlist IDs, target model,
   MTP 3/4, context 65,536, KV capacity 65,536, Mamba slots 18 and temperature zero
   fixed. Use the actual chat template. Four demanding tasks use thinking mode;
   other tasks have thinking disabled. Limits are per-task and identical across modes.
3. Run two passes per mode, reversing task order in pass two. Warm up before timing;
   flush prefix cache before each independent request. One model at a time, 64K then
   128K. This cannot eliminate between-load order or thermal effects; two repeats are
   exploratory evidence, not a population-wide statistical guarantee.
4. Collect native API output token IDs, per-request speculative counts, completion
   time, first-token delay and decode throughput. Count actual emitted tokens outside
   each shortlist, including reasoning separately where possible. No logprob request
   or retokenization is needed for the main coverage metric.
5. Score objective numeric/constraint checks and execute generated Python solutions
   in a separate CPU-only restricted container after both model runs. Review prose
   against the frozen rubric. Review labels are assistant judgments, not clinician,
   lawyer or independent blinded expert assessments. Keyword matches alone are not
   correctness. Record truncation, missing answers and material factual errors.
6. Report paired task ratios, per-domain results, pooled throughput and an unweighted
   task average separately. No invented frequency weights. Report quality separately
   from speed and expose raw outputs. Any suggested default is provisional.

Safety and state: reuse the monitored memory harness; stop the temporary server if
MemAvailable drops below 16 GiB or swap grows over 512 MiB. Preserve the stopped
production model and four support services. No application setting changes, commit,
push or public publication. Coverage excludes live browsing reliability, autonomous
long-running agents, image/audio inputs and concurrent serving; replayed documents
and dialogue are a controlled approximation of those text workloads.

Medical rubric sources: CDC common-cold treatment; NIH/NIAMS gout; NIH/NIDDK
hypoglycemia; NHS allergic rhinitis/decongestants; NCI tumor markers. Sources were
checked 2026-09-08. Individual links and concise paraphrases are stored with tasks.
All cases, budgets, contracts, observations and dialogue are fictional unless a
repository path or external source is explicitly identified.

Instrumentation addition at 22:38 KST, before the first measured response: sample
GPU temperature, utilization, power and SM clock every 10 seconds. No serving
configuration was changed.

Analysis addition at 22:48 KST, with only the first 64K pass in progress: after a
planning response reached its fixed output limit, add a secondary speed summary
restricted to pairs where both modes finish and produce a final answer. Retain all
requests in the primary summary and report truncation separately. Prompts, limits,
rubrics and primary metrics remain frozen; no 128K response had been observed.

Completion follow-up added at 22:59 KST: one 64K coding request exhausted 4096
tokens without producing a final answer. After the primary 64K requests, freeze
the union of its truncated/missing-answer tasks and repeat those tasks once per
mode with 8192 thinking tokens or 3072 non-thinking tokens. Use the same prompts,
temperature and first-repeat seed. Run each supplement before unloading that
mode. Keep this selected, single-repeat follow-up separate from primary metrics.

Interruption and resumption at 23:04 KST: the guard stopped the first load after
16 completed requests when SparkTalk's periodic CPU key-store status helper ran.
There was no OOM. Archived the original logs and kept its completed requests;
the unfinished code_telemetry request is discarded. Resume with a new load and
the same settings. Exempt only the verified image digest, exact status command,
network-none, unprivileged runc helper with no GPU devices. Other new containers
still stop the trial, and memory/swap safeguards remain active. The resumed load
and this background activity are limitations of timing precision.
