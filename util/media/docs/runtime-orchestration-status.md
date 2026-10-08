# SparkMedia runtime orchestration status

Last updated: 2026-09-02 (Asia/Seoul)

## Decision

SparkMedia must own and operate its runtime Compose definitions independently.
It must not execute Compose files from the repository-level `compose_yaml`
directory. Runtime HTTP APIs accept only fixed allowlisted runtime names; they
must never accept a compose path, service name, or shell command from a client.

The intended workflow is:

1. Inspect a single job or the complete submitted batch.
2. Determine every required API/model runtime.
3. Check active work and unified-memory headroom.
4. Keep runtimes with active work protected.
5. Stop reclaimable idle runtimes when required for the next workload.
6. Start the required SparkMedia-owned runtime and wait for health readiness.
7. Run the durable queued job; continue a batch after an individual failure.

## SparkMedia-owned runtime sources

Runtime build contexts were copied into `runtimes/` for:

- Krea image generation
- LTX video generation
- Qwen3 TTS
- Qwen3 ASR
- llama.cpp prompt/translation
- SeedVR2 upscale
- Media Access
- garment extraction
- ReActor face swap
- MiniMax H3 character turntable

Generated outputs, Python caches, benchmark results, llama.cpp build artifacts,
and nested Git metadata were intentionally not copied. MiniMax H3 uses named
input/output volumes in the SparkMedia-owned Compose definition.

## Backend implementation deployed

`internal/server/runtime_manager.go` currently contains an initial implementation:

- fixed runtime allowlist and estimated peak-memory envelopes;
- `GET /api/runtimes`;
- `POST /api/runtimes/{name}/start`;
- `POST /api/runtimes/{name}/stop`;
- active-operation protection;
- idle-runtime reclamation based on `/proc/meminfo`;
- automatic startup before image, video, and speech queue execution;
- a separate test-server boundary so tests never invoke Docker.

The generation queue no longer treats a stopped managed image, video, or speech
engine as an ordinary request failure. It starts the allowlisted runtime first.

Backend tests passed after the test/runtime boundary was added:

```text
go test -timeout 60s ./internal/server ./internal/config
```

This code is built into `dist/media` and the user service runs it with
`dist/media.yaml` and `dist/data`. `GET /api/runtimes` is live. Runtime source
migration remains incomplete for pre-existing containers, as described below.

## Pre-existing runtime instances

Several stopped containers still carry Compose labels from the old
repository-level `compose_yaml` definitions. New runtime starts must be tested
and migrated one by one to the SparkMedia-owned definitions after confirming
that no job is active. The obsolete MiniMax H3 container and duplicate source
directory were removed; its next start will use `runtimes/minimax_h3`.

## Remaining runtime work

- Add managed startup to recognition, prompt/translation, Media Access, and
  synchronous utility routes.
- Plan the whole batch before starting/stopping runtimes instead of considering
  only the next queued job.
- Replace provisional memory envelopes with measured DGX Spark values and
  expose the plan/reason in status output.
- Add runtime status and guarded start/stop controls to the Settings UI.
- Test real stop/start migration using only `runtimes/`.
- Migrate or recreate pre-existing containers from the SparkMedia-owned Compose
  definitions and verify their health endpoints.

## Multi-scene status

The Yeonhwa four-scene example is accepted at its present quality level. ReID
keeps broad face/hair/outfit identity, but the fan shape, embroidery, jewelry,
and other complex accessories can differ between independent Krea generations.

Character-preparation modal changes already deployed before runtime work:

- backdrop clicks no longer close the multi-scene or character-preparation modal;
- selected 360-degree frames can be approved and analyzed with one button;
- the readiness warning names the exact next action;
- bundled character examples contain prepared appearance prompts.

Still pending in the multi-scene UI: show each bundled reference image, story,
and its related scene list together as a visible selectable example set. The
data is linked, but the related scenes are not visible in story mode.
