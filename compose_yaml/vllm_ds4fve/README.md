# DeepSeek V4 Flash Vision Exp — 2 Spark measurement

Pinned MiaAI recipe; run from this directory with `./manage.sh`.

- Upstream: https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark
- Commit: `b1b8dfb84d855a166a05f32a90c269118b208987`
- Image: `ghcr.io/anemll/dspark-vllm-gx10:0.1.1@sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8`
- Official weights: `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp@86f746b36186f0e567729a5c06a8c918caba82a9`
- Config: `.env` (private, ignored). Both ranks use local SSD caches.
- Initial API: `http://127.0.0.1:8888/v1` on head.
- TP 2, context 1048576, sequences 6, batch 8192, GPU utilization 0.835, DSpark k6.
- No Extra, ASR, TTS services in this deployment.

To reconstruct the ignored upstream checkout, clone the upstream URL into
`upstream`, checkout the exact commit above, and copy `.env.dspark.example`
to `.env.dspark`. Set head/worker network and cache paths before use.
The current machine uses head 192.168.100.61 / 10.200.0.1 and worker
192.168.100.60 / 10.200.0.2, interface enp1s0f1np1, HCA rocep1s0f1.

## Abliterated variant

The upstream launcher supports `ABLITERATED=1`, with separate checkpoint
`drowzeys/keys-DeepSeekV4Flash-Vision-EXP-ablit` and revision key
`DSPARK_REVISION_ABLITERATED`. Current prepared configuration pins
`48095b3452a17f3e3ae8f77892399389c45de9e1` from the upstream overlay script.
Runtime validation is separate from checkpoint preparation. Gated access requires
account agreement and a token with access. Do not put tokens in this repository.
Switching requires stopping both ranks, preparing the selected checkpoint on
both nodes, and starting again. It does not abliterate official weights at runtime.
This recipe uses selective tensor ranges and verifies reconstructed shard hashes;
it does not download each changed shard in full.
Equal architecture suggests similar RAM requirements; measure rather than assume.

## 공통 관리 명령

`manage.sh setup|image|model|start|stop|restart|status|logs|validate`를 사용한다.
`setup`에 모델 준비가 포함된다. 설정은 `.env`/`env.sample`, 모델 종류는
`MODEL_VARIANT=official|abliterated`이며 `setup`/`model`에 `--official` 또는
`--abliterated`를 지정할 수 있다. HF_TOKEN 환경변수 또는 `--ask-token` 숨김 입력을
사용한다. 상세 규칙은 [공통 CLI](../runtime-common/README.md)를 참조한다.

## Selective abliterated preparation

`model --abliterated` requires the pinned official model in each host's cache.
It fetches safetensors headers and L10–35 `attn.wo_b.weight` / scale byte ranges,
then reconstructs separate blobs using the local original. Every reconstructed
shard must match the published abliterated SHA-256 before it becomes usable.
Header/layout or hash mismatches stop preparation; there is no automatic full-shard fallback.
The worker receives the small patch bundle and reconstructs from its own original
cache. Original blobs are never modified. The final checkpoint still occupies
space for changed shards locally; this optimization reduces network downloads,
not necessarily disk usage. Partial downloads left by the old downloader are
preserved but are no longer needed by selective preparation.
