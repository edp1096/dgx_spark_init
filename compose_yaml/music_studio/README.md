# Talk music generation

One ARM64/CUDA 13 runtime provides two ACE-Step instrumental music choices:

| Request model | Weights | Generation |
| --- | --- | --- |
| `ace-turbo` | ACE-Step 1.5 bundled Turbo | 8 steps, shift 3, 1.7B music LM |
| `ace-xl` | ACE-Step 1.5 XL Base | 50 steps, CFG 7, 1.7B music LM |

The `music_generate` Talk tool attaches a WAV to the current conversation.
Shorts plans use `music.kind="generated"`, `model`, `prompt`, `seed`, optional
`bpm`, and `gain`. The director chooses generated music for requested BGM.
Voice ducking, loudness normalization and fades are applied when mixing a video.
Unchanged revisions reuse the original generated track and provenance.

Requests accept 10–120 seconds. Generation runs in a child process; completion,
failure and client cancellation release its CUDA allocations. Talk serializes
GPU jobs and checks actual host memory. It may temporarily stop an idle image
runtime and restore it after music generation. The conversation LLM stays up.
Port 8727 binds to localhost. This private worker requires Talk's
admission controller; it is not a public endpoint.

Outputs record the model/source revisions, seed, actual length, rate, channels,
runtime, CUDA peaks, waveform measurements and SHA-256. Instrumental prompting
does not guarantee absence of vocals. Waveform checks do not judge musical
quality. Shorts checks actual audio coverage, internal pauses and trailing silence.
It generates an extended source (25% extra initially, then 50%/75% on retries,
capped at 120 seconds) and checks the requested excerpt before trimming. It
tries at most three seeds if waveform continuity fails, retaining each candidate
and its measured rejection reason. The final
record identifies the actual accepted seed. There is no automatic synthetic-note,
silence-padding or looping fallback.

Build with `Dockerfile.ace`. The supported source revision is pinned in that
file. Talk packages this build context and downloads the two pinned ACE-Step
repositories defined in `model_assets.go` under `$HF_CACHE/edp1096/music-studio`.
Weight mounts are read-only. ACE-Step's automatic Python model-code
synchronization operates on per-request copies, preserving downloaded files.

Upstream sources:

- https://github.com/ace-step/ACE-Step-1.5 (MIT)
- https://huggingface.co/ACE-Step/Ace-Step1.5
- https://huggingface.co/ACE-Step/acestep-v15-xl-base
