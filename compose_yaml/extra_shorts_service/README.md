Ask Talk to make a short. Progress and the finished MP4 appear in the same conversation. Ask in chat to check, stop or retry a job; no separate screen is needed.

720×1280, 24fps. Model-authored layouts, Fish narration, timed captions and HyperFrames composition. AI images/video, declarative Manim, imported/original Lottie or SVG, and model-generated BGM are optional. Jobs survive browser disconnects and worker restarts; canceled or failed jobs can resume. Publishing is not included.

Source: [HyperFrames](https://github.com/heygen-com/hyperframes), [Manim](https://github.com/ManimCommunity/manim), [Lottie](https://github.com/airbnb/lottie-web), [Drama3](https://huggingface.co/fishaudio/drama-3).
Fish commercial use follows its [Community License](https://huggingface.co/fishaudio/drama-3/blob/main/LICENSE), including its revenue threshold and hosted-service terms. Built with Fish Audio.

The studio now uses an editable production plan and model-authored CSS, HTML/SVG overlays and GSAP. `shorts_assets` searches/imports free LottieFiles source pages with license metadata, or stores original Lottie/SVG. `shorts_templates` stores and inspects actual template code. `shorts_project` exposes capabilities, drafts, revisions, previews and rendering. Revisions preserve prior jobs and can reuse matching media and speech. All assets and templates belong to a conversation.

The worker measures narration before composing the timeline. Short/long narration is rewritten by the local model (up to three attempts), with an explicit error for verbatim text that cannot fit. Music is mixed after checking the narration-only track. Voice controls are synthetic reference presets (`sohee`, `uncle_fu`), pitch-preserving speed, and sentence spacing. Requested BGM uses ACE-Step 1.5 Turbo (default), or XL Base through the Talk music lease. Generated tracks retain model/revision/seed and measured audio metadata; unchanged revisions reuse them. Tracks shorter than the video fail explicitly. Legacy synthesized keys/pad remain available only when explicitly requested. See [music runtime](../music_studio/README.md).

Before rendering, Chromium audits actual caption bounds, font size, wrapping, timing, loaded media and selected decorations. The local vision model reviews preview frames, and the designer receives failures for repair. `quality.json` records applied settings and measured verification; `sources.json` preserves provenance. Caption phrase timing is proportional within synthesized sentence segments, not forced word alignment. A preview is not marked as a completed video.

Manim scenes accept `manim.background` and up to 50 `manim.objects`. Objects support `type` (text/circle/rect/bar/line/arrow/polygon), position, dimensions, colors, fill, text/font size, start/duration, and fade/write/draw/grow animations. Line/arrow use `from`/`to`; polygon uses `vertices`. This declarative path complements freeform SVG/GSAP authoring.

Tests run in the worker image: `python3 -m unittest -v test_worker test_authoring`. The browser test exercises accepted layout and rejected hidden captions/decorations.

Photo-to-video productions use the same durable worker, through `clip_produce` and
`clip_manage` in Talk. A source photo and a natural-language brief are enough:
an intent-routing step distinguishes fresh execution, new versions, status,
cancel/resume and non-video questions from the latest user instruction and actual
job records. It validates the quoted instruction and source IDs before invoking
the production tool. Terminal status text comes from the saved job and delivery
receipt, so an old failure cannot be narrated as a new attempt without execution.
the local model plans an edit and motion, Qwen edits the actual source, a vision
review compares the full images and enlarged regions, and H3 animates the accepted
image. The output is one 124-frame, 24fps clip; 480p and 720p are pixel budgets that
preserve the input aspect ratio. This path does not edit an existing video or
perform face/person swaps.

`clip_pipeline.py` separates image defects from defects introduced during motion.
A single localized image defect uses a bounded inpaint region and verifies every
RGB pixel outside the mask. Repeated local failure switches back to the original
source. Video reviews inspect six timestamped frames and five adjacent 24fps
frames near the strongest sampled visual change, with focused crops for uncertain
findings. These are sampled visual checks, not exhaustive frame or audio review.
Before regenerating a failed video, the local model rewrites the motion instruction
from the accepted image and failed frames. It describes the desired motion and
stable scene rather than repeatedly appending names of unwanted artifacts.
Each image/video stage allows an initial generation plus two quality repairs.
An exhausted budget cannot be reset with retry, and failed results are not attached.

Production state lives in `clip_jobs` and `_clips/<id>/`: source hashes, original
user wording, the model plan, seeds, generation receipts, reviews and accepted
artifacts survive interruption. Completed steps are reused on retry. Brief network
outages use bounded reconnection retries with the same generation request. Cancel
wins a race with completion; a stopped task cannot be restarted by the same tool
loop. A new explicit resume instruction is required. The final MP4 is attached
once, and `report.json` records actual dimensions, duration, audio presence, input
lineage, review evidence and repair history. Intermediate candidates stay in the
production directory.

Talk exposes `/api/clip-production` for admission/listing and
`/api/clip-production/<id>` for status, `cancel`, `retry` and final files. The
internal worker API is `/v1/clips`; it shares one processing queue and memory
accounting with the shorts service. Tests:
`python3 -m unittest test_clip_pipeline test_worker test_music test_authoring`.

Voice catalog and chat selection
-------------------------------
`shorts_voices` lists nine installed voices with cached Fish WAV previews, previews
one, selects a conversation default, or applies a voice to an existing video.
Selection is stored in SQLite and snapshotted into each new project; explicit
`voice_preset` takes precedence. Applying to an existing video creates a new job,
keeps the script/media/music/design, and changes the voice without training.
Voice-only changes first fit playback pace within 0.8–1.2× using the saved raw
synthesis, without another Fish call. If the unchanged script still cannot fit
the requested duration, it fails explicitly
rather than silently rewriting it. The original job remains intact.
The canonical catalog, synthetic references and Fish previews live in
`../fish_drama3/voices`; `sync_voice_assets.py` copies them into this build context.
Speech receipts verify the requested voice ID and reference SHA256. Unchanged
Sohee/Uncle Fu references preserve their existing speech cache keys.
For voice-only revisions, the original artwork is immutable, including when a
saved template has since changed. An approved source visual review is reused
only when artwork and visual content match; the new output still receives a
fresh browser check of layout, media and caption timing. Opening-title opacity
exits are allowed only in the last 20% of their interval (at most 0.5 seconds),
with separate steady-state visibility checks. Headline color is measured from
the actual text spans rather than an inherited container color.


User reference voices and standalone audio
-----------------------------------------
Talk accepts uploaded audio or a direct media URL as a reference. Ask to register
it with a name for persistent use across local conversations, or request a temporary
voice for the current conversation. This conditions Fish on a reference; it does
not train model weights. Help and capability questions create no generation job.

The decoder probes the actual audio stream, including WAV, MP3, FLAC, Ogg,
M4A/AAC/ALAC, WebM, AIFF, WMA, AMR and CAF inputs. Invalid or unsupported streams
return an error. Uploads are limited to 512 MiB; automatic analysis supports two hours. Longer
inputs require a selected range. Ambiguous tracks or channels require selection. CPU VAD proposes up to three contiguous references of
at most 15 seconds. Less than one second of detected speech requires more input;
one to five seconds carries a quality warning. References are not padded, repeated
or stitched. Candidate audio, source times and recognized text are reviewable in
chat. A candidate, time range, channel or transcript can be corrected in a follow-up.
Speaker analysis reports uncertainty when unavailable.

`voice_tools` prepares references, generates standalone WAV, lists/inspects voices,
and renames, promotes or deletes them. Overall delivery belongs in `instructions`;
inline Fish tags remain in the exact spoken text. Output validation checks ASR
coverage and allows two synthesis attempts; it does not prove voice identity or
emotional fidelity. Failed validation is reported instead of claiming success.
Completed registration attaches the selected reference and generated preview.

Voice IDs are stable and reference revisions immutable. Each generation pins the
reference revision and hashes. Persistent registrations survive service restarts.
Temporary voices are visible only in their owning conversation and expire after
24 hours, with active jobs protected; they cannot become a conversation default
until promoted. Cleanup retains provenance and chat attachments. Deleting a voice
prevents new selection while preserving historical pinned video revisions.
Applying a custom voice to an existing short creates a new version.

`/api/voices` and `/api/voices/jobs` expose the conversation-scoped operations.
The worker persists jobs in SQLite, resumes interrupted work, and publishes each
result version once. Source analysis, Fish generation and output transcription
use sequential managed workloads. The main chat model remains resident.

Additional tests: `python3 -m unittest test_user_voices test_voices`.

Registration previews keep the greeting and following sentence in one bounded
Fish request, using the original text without an added performance tag. This
avoids synthesizing a very short greeting in an independent context. The optional
`continuous` generation flag supports up to 160 spoken characters; existing
sentence splitting remains the default for ordinary untagged generation.
The continuous setting is included in checkpoint identity, so a previously split
preview is not silently reused after this change. Existing library revisions and
chat attachments remain immutable.
