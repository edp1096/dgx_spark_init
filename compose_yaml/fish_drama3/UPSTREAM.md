Drama3: `watchpen/drama-3-int4-koen160@fd0147b1bb988ea26718ee75e95de8b07a07c156`.
SGLang Omni: `1240a6993c8073593aa0ad074859ccee2797adff` (Apache-2.0).
Sampling fixes: `ca9e8789fcd84ed6f7272276cb99447eec751787`: retain the top-p crossing token and cap acoustic codebooks at top-k 30 independently of semantic sampling. Included source files retain upstream notices.
Hadamard8: supplied with the pinned KoEn160 model; optional unpublished route-statistics hook removed.
Reference voice: synthetic Qwen3-TTS Sohee, generated locally; no recorded human voice.
Default base: published ARM64 CUDA/PyTorch image pinned by digest in the Dockerfile. The first local build reused the same tested stack through `FISH_BASE_IMAGE=local/drama3-omni:1240a69-trial4`.

Source: https://github.com/sgl-project/sglang-omni
Source: https://huggingface.co/watchpen/drama-3-int4-koen160
Source: https://huggingface.co/fishaudio/drama-3

Voice catalog: `voices/catalog.json` contains nine synthetic Qwen3-TTS 0.6B
CustomVoice references, their exact transcripts and SHA256 hashes. Sohee and
Uncle Fu retain their original reference bytes. The other seven references are
selected from locally generated candidates after duration, clipping and Korean
ASR checks. Fish receives the selected WAV and its matching transcript on every
synthesis request; this is reference conditioning, with no weight updates.
`voices/previews` contains actual Fish synthesis of a shared preview script.
Run `python3 sync_voice_assets.py` after updating the catalog/audio to synchronize
the Shorts build context, then package Talk support assets and rebuild both images.

Native directions: tagged narration is sent as one request, with a maximum of
160 spoken characters per scene. Balanced scopes and brackets are checked before
the batch starts. Untagged narration keeps the existing sentence chunking.
Caption segments and ASR expectations contain spoken words only. Tagged output
keeps native silence and gain; the video worker bypasses loudness normalization
and copies the WAV exactly at the default speed. Explicit global speed changes
still use atempo. Tagged narration is not rewritten to fit a video duration.

The waveform guard rejects cap hits, empty output, and unexpected silent tails
over two seconds before trimming. An explicit terminal [pause] permits its tail.
Internal pauses in tagged speech are preserved. These guards do not classify
emotion or guarantee tag adherence; the video pipeline still checks transcript
coverage. Tagged speech uses cache policy 4 so older split/normalized results
cannot be reused as native delivery.
