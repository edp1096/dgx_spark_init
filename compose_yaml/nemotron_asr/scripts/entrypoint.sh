#!/bin/sh
set -eu
if [ "${1:-}" = serve ]; then
  previous=""
  for argument in "$@"; do
    if [ "$previous" = --asr-model ] && [ "$argument" = /models/nemotron-3.5-asr-streaming-0.6b.q5_k.gguf ]; then
      printf '%s  %s\n' f0dab30ca22a606c2ab9a65efae88421851c6ca4ab01160708ab733b4467ce58 "$argument" | sha256sum --check --status || {
        echo 'Qualified Q5_K model is missing or corrupt; run model preparation.' >&2
        exit 1
      }
    fi
    previous="$argument"
  done
fi
# Existing installations without diarization weights retain ordinary ASR.
if [ "${1:-}" = serve ] && [ -s /models/Nemotron-3-Diarization.q8_0.gguf ]; then
  exec /opt/nemo-speech/nemo-speech "$@" --diar-model /models/Nemotron-3-Diarization.q8_0.gguf --diar-preset v3-offline
fi
exec /opt/nemo-speech/nemo-speech "$@"
