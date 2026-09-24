#!/bin/sh
set -eu
# Existing installations without diarization weights retain ordinary ASR.
if [ "${1:-}" = serve ] && [ -s /models/Nemotron-3-Diarization.q8_0.gguf ]; then
  exec /opt/nemo-speech/nemo-speech "$@" --diar-model /models/Nemotron-3-Diarization.q8_0.gguf --diar-preset v3-offline
fi
exec /opt/nemo-speech/nemo-speech "$@"
