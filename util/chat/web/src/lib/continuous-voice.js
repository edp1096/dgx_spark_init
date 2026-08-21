const calibrationSeconds = 0.9;
const speechStartSeconds = 0.12;
const speechEndSeconds = 0.95;
const minimumSpeechSeconds = 0.3;
const maximumSpeechSeconds = 60;
const preRollSeconds = 0.45;

export function voiceActivityThreshold(noiseFloor) {
  return Math.min(0.12, Math.max(0.018, noiseFloor * 2.4));
}

const ignorableVoiceFillers = new Set([
  '아', '어', '오', '우', '후', '음', '흠', '큼', '으음', '음음', '흠흠',
  'uh', 'um', 'umm', 'hm', 'hmm',
]);

export function isIgnorableVoiceTranscript(text) {
  const normalized = String(text || '')
    .trim()
    .toLocaleLowerCase();
  const parts = normalized.split(/[\s.,!?;:'"`~…·。！？、，]+/gu).filter(Boolean);
  return parts.length === 0 || parts.every((part) => ignorableVoiceFillers.has(part));
}

export function encodeVoiceWAV(chunks, sampleRate) {
  const sampleCount = chunks.reduce((total, chunk) => total + chunk.length, 0);
  const buffer = new ArrayBuffer(44 + (sampleCount * 2));
  const view = new DataView(buffer);
  const text = (offset, value) => { for (let index = 0; index < value.length; index += 1) view.setUint8(offset + index, value.charCodeAt(index)); };
  text(0, 'RIFF');
  view.setUint32(4, 36 + (sampleCount * 2), true);
  text(8, 'WAVE');
  text(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  text(36, 'data');
  view.setUint32(40, sampleCount * 2, true);
  let offset = 44;
  for (const chunk of chunks) {
    for (const value of chunk) {
      const sample = Math.max(-1, Math.min(1, value));
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
      offset += 2;
    }
  }
  return buffer;
}

export async function beginContinuousVoice(scope = globalThis, callbacks = {}) {
  if (!scope?.isSecureContext || !scope?.navigator?.mediaDevices?.getUserMedia) {
    throw new Error('이 브라우저에서는 연속 음성 입력을 사용할 수 없습니다.');
  }
  const AudioContextClass = scope.AudioContext || scope.webkitAudioContext;
  if (!AudioContextClass) throw new Error('이 브라우저에서는 음성 감지를 사용할 수 없습니다.');
  const stream = await scope.navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });
  let audioContext;
  try {
    audioContext = new AudioContextClass();
    await audioContext.resume();
  } catch (error) {
    stream.getTracks().forEach((track) => track.stop());
    if (audioContext) await audioContext.close().catch(() => {});
    throw error;
  }

  const sampleRate = audioContext.sampleRate;
  const calibrationSamples = Math.round(sampleRate * calibrationSeconds);
  const speechStartSamples = Math.round(sampleRate * speechStartSeconds);
  const speechEndSamples = Math.round(sampleRate * speechEndSeconds);
  const minimumSpeechSamples = Math.round(sampleRate * minimumSpeechSeconds);
  const maximumSpeechSamples = Math.round(sampleRate * maximumSpeechSeconds);
  const maximumPreRollSamples = Math.round(sampleRate * preRollSeconds);
  const source = audioContext.createMediaStreamSource(stream);
  const silentGain = audioContext.createGain();
  silentGain.gain.value = 0;
  silentGain.connect(audioContext.destination);
  let captureNode;
  let workletURL = '';
  let stopped = false;
  let paused = false;
  let processedSamples = 0;
  let noiseFloor = 0.006;
  let candidateSamples = 0;
  let speechSamples = 0;
  let silenceSamples = 0;
  let speaking = false;
  let currentState = '';
  let segmentContext = null;
  let segmentChunks = [];
  let preRollChunks = [];
  let preRollSampleCount = 0;

  function state(value) {
    if (value === currentState) return;
    currentState = value;
    callbacks.onState?.(value);
  }

  function keepPreRoll(chunk) {
    preRollChunks.push(chunk);
    preRollSampleCount += chunk.length;
    while (preRollSampleCount > maximumPreRollSamples && preRollChunks.length > 1) {
      preRollSampleCount -= preRollChunks[0].length;
      preRollChunks.shift();
    }
  }

  function finishSegment() {
    if (!speaking) return;
    const chunks = segmentChunks;
    const context = segmentContext;
    const accepted = speechSamples >= minimumSpeechSamples;
    speaking = false;
    candidateSamples = 0;
    speechSamples = 0;
    silenceSamples = 0;
    segmentChunks = [];
    segmentContext = null;
    preRollChunks = [];
    preRollSampleCount = 0;
    if (!stopped) state('listening');
    if (accepted && chunks.length) {
      callbacks.onUtterance?.(new scope.Blob([encodeVoiceWAV(chunks, sampleRate)], { type: 'audio/wav' }), context);
    }
  }

  function processBlock(rawBlock) {
    if (stopped || paused || !rawBlock?.length) return;
    const chunk = new Float32Array(rawBlock);
    let power = 0;
    for (const sample of chunk) power += sample * sample;
    const rms = Math.sqrt(power / chunk.length);
    processedSamples += chunk.length;
    const threshold = voiceActivityThreshold(noiseFloor);

    if (processedSamples <= calibrationSamples) {
      noiseFloor = Math.max(noiseFloor, rms * 0.85);
      keepPreRoll(chunk);
      state('calibrating');
      return;
    }

    if (!speaking) {
      keepPreRoll(chunk);
      if (rms < threshold) noiseFloor = (noiseFloor * 0.985) + (rms * 0.015);
      if (rms >= threshold) {
        candidateSamples += chunk.length;
        if (candidateSamples >= speechStartSamples) {
          speaking = true;
          speechSamples = candidateSamples;
          silenceSamples = 0;
          segmentContext = callbacks.getContext?.() || null;
          segmentChunks = preRollChunks;
          preRollChunks = [];
          preRollSampleCount = 0;
          state('speaking');
        }
      } else {
        candidateSamples = 0;
        state('listening');
      }
      return;
    }

    segmentChunks.push(chunk);
    speechSamples += chunk.length;
    if (rms >= Math.max(0.012, threshold * 0.72)) silenceSamples = 0;
    else silenceSamples += chunk.length;
    if ((silenceSamples >= speechEndSamples && speechSamples >= minimumSpeechSamples)
      || speechSamples >= maximumSpeechSamples) finishSegment();
  }

  async function installCapture() {
    const AudioWorkletNodeClass = scope.AudioWorkletNode;
    if (audioContext.audioWorklet && AudioWorkletNodeClass && scope.URL?.createObjectURL) {
      const processorSource = `
        class SparkTalkVoiceCapture extends AudioWorkletProcessor {
          constructor() { super(); this.buffer = new Float32Array(2048); this.offset = 0; }
          process(inputs) {
            const input = inputs[0] && inputs[0][0];
            if (!input) return true;
            let sourceOffset = 0;
            while (sourceOffset < input.length) {
              const count = Math.min(input.length - sourceOffset, this.buffer.length - this.offset);
              this.buffer.set(input.subarray(sourceOffset, sourceOffset + count), this.offset);
              this.offset += count; sourceOffset += count;
              if (this.offset === this.buffer.length) {
                this.port.postMessage(this.buffer.buffer, [this.buffer.buffer]);
                this.buffer = new Float32Array(2048); this.offset = 0;
              }
            }
            return true;
          }
        }
        registerProcessor('sparktalk-voice-capture', SparkTalkVoiceCapture);
      `;
      workletURL = scope.URL.createObjectURL(new scope.Blob([processorSource], { type: 'text/javascript' }));
      try {
        await audioContext.audioWorklet.addModule(workletURL);
        captureNode = new AudioWorkletNodeClass(audioContext, 'sparktalk-voice-capture');
        captureNode.port.onmessage = (event) => processBlock(new Float32Array(event.data));
      } finally {
        scope.URL.revokeObjectURL(workletURL);
        workletURL = '';
      }
    } else if (audioContext.createScriptProcessor) {
      captureNode = audioContext.createScriptProcessor(2048, 1, 1);
      captureNode.onaudioprocess = (event) => processBlock(event.inputBuffer.getChannelData(0));
    } else {
      throw new Error('이 브라우저에서는 연속 음성 캡처를 사용할 수 없습니다.');
    }
    source.connect(captureNode);
    captureNode.connect(silentGain);
  }

  async function resumeWhenVisible() {
    if (scope.document?.visibilityState === 'visible' && audioContext.state === 'suspended') {
      await audioContext.resume().catch(() => {});
    }
  }

  async function stop() {
    if (stopped) return;
    if (speaking) finishSegment();
    stopped = true;
    scope.document?.removeEventListener('visibilitychange', resumeWhenVisible);
    if (captureNode) {
      if ('onaudioprocess' in captureNode) captureNode.onaudioprocess = null;
      if (captureNode.port) captureNode.port.onmessage = null;
      captureNode.disconnect();
    }
    source.disconnect();
    silentGain.disconnect();
    stream.getTracks().forEach((track) => track.stop());
    if (workletURL) scope.URL.revokeObjectURL(workletURL);
    await audioContext.close().catch(() => {});
    state('off');
  }

  function setPaused(value) {
    if (stopped) return;
    paused = Boolean(value);
    candidateSamples = 0;
    speechSamples = 0;
    silenceSamples = 0;
    speaking = false;
    segmentChunks = [];
    segmentContext = null;
    preRollChunks = [];
    preRollSampleCount = 0;
    state(paused ? 'paused' : (processedSamples <= calibrationSamples ? 'calibrating' : 'listening'));
  }

  try {
    await installCapture();
  } catch (error) {
    await stop();
    throw error;
  }
  scope.document?.addEventListener('visibilitychange', resumeWhenVisible);
  state('calibrating');
  return { stop, setPaused };
}
