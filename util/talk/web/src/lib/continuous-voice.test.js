import assert from 'node:assert/strict';
import test from 'node:test';
import {
  beginContinuousVoice, encodeVoiceWAV, isIgnorableVoiceTranscript, voiceActivityThreshold,
} from './continuous-voice.js';

test('continuous voice threshold adapts while retaining safe bounds', () => {
  assert.equal(voiceActivityThreshold(0.001), 0.018);
  assert.equal(voiceActivityThreshold(0.02), 0.048);
  assert.equal(voiceActivityThreshold(0.2), 0.12);
});

test('continuous voice ignores punctuation and standalone hesitation sounds', () => {
  for (const text of ['.', '…', '아.', '음', '흠!', '큼', 'hmm...', '아. 오. 후.']) {
    assert.equal(isIgnorableVoiceTranscript(text), true, text);
  }
  for (const text of ['네', '응', '아 맞다', '아. 네.', '음성 입력', 'hello']) {
    assert.equal(isIgnorableVoiceTranscript(text), false, text);
  }
});

test('WAV encoder creates a mono PCM file with the requested sample rate', () => {
  const wav = encodeVoiceWAV([new Float32Array([0, -1, 1])], 16000);
  const bytes = Buffer.from(wav);
  const view = new DataView(wav);

  assert.equal(bytes.subarray(0, 4).toString(), 'RIFF');
  assert.equal(bytes.subarray(8, 12).toString(), 'WAVE');
  assert.equal(view.getUint16(22, true), 1);
  assert.equal(view.getUint32(24, true), 16000);
  assert.equal(view.getUint16(34, true), 16);
  assert.equal(view.getUint32(40, true), 6);
});

test('continuous voice emits independent WAV utterances with audio pre-roll', async () => {
  let processor;
  let trackStopped = false;
  const utterances = [];
  const states = [];
  const stream = { getTracks: () => [{ stop() { trackStopped = true; } }] };
  class FakeAudioContext {
    constructor() {
      this.state = 'running';
      this.sampleRate = 16000;
      this.destination = {};
    }
    async resume() {}
    async close() { this.state = 'closed'; }
    createMediaStreamSource() { return { connect() {}, disconnect() {} }; }
    createGain() { return { gain: { value: 1 }, connect() {}, disconnect() {} }; }
    createScriptProcessor() {
      processor = { onaudioprocess: null, connect() {}, disconnect() {} };
      return processor;
    }
  }
  const scope = {
    isSecureContext: true,
    navigator: { mediaDevices: { async getUserMedia() { return stream; } } },
    AudioContext: FakeAudioContext,
    Blob,
    document: { visibilityState: 'visible', addEventListener() {}, removeEventListener() {} },
  };
  const listener = await beginContinuousVoice(scope, {
    onState: (state) => states.push(state),
    onUtterance: (blob) => utterances.push(blob),
  });
  const feed = (level, blocks = 1) => {
    for (let index = 0; index < blocks; index += 1) {
      const samples = new Float32Array(2048);
      samples.fill(level);
      processor.onaudioprocess({ inputBuffer: { getChannelData: () => samples } });
    }
  };

  feed(0.002, 8); // 0.9초 소음 측정을 넘긴다.
  feed(0.04, 1);
  feed(0.002, 8);
  feed(0.04, 1);
  feed(0.002, 8);

  assert.equal(utterances.length, 2);
  for (const utterance of utterances) {
    assert.equal(utterance.type, 'audio/wav');
    const bytes = Buffer.from(await utterance.arrayBuffer());
    assert.equal(bytes.subarray(0, 4).toString(), 'RIFF');
    assert.equal(bytes.subarray(8, 12).toString(), 'WAVE');
    // 감지된 발화뿐 아니라 직전 PCM 프리롤까지 포함되어야 한다.
    assert.ok(bytes.length > 44 + (2048 * 2));
  }
  assert.ok(states.includes('speaking'));

  listener.setPaused(true);
  feed(0.04, 2);
  feed(0.002, 8);
  assert.equal(states.at(-1), 'paused');
  assert.equal(utterances.length, 2);
  listener.setPaused(false);
  assert.equal(states.at(-1), 'listening');

  await listener.stop();
  assert.equal(trackStopped, true);
  assert.equal(states.at(-1), 'off');
});
