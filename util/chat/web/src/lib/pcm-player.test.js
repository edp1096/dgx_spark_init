import assert from 'node:assert/strict';
import test from 'node:test';
import { PCMStreamPlayer, outputDrainDelayMs, pcm16LEToFloat32, resolveSpeechSeed } from './pcm-player.js';

test('converts signed little-endian PCM16 samples', () => {
  const bytes = new Uint8Array([0x00, 0x80, 0x00, 0x00, 0xff, 0x7f]);
  const samples = pcm16LEToFloat32(bytes);
  assert.equal(samples.length, 3);
  assert.equal(samples[0], -1);
  assert.equal(samples[1], 0);
  assert.equal(samples[2], 1);
});

test('uses a fixed seed or creates one random seed for a playback session', () => {
  const fakeCrypto = { getRandomValues(values) { values[0] = 0xf1234567; return values; } };
  assert.equal(resolveSpeechSeed(240, fakeCrypto), 240);
  assert.equal(resolveSpeechSeed(-1, fakeCrypto), 0x71234567);
});

test('uses reported output latency with bounded universal drain time', () => {
  assert.equal(outputDrainDelayMs({}), 220);
  assert.equal(outputDrainDelayMs({ baseLatency: 0.02, outputLatency: 0.08 }), 220);
  assert.equal(outputDrainDelayMs({ baseLatency: 0.1, outputLatency: 0.2 }), 420);
  assert.equal(outputDrainDelayMs({ baseLatency: 0.4, outputLatency: 0.4 }), 500);
});

test('waits for the final node and hardware drain time before closing', async () => {
  let sourceEnded = false;
  let contextClosed = false;
  let source;
  let drainDelay = 0;
  let releaseDrain;
  class FakeAudioContext {
    constructor() {
      this.state = 'running';
      this.currentTime = 0;
      this.destination = {};
    }
    createBuffer(_channels, length, sampleRate) {
      return { duration: length / sampleRate, copyToChannel() {} };
    }
    createBufferSource() {
      source = {
        connect() {},
        start() {},
        stop() { this.onended?.(); },
      };
      return source;
    }
    async resume() {}
    async close() { contextClosed = true; }
  }

  const player = new PCMStreamPlayer({
    sampleRate: 24000,
    AudioContextClass: FakeAudioContext,
    sleep(milliseconds) {
      drainDelay = milliseconds;
      return new Promise((resolve) => { releaseDrain = resolve; });
    },
  });
  await player.append(new Uint8Array(4800));
  const finishing = player.finish();
  assert.equal(contextClosed, false);
  sourceEnded = true;
  source.onended();
  await Promise.resolve();
  assert.equal(drainDelay, 220);
  assert.equal(contextClosed, false);
  releaseDrain();
  await finishing;
  assert.equal(sourceEnded, true);
  assert.equal(contextClosed, true);
});
