import assert from 'node:assert/strict';
import test from 'node:test';
import { PCMStreamPlayer, pcm16LEToFloat32, resolveSpeechSeed } from './pcm-player.js';

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

test('waits for the final scheduled audio node before closing', async () => {
  let sourceEnded = false;
  let contextClosed = false;
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
      return {
        connect() {},
        start() { setTimeout(() => { sourceEnded = true; this.onended?.(); }, 10); },
        stop() { this.onended?.(); },
      };
    }
    async resume() {}
    async close() { contextClosed = true; }
  }

  const player = new PCMStreamPlayer({ sampleRate: 24000, AudioContextClass: FakeAudioContext });
  await player.append(new Uint8Array(4800));
  const finishing = player.finish();
  assert.equal(contextClosed, false);
  await finishing;
  assert.equal(sourceEnded, true);
  assert.equal(contextClosed, true);
});
