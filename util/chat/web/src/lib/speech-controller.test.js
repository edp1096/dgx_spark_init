import assert from 'node:assert/strict';
import test from 'node:test';
import { createSpeechController } from './speech-controller.js';

test('speech controller drains queued speech before release', async () => {
  const requests = [];
  const statuses = [];
  const playback = [];
  class Player {
    constructor({ onStart }) { this.onStart = onStart; this.started = false; }
    async append(bytes) { if (!this.started) { this.started = true; this.onStart(); } this.bytes = bytes; }
    async finish() { this.finished = true; }
    stop() { this.stopped = true; }
  }
  const controller = createSpeechController({
    stream: async (text, _signal, onChunk) => {
      requests.push(text);
      await onChunk(new Uint8Array([1, 2]), 24000);
    },
    PlayerClass: Player,
    onStatus: (status) => statuses.push(status),
    onPlaybackChange: (value) => playback.push(value),
  });
  const session = controller.create('answer:1', 'chat-1');
  controller.enqueue(session, '첫 문장');
  controller.enqueue(session, '둘째 문장');
  await controller.close(session);
  assert.deepEqual(requests, ['첫 문장', '둘째 문장']);
  assert.deepEqual(playback, [true, false]);
  assert.deepEqual(statuses.at(-1), { loadingKey: '', playingKey: '' });
  assert.equal(controller.isCurrent(session), false);
});

test('speech controller aborts and stops the active player', async () => {
  let player;
  class Player {
    constructor({ onStart }) { player = this; this.onStart = onStart; }
    async append() { this.onStart(); }
    async finish() {}
    stop() { this.stopped = true; }
  }
  const controller = createSpeechController({
    stream: async (_text, _signal, onChunk) => onChunk(new Uint8Array([1]), 24000),
    PlayerClass: Player,
  });
  const session = controller.create('answer:2', 'chat-2');
  controller.enqueue(session, '문장');
  await Promise.resolve();
  controller.stop();
  assert.equal(session.controller.signal.aborted, true);
  assert.equal(player.stopped, true);
});
