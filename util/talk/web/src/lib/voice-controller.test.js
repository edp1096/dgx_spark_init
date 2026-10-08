import assert from 'node:assert/strict';
import test from 'node:test';
import { createVoiceController, formatVoiceError } from './voice-controller.js';

test('voice controller records, transcribes, and delivers text to its session', async () => {
  const states = [];
  const transcripts = [];
  const controller = createVoiceController({
    environment: () => ({ isSecureContext: true, navigator: { mediaDevices: { getUserMedia() {} } }, MediaRecorder: class {} }),
    transcribe: async () => ({ text: '안녕하세요' }),
    getActiveSessionId: () => 'chat-1',
    onTranscript: (sessionId, text) => transcripts.push({ sessionId, text }),
    onState: (state) => states.push(state),
    beginRecording: async () => ({ stop: async () => new Blob(['voice'], { type: 'audio/webm' }) }),
  });
  await controller.startManual();
  await controller.stopManual();
  assert.deepEqual(transcripts, [{ sessionId: 'chat-1', text: '안녕하세요' }]);
  assert.equal(states.at(-1).manualState, 'idle');
});

test('voice errors retain actionable browser permission messages', () => {
  assert.match(formatVoiceError({ name: 'NotAllowedError' }), /권한/u);
  assert.match(formatVoiceError({ name: 'NotFoundError' }), /마이크/u);
});

test('continuous voice becomes enabled as soon as startup is requested', async () => {
  const states = [];
  let listenerState;
  const controller = createVoiceController({
    environment: () => ({ isSecureContext: true, navigator: { mediaDevices: { getUserMedia() {} } }, MediaRecorder: class {} }),
    transcribe: async () => ({ text: '' }),
    getActiveSessionId: () => 'chat-1',
    onState: (state) => states.push({ ...state }),
    beginContinuous: async (_scope, callbacks) => {
      listenerState = callbacks.onState;
      callbacks.onState('calibrating');
      return { stop: async () => {}, setPaused() {} };
    },
  });

  await controller.toggleContinuous();
  assert.equal(states.find((state) => state.continuousState === 'requesting')?.continuousEnabled, true);
  assert.equal(states.at(-1).continuousEnabled, true);
  listenerState('listening');
  assert.equal(states.at(-1).continuousState, 'listening');
  assert.equal(states.at(-1).continuousEnabled, true);

  await controller.toggleContinuous();
  assert.equal(states.at(-1).continuousState, 'off');
  assert.equal(states.at(-1).continuousEnabled, false);
});
