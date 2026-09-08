import assert from 'node:assert/strict';
import test from 'node:test';
import { preferredVoiceMIME, voiceFilename, voiceRecordingSupported } from './voice-recorder.js';

test('voice recording requires a secure context and media APIs', () => {
  const mediaRecorder = function MediaRecorder() {};
  assert.equal(voiceRecordingSupported({ isSecureContext: true, navigator: { mediaDevices: { getUserMedia() {} } }, MediaRecorder: mediaRecorder }), true);
  assert.equal(voiceRecordingSupported({ isSecureContext: false, navigator: { mediaDevices: { getUserMedia() {} } }, MediaRecorder: mediaRecorder }), false);
});

test('voice recording prefers Opus and uses matching filenames', () => {
  const recorder = { isTypeSupported: (mime) => mime === 'audio/webm;codecs=opus' };
  assert.equal(preferredVoiceMIME(recorder), 'audio/webm;codecs=opus');
  assert.equal(voiceFilename('audio/webm;codecs=opus'), 'voice.webm');
  assert.equal(voiceFilename('audio/ogg;codecs=opus'), 'voice.ogg');
  assert.equal(voiceFilename('audio/mp4'), 'voice.m4a');
  assert.equal(voiceFilename('audio/wav'), 'voice.wav');
});
