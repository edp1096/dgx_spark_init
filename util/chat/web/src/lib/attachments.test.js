import test from 'node:test';
import assert from 'node:assert/strict';
import { attachmentKind, canPreviewVideo, hasFileDrag, isSupportedAttachmentFile } from './attachments.js';

test('accepts supported image, audio, and video files with MIME or extension fallbacks', () => {
  assert.equal(isSupportedAttachmentFile({ name: 'photo.png', type: 'image/png' }), true);
  assert.equal(isSupportedAttachmentFile({ name: 'voice.wav', type: '' }), true);
  assert.equal(isSupportedAttachmentFile({ name: 'clip.wmv', type: 'application/octet-stream' }), true);
  assert.equal(isSupportedAttachmentFile({ name: 'notes.txt', type: 'text/plain' }), false);
});

test('recognizes file drags without depending on a file input change event', () => {
  assert.equal(hasFileDrag({ files: [], items: [{ kind: 'file' }], types: ['Files'] }), true);
  assert.equal(hasFileDrag({ files: [], items: [{ kind: 'string' }], types: ['text/plain'] }), false);
});

test('classifies attachments and limits inline video previews to browser-friendly formats', () => {
  assert.equal(attachmentKind({ name: 'voice.ogg', mime: 'audio/ogg' }), 'audio');
  assert.equal(attachmentKind({ name: 'movie.ogg', mime: 'video/ogg' }), 'video');
  assert.equal(attachmentKind({ name: 'photo.webp', mime: 'image/webp' }), 'image');
  assert.equal(canPreviewVideo({ mime: 'video/mp4' }), true);
  assert.equal(canPreviewVideo({ mime: 'video/x-ms-wmv' }), false);
});
