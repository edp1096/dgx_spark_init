import assert from 'node:assert/strict';
import test from 'node:test';
import { createAttachmentController } from './attachment-controller.js';

test('attachment controller preserves independent drafts across sessions', async () => {
  const states = [];
  const controller = createAttachmentController({
    uploadFile: async (file) => ({ id: file.name, name: file.name, size: file.size, mime: file.type }),
    uploadURL: async () => ({ id: 'url', name: 'video.mp4', size: 12, mime: 'video/mp4' }),
    onState: (state) => states.push(state),
  });
  controller.select('one');
  controller.addFiles([{ name: 'one.png', type: 'image/png', size: 10 }]);
  await new Promise((resolve) => setTimeout(resolve, 0));
  controller.select('two');
  await controller.addURL('https://example.com/video');
  assert.deepEqual(controller.snapshot().pending.map((item) => item.id), ['url']);
  controller.select('one');
  assert.deepEqual(controller.snapshot().pending.map((item) => item.id), ['one.png']);
  assert.ok(states.length > 2);
});

test('attachment controller rejects unsupported and oversized files', () => {
  const errors = [];
  const controller = createAttachmentController({
    uploadFile: async () => ({}), uploadURL: async () => ({}),
    onError: (_sessionId, message) => errors.push(message),
  });
  controller.select('one');
  assert.equal(controller.addFiles([{ name: 'note.txt', type: 'text/plain', size: 10 }]), false);
  assert.equal(controller.addFiles([{ name: 'large.png', type: 'image/png', size: 16 * 1024 * 1024 }]), false);
  assert.match(errors[0], /지원되는/u);
  assert.match(errors[1], /15MB/u);
});
