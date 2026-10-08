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
	assert.equal(controller.addFiles([{ name: 'legacy.hwp', type: 'application/octet-stream', size: 10 }]), false);
  assert.equal(controller.addFiles([{ name: 'large.png', type: 'image/png', size: 257 * 1024 * 1024 }]), false);
  assert.match(errors[0], /지원되는/u);
  assert.match(errors[1], /256MiB/u);
});

test('inherits shared size and applies only explicit type overrides', async () => {
  let limits = { max_file_mb: 256, max_files: 6 };
  const errors = [], uploads = [];
  const controller = createAttachmentController({
    getLimits: () => limits,
    uploadFile: async (file) => { uploads.push(file.name); return { id: file.name, name: file.name, size: file.size, mime: file.type }; },
    uploadURL: async () => ({}), onError: (_id, message) => { if (message) errors.push(message); },
  });
  controller.select('one');
  assert.equal(controller.addFiles([{name:'boundary.zip',type:'application/zip',size:256*1024*1024}]),true);
  await new Promise(resolve => setTimeout(resolve,0));
  assert.equal(controller.addFiles([{name:'too-big.zip',size:256*1024*1024+1}]),false);
  limits = {...limits, type_limits_mb:{image:15}};
  assert.equal(controller.addFiles([{name:'photo.png',type:'image/png',size:16*1024*1024}]),false);
  assert.equal(controller.addFiles([{name:'other.zip',size:128*1024*1024}]),true);
  limits = {...limits, max_file_mb:128, type_limits_mb:{}};
  assert.equal(controller.addFiles([{name:'photo.png',type:'image/png',size:16*1024*1024}]),true);
  assert.match(errors[0],/256MiB/); assert.match(errors[1],/15MiB/);
});
