import assert from 'node:assert/strict';
import test from 'node:test';
import { createChatSessionController } from './chat-session-controller.js';

test('chat session controller caches messages and restores background runs', async () => {
  const visible = [];
  const controller = createChatSessionController({
    loadMessages: async (id) => [{ id: `${id}-loaded` }],
    onMessages: (messages, id) => visible.push({ id, messages }),
  });
  await controller.activate('one');
  controller.publish('one', [{ id: 'one-local' }]);
  const run = controller.start('two', [{ id: 'two-running' }], 0);
  await controller.activate('two');
  assert.deepEqual(visible.at(-1), { id: 'two', messages: run.messages });
  controller.finish('two', run);
  await controller.activate('one');
  assert.deepEqual(visible.at(-1).messages, [{ id: 'one-local' }]);
});

test('chat session controller aborts and clears per-session state', async () => {
  let active = '';
  const controller = createChatSessionController({
    loadMessages: async () => [], onActive: (id) => { active = id; },
  });
  await controller.activate('one');
  const run = controller.start('one', [], 0);
  controller.abort('one');
  assert.equal(run.controller.signal.aborted, true);
  controller.remove('one');
  assert.equal(active, '');
  assert.deepEqual(controller.getMessages('one'), []);
});
