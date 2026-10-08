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

function deferred() { let resolve; const promise = new Promise(r => resolve = r); return {promise,resolve}; }
test('out-of-order session loads never pollute another session cache', async () => {
  const a=deferred(), b=deferred();const visible=[];
  const c=createChatSessionController({loadMessages:id => id==='a'?a.promise:b.promise,onMessages:(m,id)=>visible.push({m,id})});
  const first=c.activate('a');const second=c.activate('b');
  b.resolve([{id:'b'}]);await second;a.resolve([{id:'a'}]);await first;
  assert.equal(visible.at(-1).id,'b');assert.deepEqual(c.getMessages('b'),[{id:'b'}]);
  await c.activate('a');assert.deepEqual(c.getMessages('a'),[{id:'a'}]);
});
test('older same-session request cannot overwrite the latest activation', async () => {
  const old=deferred(), recent=deferred();let count=0;const visible=[];
  const c=createChatSessionController({loadMessages:()=>++count===1?old.promise:recent.promise,onMessages:m=>visible.push(m)});
  const first=c.activate('a');const second=c.activate('a');recent.resolve([{id:'new'}]);await second;old.resolve([{id:'old'}]);await first;
  assert.deepEqual(c.getMessages('a'),[{id:'new'}]);assert.deepEqual(visible.at(-1),[{id:'new'}]);
});
test('additional input retries retain the id and late replies do not mutate a replacement turn', async () => {
  const c=createChatSessionController({loadMessages:async()=>[]});const run=c.start('a',[],0,{turnId:'turn'});const ids=[];
  await assert.rejects(c.sendInput('a','hello',async(s,t,id)=>{ids.push(id);throw Error('network')}));
  await c.sendInput('a','hello',async(s,t,id)=>{ids.push(id);return {}});assert.equal(ids[0],ids[1]);assert.equal(run.pendingInput,null);
  const pending=deferred();let saved=false;const response=c.sendInput('a','late',()=>pending.promise,()=>saved=true);
  c.finish('a',run);c.start('a',[],0,{turnId:'new'});pending.resolve({});assert.equal(await response,false);assert.equal(saved,false);
});
test('dispose aborts active turns and rejects stale load publication', async () => {
  const pending=deferred();let published=0;
  const c=createChatSessionController({loadMessages:()=>pending.promise,onMessages:()=>published++});const loading=c.activate('a');const run=c.start('b',[],0);c.dispose();pending.resolve([]);await loading;
  assert.equal(run.controller.signal.aborted,true);assert.equal(published,0);
});

test('an optimistic turn published during loading wins over the old server response', async () => {
  const pending=deferred();const c=createChatSessionController({loadMessages:()=>pending.promise});const loading=c.activate('a');
  const run=c.start('a',[{id:'optimistic'}],0);pending.resolve([{id:'stale'}]);await loading;
  assert.deepEqual(c.getMessages('a'),run.messages);
});
