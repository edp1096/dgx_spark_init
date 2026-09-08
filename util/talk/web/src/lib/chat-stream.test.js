import test from 'node:test';
import assert from 'node:assert/strict';
import { createStreamHandlers } from './chat-stream.js';

test('stream handlers update one assistant message and publish every event', () => {
  const message = { content: '', reasoning_content: '', tool_trace: [], activity: '' };
  let publishes = 0;
  const handlers = createStreamHandlers(message, () => { publishes += 1; });

  handlers.reasoning('생각');
  handlers.toolStart({ id: 'one', name: 'web_search' });
  handlers.toolResult({ id: 'one', result: '검색 결과' });
  handlers.delta('답변');

  assert.equal(message.reasoning_content, '생각');
  assert.equal(message.content, '답변');
  assert.equal(message.tool_trace[0].result, '검색 결과');
  assert.equal(message.activity, 'answer');
  assert.equal(publishes, 4);
});

test('SSH tool approval and raw output update the active tool', () => {
  const message = { content: '', reasoning_content: '', tool_trace: [], activity: '' };
  const handlers = createStreamHandlers(message, () => {});
  handlers.toolStart({ id: 'ssh-one', name: 'ssh_exec', arguments: '{"host":"dgx-main"}' });
  handlers.toolApproval({ id: 'ssh-one', approval_id: 'approval-1', command: 'nvidia-smi', host_name: 'DGX Spark' });
  assert.equal(message.tool_trace[0].approval_required, true);
  handlers.toolApprovalResolved({ id: 'ssh-one', approved: true });
  handlers.toolExecution({ id: 'ssh-one', status: 'running' });
  handlers.toolOutput({ id: 'ssh-one', stream: 'stdout', delta: 'GPU OK\n' });
  handlers.toolResult({ id: 'ssh-one', result: '{"exit_code":0}' });
  assert.equal(message.tool_trace[0].approval_required, false);
  assert.equal(message.tool_trace[0].approved, true);
  assert.equal(message.tool_trace[0].output, 'GPU OK\n');
  assert.equal(message.tool_trace[0].running, false);
});

import {consumeSSE} from '../api/chat.js';

function response(events){
 const encoded=new TextEncoder().encode(events);
 return new Response(new ReadableStream({start(c){for(let i=0;i<encoded.length;i+=7)c.enqueue(encoded.slice(i,i+7));c.close();}}));
}
test('chat stream requires explicit completion and preserves interrupted code',async()=>{
 let text='',done=false;
 const partial='event: delta\ndata: {"delta":"func main() { 한글"}\n\n';
 await assert.rejects(consumeSSE(response(partial),{delta:t=>text+=t,done:()=>done=true}),/완료 신호 없이/);
 assert.equal(text,'func main() { 한글');assert.equal(done,false);
 await consumeSSE(response(partial+'event: done\ndata: {}\n\n'),{done:()=>done=true});
 assert.equal(done,true);
});
test('backend limit remains an error even if a done event follows',async()=>{
 let done=false;
 await assert.rejects(consumeSSE(response('event: error\ndata: {"error":"출력 한도"}\n\nevent: done\ndata: {}\n\n'),{done:()=>done=true}),/출력 한도/);
 assert.equal(done,false);
});
