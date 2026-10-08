import test from 'node:test';
import assert from 'node:assert/strict';
import {generationProgressLabel} from './generation-progress.js';
import {createStreamHandlers} from './chat-stream.js';
test('generation events update live ETA and retain logs after tool result',()=>{
 const message={tool_trace:[]};const h=createStreamHandlers(message,()=>{});h.toolStart({id:'video-one',name:'video_generate'});
 h.toolOutput({id:'video-one',delta:'[0:20] 샘플링 2/20\n',progress:{kind:'h3',stage:'sample',step:2,total:20,elapsed_seconds:20,eta_seconds:202,eta_scope:'total'}});
 assert.match(generationProgressLabel(message.tool_trace[0]),/샘플링 2\/20 · 전체 예상 3:22 남음 · 경과 0:20/);
 h.toolOutput({id:'video-one',progress:{kind:'h3',stage:'decode_audio',elapsed_seconds:205,eta_seconds:3,eta_scope:'total'}});
 assert.match(generationProgressLabel(message.tool_trace[0]),/음성 디코딩/);
 h.toolResult({id:'video-one',result:'saved'});assert.equal(message.tool_trace[0].output,'[0:20] 샘플링 2/20\n');
});
test('cold image ETA does not claim whole-job remaining time',()=>{
 assert.match(generationProgressLabel({name:'image_generate',progress:{kind:'qwim',stage:'sample',step:3,total:40,elapsed_seconds:4,eta_seconds:39,eta_scope:'stage'}}),/현재 단계 예상/);
 assert.match(generationProgressLabel({progress:{kind:'qwim',stage:'decode_video',elapsed_seconds:30,eta_seconds:null}}),/이미지 디코딩 · 예상 시간 계산 중/);
});
