import test from 'node:test';
import assert from 'node:assert/strict';
import {generationProgressLabel, isGenerationTool} from './generation-progress.js';
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
test('shorts reuses the chat progress stream without inventing a total ETA',()=>{
 const message={tool_trace:[]}; const h=createStreamHandlers(message,()=>{});
 h.toolStart({id:'short-one',name:'shorts_generate'});
 h.toolOutput({id:'short-one',delta:'Fish 음성 준비·합성\n',progress:{kind:'shorts',detail:'Fish 음성 준비·합성',step:15,total:100,elapsed_seconds:42}});
 assert.equal(isGenerationTool(message.tool_trace[0]),true);
 assert.equal(generationProgressLabel(message.tool_trace[0]),'Fish 음성 준비·합성 · 15% · 경과 0:42');
 h.toolOutput({id:'short-one',progress:{kind:'shorts',detail:'세로 영상 렌더링 120/720프레임 · 약 50초 남음',step:78,total:100,elapsed_seconds:150}});
 assert.match(generationProgressLabel(message.tool_trace[0]),/약 50초 남음 · 78% · 경과 2:30/);
 h.toolResult({id:'short-one',result:'completed'});
 assert.equal(message.tool_trace[0].running,false);
 assert.equal(message.tool_trace[0].output,'Fish 음성 준비·합성\n');
});
test('photo production shows review and repair stages in the chat stream',()=>{
 const message={tool_trace:[]};const h=createStreamHandlers(message,()=>{});
 h.toolStart({id:'clip-one',name:'clip_produce'});
 assert.equal(isGenerationTool(message.tool_trace[0]),true);
 assert.equal(generationProgressLabel(message.tool_trace[0]),'사진 영상 제작 준비 중…');
 h.toolOutput({id:'clip-one',progress:{kind:'clip',detail:'원본과 편집 결과의 손·얼굴·구도 검사',step:35,total:100,elapsed_seconds:70}});
 assert.equal(generationProgressLabel(message.tool_trace[0]),'원본과 편집 결과의 손·얼굴·구도 검사 · 35% · 경과 1:10');
 h.toolResult({id:'clip-one',result:'failed'});
 assert.equal(message.tool_trace[0].running,false);
});
