import test from 'node:test';
import assert from 'node:assert/strict';
import { modelRuntimeState, runtimeOperationLabel } from './runtime-state.js';
const base = {selected_bundle:'coding', components:[{id:'llm',role:'llm',status:'running',health:'online'}]};
test('auxiliary work never labels a ready model as starting',()=>{
  for(const action of ['workload','prepare']) {
    const s=modelRuntimeState({...base,operation:{state:'running',action,bundle_id:'coding',component_id:'extra-embedding'}},{status:'ok'});
    assert.equal(s.label,'실행 중');assert.equal(s.online,true);assert.equal(s.pending,false);
  }
  assert.equal(modelRuntimeState({...base,operation:{state:'running',action:'start',component_id:'extra-shorts'}}).label,'실행 중');
});
test('set and component lifecycle use readiness, not container existence alone',()=>{
  assert.equal(modelRuntimeState({...base,operation:{state:'running',action:'start',bundle_id:'coding'}}).label,'기동 중');
  assert.equal(modelRuntimeState({...base,operation:{state:'running',action:'stop',bundle_id:'coding'}}).label,'중지 중');
  assert.equal(modelRuntimeState({...base,operation:{state:'running',action:'restart',component_id:'llm'}}).label,'기동 중');
  for(const [status,health,label] of [['running','starting','기동 중'],['running','online','실행 중'],['exited','offline','중지됨'],['missing','offline','미기동'],['running','failed','실패'],['running','unresponsive','연결 이상']]) {
    assert.equal(modelRuntimeState({...base,components:[{id:'llm',role:'llm',status,health}]},{status:'ok'}).label,label);
  }
  assert.equal(modelRuntimeState(null,{status:'ok'}).label,'연결됨');
  assert.equal(runtimeOperationLabel({action:'workload'}),'작업 중');
});
