import assert from 'node:assert/strict';
import test from 'node:test';
import { modelDisplayName, selectedModelWeight, modelWeightSummary } from './model-presentation.js';

test('catalog names hide internal IDs while keeping distinct engines and custom names', () => {
 const catalog = {bundles:[
  {id:'exl3',name:'Qwen 3.8 Flash-Next EXL3',model_id:'qwen38fn_exl3'},
  {id:'qad',name:'Qwen 3.8 Flash-Next QAD',model_id:'original',bindings:{llm:{model:'publisher/abliterated'}}},
  {id:'custom',name:'내 업무 모델',model_id:'original'},
 ],components:[{id:'llm',role:'llm'}, {id:'asr',role:'asr',name:'Nemotron 3.5 ASR',model:'nemotron-3.5-asr-streaming-0.6b'}]};
 catalog.bundles[0].bindings={asr:{model:'nemotron-3.5-asr-streaming-0.6b'}};
 assert.equal(modelDisplayName('qwen38fn_exl3',catalog),'Qwen 3.8 Flash-Next EXL3');
 assert.equal(modelDisplayName('publisher/abliterated',catalog),'Qwen 3.8 Flash-Next QAD');
 assert.equal(modelDisplayName('original',catalog,'custom'),'내 업무 모델');
 assert.equal(modelDisplayName('nemotron-3.5-asr-streaming-0.6b',catalog),'Nemotron 3.5 ASR');
 assert.equal(modelDisplayName('unknown/external-model',catalog),'unknown/external-model');
});

test('unsaved weight edits use the same metadata choices as preparation', () => {
 const component={runtime_options:{MODEL_VARIANT:'abliterated'},model_presentation:{selected_variant:'official',weights:[
  {id:'official',label:'원본',format:'NVFP4'},
  {id:'abliterated',label:'Abliterated',format:'EXL3 3bit'},
 ]}};
 assert.equal(selectedModelWeight(component).id,'abliterated');
 assert.equal(modelWeightSummary(component),'Abliterated · EXL3 3bit');
 assert.equal(modelWeightSummary({}), '');
});
