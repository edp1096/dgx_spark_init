import test from 'node:test';
import assert from 'node:assert/strict';
import { formatMemory, residentMemoryLabel, memoryBudgetLabel } from './runtime-memory.js';

test('memory uses MiB for small services without hiding them behind zero GiB', () => {
  assert.equal(formatMemory(4 / 1024), '4.0 MiB');
  assert.equal(formatMemory(100.125), '100.1 GiB');
  for (const value of [undefined, null, NaN, -1]) assert.equal(formatMemory(value), '—');
});

test('running unknown or external memory is never presented as zero', () => {
  assert.equal(residentMemoryLabel({ status: 'running', memory_measured: false, resident_memory_gib: 0 }), '점유 측정 불가');
  assert.equal(residentMemoryLabel({ status: 'external', controller: 'external', resident_memory_gib: 0 }), '점유 측정 불가');
  assert.equal(residentMemoryLabel({ status: 'exited', controller: 'compose' }), '점유 0.0 MiB');
  assert.equal(residentMemoryLabel({ status: 'running', memory_measured: true, resident_memory_gib: 0 }), '점유 0.0 MiB');
  assert.equal(residentMemoryLabel({ status: 'running', memory_measured: true, resident_memory_gib: 10 / 1024 }), '점유 10.0 MiB');
});

test('budget remains distinct from measured occupancy and input-dependent ASR', () => {
  assert.equal(memoryBudgetLabel({ memory_gib: .1, resident_memory_gib: .01 }), '예산 102.4 MiB');
  assert.equal(memoryBudgetLabel({ id: 'nemotron-asr', memory_gib: 6 }), '입력별 예산 산정');
  assert.equal(memoryBudgetLabel({ memory_gib: 0 }), '예산 미설정');
});

test('warm budgets do not re-add a cold profile or allocator subsets', () => {
  const c = {memory_measured:true,health:'online',resident_memory_gib:99,memory_gib:108,workspace_memory_gib:0,
    engine_memory:{schema:1,unit:'GiB',cuda_reserved_gib:92,cuda_peak_reserved_gib:93,kv_and_qsa_gib:13.8}};
  assert.equal(memoryBudgetLabel(c),'실행 예산 100.0 GiB');
  assert.equal(memoryBudgetLabel({memory_measured:true,resident_memory_gib:6,memory_gib:5.25,workspace_memory_gib:2.5}),'작업 예산 8.5 GiB');
});
