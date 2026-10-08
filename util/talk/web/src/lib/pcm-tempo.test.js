import assert from 'node:assert/strict';
import test from 'node:test';
import { PCMTempo } from './pcm-tempo.js';

const sampleRate = 24000;

function render(samples, rate) {
  const tempo = new PCMTempo({ sampleRate, speakRate: rate });
  const blocks = [];
  const sizes = [1, 37, 1111, 257, 4800, 13];
  for (let offset = 0, block = 0; offset < samples.length; block += 1) {
    const end = Math.min(samples.length, offset + sizes[block % sizes.length]);
    blocks.push(tempo.append(samples.subarray(offset, end)));
    offset = end;
  }
  blocks.push(tempo.finish());
  const output = new Float32Array(blocks.reduce((n, b) => n + b.length, 0));
  let offset = 0;
  for (const block of blocks) { output.set(block, offset); offset += block.length; }
  return { output, blocks };
}

function frequency(samples) {
  const crossings = [];
  for (let i = 1; i < samples.length; i += 1) {
    if (samples[i - 1] <= 0 && samples[i] > 0) crossings.push(i);
  }
  return (crossings.length - 1) * sampleRate / (crossings.at(-1) - crossings[0]);
}

test('1.0 preserves every sample exactly', () => {
  const samples = Float32Array.from({ length: 7001 }, (_, i) => Math.sin(i * 0.051));
  assert.deepEqual(render(samples, 1).output, samples);
});

for (const rate of [0.5, 1.2, 1.3, 2]) {
  test(`${rate} changes duration while retaining pitch, streaming output and the tail`, () => {
    const samples = Float32Array.from({ length: 2 * sampleRate }, (_, i) =>
      0.4 * Math.sin(2 * Math.PI * (i < 1.7 * sampleRate ? 220 : 660) * i / sampleRate));
    const { output, blocks } = render(samples, rate);
    assert.equal(output.length, Math.round(samples.length / rate));
    assert.ok(blocks.slice(0, -1).some((b) => b.length > 0), 'must play before EOF');
    const middle = output.subarray(Math.floor(0.3 * sampleRate / rate), Math.floor(1.4 * sampleRate / rate));
    assert.ok(Math.abs(frequency(middle) - 220) < 3, 'main tone must retain pitch');
    const tail = output.subarray(output.length - Math.floor(0.15 * sampleRate / rate), output.length - Math.floor(0.04 * sampleRate / rate));
    assert.ok(Math.abs(frequency(tail) - 660) < 8, 'final audio must survive flushing');
    assert.ok(tail.reduce((sum, value) => sum + value * value, 0) / tail.length > 0.01);
  });
}

test('flushes utterances shorter than the processing lookahead', () => {
  const samples = Float32Array.from({ length: 1200 }, (_, i) => 0.4 * Math.sin(i * 0.1));
  const { output } = render(samples, 1.3);
  assert.equal(output.length, Math.round(samples.length / 1.3));
  assert.ok(output.some((sample) => Math.abs(sample) > 0.1));
});
