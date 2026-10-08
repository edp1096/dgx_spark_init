import assert from 'node:assert/strict';
import test from 'node:test';
import { streamSpeech } from '../api/media.js';

for (const rate of [1, 1.2, 1.3]) {
  test(`speech stream forwards ${rate} playback metadata with PCM chunks`, async (t) => {
    const original = globalThis.fetch;
    t.after(() => { globalThis.fetch = original; });
    globalThis.fetch = async (_url, options) => {
      assert.deepEqual(JSON.parse(options.body), { text: '읽기' });
      return new Response(new Uint8Array([1, 2, 3, 4]), {
        headers: { 'X-Audio-Sample-Rate': '24000', ...(rate === 1 ? {} : { 'X-Audio-Speak-Rate': String(rate) }) },
      });
    };
    const chunks = [];
    const result = await streamSpeech('읽기', undefined, (bytes, sampleRate, speakRate) => {
      chunks.push(...bytes);
      assert.equal(sampleRate, 24000);
      assert.equal(speakRate, rate);
    });
    assert.deepEqual(chunks, [1, 2, 3, 4]);
    assert.deepEqual(result, { sampleRate: 24000, speakRate: rate });
  });
}
