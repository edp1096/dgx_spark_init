import test from 'node:test';
import assert from 'node:assert/strict';
import { createStreamHandlers } from './chat-stream.js';
import { downloadPercent, downloadStreamLabel, mediaDownloadLabel } from './media-download-progress.js';

test('media progress keeps completed video while audio downloads and analysis starts', () => {
  const message = { tool_trace: [] };
  const h = createStreamHandlers(message, () => {});
  h.toolStart({ id: 'download', name: 'media_import' });
  h.toolOutput({ id: 'download', progress: { stage: 'downloading', streams: [
    { id: 'v', kind: 'video', status: 'finished', downloaded_bytes: 1024 },
    { id: 'a', kind: 'audio', status: 'downloading', downloaded_bytes: 512, total_bytes: 1024, speed_bps: 128, eta_seconds: 4 },
  ] } });
  assert.match(mediaDownloadLabel(message.tool_trace[0]), /음성 · 50.0%.*128 B\/s.*약 0분 4초 남음/);
  h.toolOutput({ id: 'download', progress: { stage: 'analyzing' } });
  assert.equal(message.tool_trace[0].progress.streams.length, 2);
  assert.match(mediaDownloadLabel(message.tool_trace[0]), /다운로드 완료.*분석/);
});

test('unknown totals and stalled streams do not claim false completion or stale ETA', () => {
  assert.equal(downloadPercent({ status: 'downloading', downloaded_bytes: 512 }), null);
  assert.match(downloadStreamLabel({ kind: 'audio', status: 'downloading', downloaded_bytes: 512, eta_seconds: null }), /남은 시간 계산 중/);
  const stalled = downloadStreamLabel({ kind: 'audio', status: 'downloading', downloaded_bytes: 512, stalled: true, eta_seconds: 4 });
  assert.match(stalled, /응답 대기 중/);
  assert.doesNotMatch(stalled, /4초/);
  assert.equal(downloadPercent({ status: 'downloading', fragment: 43, fragments: 100, total_bytes: 50, downloaded_bytes: 200 }), 43);
});
