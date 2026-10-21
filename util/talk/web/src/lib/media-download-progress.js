const stages = { preparing: '미디어 서비스 준비 중', queued: '다운로드 대기 중', metadata: '영상·음성 정보 확인 중', downloading: '미디어 다운로드 중', processing: '영상·음성 병합 및 변환 중', downloaded: '다운로드 완료', analyzing: '다운로드 완료 · 영상·음성 분석 준비 중', failed: '미디어 다운로드 실패' };
const names = { video: '영상', audio: '음성', video_audio: '영상·음성' };

export function downloadBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) return '—';
  if (bytes < 1024) return `${Math.round(bytes)} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MiB`;
}

export function downloadPercent(stream) {
  if (stream.status === 'finished') return 100;
  const ratio = stream.fragments > 0 ? stream.fragment / stream.fragments : stream.total_bytes > 0 ? stream.downloaded_bytes / stream.total_bytes : null;
  return ratio === null ? null : Math.max(0, Math.min(100, ratio * 100));
}

export function downloadStreamLabel(stream) {
  const name = names[stream.kind] || '미디어';
  if (stream.status === 'queued') return `${name} · 대기 중`;
  const percent = downloadPercent(stream);
  const total = stream.total_bytes > 0 ? ` / ${stream.estimated ? '약 ' : ''}${downloadBytes(stream.total_bytes)}` : '';
  const received = `${downloadBytes(stream.downloaded_bytes || 0)}${total}`;
  if (stream.status === 'finished') return `${name} · 완료 · ${received}`;
  let eta = '현재 파일 남은 시간 계산 중';
  if (Number.isFinite(stream.eta_seconds) && stream.eta_seconds >= 0) {
    const seconds = Math.ceil(stream.eta_seconds);
    eta = `현재 파일 약 ${Math.floor(seconds / 60)}분 ${seconds % 60}초 남음`;
  }
  return [name, percent === null ? null : `${percent.toFixed(1)}%`, received, stream.stalled ? '응답 대기 중' : stream.speed_bps > 0 ? `${downloadBytes(stream.speed_bps)}/s` : '속도 측정 중', stream.stalled ? null : eta].filter(Boolean).join(' · ');
}

export function mediaDownloadLabel(tool) {
  const p = tool?.progress;
  if (!p) return stages.preparing;
  if (p.stage === 'downloading') {
    const active = p.streams?.find(s => s.status === 'downloading');
    if (active) return downloadStreamLabel(active);
  }
  return stages[p.stage] || '미디어 처리 중';
}
