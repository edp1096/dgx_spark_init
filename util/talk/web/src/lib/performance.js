export function formatMetric(value, estimated, unit) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) return '—';
  const formatted = new Intl.NumberFormat('ko-KR', {
    maximumFractionDigits: unit === 's' ? 2 : 1,
  }).format(value);
  return `${estimated ? '≈ ' : ''}${formatted} ${unit}`;
}

export function performanceTitle(performance, kind) {
  const approximate = performance?.[`${kind}_estimated`];
  const definitions = {
    pp: '입력 처리 속도. 서버가 알려준 캐시 재사용 토큰은 제외합니다.',
    tg: '추론·답변·도구 호출의 생성 속도. 입력 처리와 도구 실행 대기는 제외합니다.',
    ttft: '첫 모델 호출에서 첫 생성 토큰까지 걸린 시간. 추론 토큰도 포함합니다.',
  };
  const detail = approximate
    ? '≈ 서버의 정확한 시간 통계가 없어 수신 시간 또는 글자 수로 추정한 값입니다.'
    : '서버의 토큰 수와 처리 시간으로 계산한 값입니다.';
  const calls = performance?.calls > 1 && kind !== 'ttft'
    ? ` ${performance.calls}회 모델 호출의 토큰 수와 처리 시간을 합산했습니다.` : '';
  const cache = kind === 'pp' && performance?.cached_tokens > 0
    ? ` 캐시 재사용 ${performance.cached_tokens.toLocaleString('ko-KR')}토큰.` : '';
  return `${definitions[kind]} ${detail}${calls}${cache}`;
}
