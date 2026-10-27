// The model lifecycle is separate from auxiliary inference and preparation.
export function modelRuntimeState(runtime, health = {}) {
  const core = runtime?.components?.find(component => component.role === 'llm');
  const operation = runtime?.operation;
  const active = operation?.state === 'running';
  const targetsModel = active && ((operation.bundle_id && operation.bundle_id === runtime?.selected_bundle) || (core && operation.component_id === core.id));
  if (targetsModel && operation.action === 'stop') return { label: '중지 중', pending: true, online: false };
  if ((targetsModel && ['start', 'restart'].includes(operation.action)) || core?.health === 'starting') return { label: '기동 중', pending: true, online: false };
  if (core) {
    if (core.health === 'failed') return { label: '실패', pending: false, online: false };
    if (['exited', 'dead', 'created'].includes(core.status)) return { label: '중지됨', pending: false, online: false };
    if (core.status === 'missing') return { label: '미기동', pending: false, online: false };
    if (core.health === 'online') return { label: '실행 중', pending: false, online: true };
    return { label: core.health === 'unresponsive' ? '연결 이상' : '확인 중', pending: false, online: false };
  }
  return { label: health.status === 'ok' ? '연결됨' : health.status === 'checking' ? '확인 중' : '연결 오류', pending: false, online: health.status === 'ok' };
}

export function runtimeOperationLabel(operation) {
  return ({start: '기동 중', restart: '재기동 중', stop: '중지 중', workload: '작업 중', prepare: '준비 중'})[operation?.action] || '처리 중';
}
