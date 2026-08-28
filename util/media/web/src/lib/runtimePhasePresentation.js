const phaseLabels = {
  preparing: '입력 준비',
  model_loading: '모델 탑재',
  conditioning: '조건 인코딩',
  sampling: '추론',
  decoding: '디코딩',
  model_unloading: '모델 해제',
  cache_retaining: '캐시 유지',
  finalizing: '결과 정리',
  completed: '처리 완료'
}

const actionLabels = {
  load: '메모리 적재',
  unload: '메모리 해제',
  retain: '캐시 유지',
  swap: '모델 교체'
}

export function runtimePhasePresentation(job) {
  const phase = job?.params?.runtime_phase
  if (!phase || phase.operation_id !== job?.id || !phaseLabels[phase.phase]) return null
  const component = String(phase.component || '').trim()
  const phaseLabel = phaseLabels[phase.phase]
  const progress = Math.min(100, Math.max(0, Number(phase.progress || 0) * 100))
  const action = actionLabels[phase.memory_action] || ''
  const detail = [String(phase.detail || '').trim(), action].filter(Boolean).join(' · ')
  return {
    phase: phase.phase,
    label: component ? `${component} · ${phaseLabel}` : phaseLabel,
    detail,
    progress,
    residentAfter: typeof phase.resident_after === 'boolean' ? phase.resident_after : null
  }
}
