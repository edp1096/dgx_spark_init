export function formatMemory(value) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) return '—';
  return value < 1 ? `${(value * 1024).toFixed(1)} MiB` : `${value.toFixed(1)} GiB`;
}

export function residentMemoryLabel(component) {
  if (component.memory_measured === true) return `점유 ${formatMemory(component.resident_memory_gib)}`;
  if (component.controller !== 'external' && ['exited', 'missing', 'created', 'dead'].includes(component.status)) return '점유 0.0 MiB';
  return '점유 측정 불가';
}

export function memoryBudgetLabel(component) {
  if (component.id === 'nemotron-asr') return '입력별 예산 산정';
  if (component.request_memory_gib > 0) return `작업 상한 ${formatMemory(component.request_memory_gib)}`;
  if (component.memory_measured && component.health === 'online' && component.engine_memory?.schema === 1 && component.engine_memory?.unit === 'GiB') {
    const m = component.engine_memory;
    const extra = Math.max(component.workspace_memory_gib || 0, m.cuda_peak_reserved_gib - m.cuda_reserved_gib);
    return `실행 예산 ${formatMemory(component.resident_memory_gib + extra)}`;
  }
  if (component.memory_measured && component.workspace_memory_gib > 0) {
    return `작업 예산 ${formatMemory(Math.max(component.memory_gib || 0, component.resident_memory_gib + component.workspace_memory_gib))}`;
  }
  return component.memory_gib > 0 ? `예산 ${formatMemory(component.memory_gib)}` : '예산 미설정';
}

export function residentMemoryDetail(component) {
  if (!component.memory_measured) return '현재 점유량을 측정할 수 없습니다.';
  const total = `GPU ${formatMemory(component.gpu_memory_gib)} + CPU·공유·커널 ${formatMemory(component.host_memory_gib)} · 회수 가능한 파일 캐시 제외`;
  const m = component.engine_memory;
  if (!m || m.schema !== 1 || m.unit !== 'GiB') return total;
  return `${total}\nKV·QSA ${formatMemory(m.kv_and_qsa_gib)} · Mamba ${formatMemory(m.mamba_cache_gib)} · CUDA 할당 ${formatMemory(m.cuda_allocated_gib)} / 예약 ${formatMemory(m.cuda_reserved_gib)}\n모델 적재 증가분: 본체 ${formatMemory(m.target_load_delta_gib)}, MTP ${formatMemory(m.draft_load_delta_gib)} · 위 항목들은 전체 점유에 포함되며 중복 합산하지 않습니다.`;
}
