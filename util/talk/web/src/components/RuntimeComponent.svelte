<script>
  import { residentMemoryLabel, memoryBudgetLabel, residentMemoryDetail } from '../lib/runtime-memory.js';
  import { modelWeightSummary } from '../lib/model-presentation.js';
  export let component;
  export let state = '';
</script>

<div class="runtime-component" role="group" aria-label={component.name}>
  <span class:online={component.health === 'online'} class:starting={component.health === 'starting'} class:failed={component.health === 'failed' || component.health === 'unresponsive'} title={component.health_error ? `최근 상태 확인: ${component.health_error} · ${component.health_latency_ms ?? 0}ms · 연속 실패 ${component.health_failures ?? 0}회` : `상태 확인 ${component.health_latency_ms ?? 0}ms`}>
    <i></i><span><b>{component.name}</b><small>{component.phase || modelWeightSummary(component) || component.role} · {component.host || 'local'}</small></span>
  </span>
  <div>
    <b>{state}</b>
    <small title={residentMemoryDetail(component)}>{residentMemoryLabel(component)}</small>
    <small title="현재 구성의 예상 메모리 예산">{memoryBudgetLabel(component)}</small>
  </div>
</div>
