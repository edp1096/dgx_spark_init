<script>
  import { modelCapabilities, normalizeReasoningEffort, reasoningEffortLabel } from '../lib/model-capabilities.js';

  export let value = 'medium';
  export let drawer = false;

  const levels = modelCapabilities('qwen3.8').reasoningLevels;
  $: normalized = normalizeReasoningEffort('qwen3.8', value);
  $: index = Math.max(0, levels.indexOf(normalized));
  $: label = reasoningEffortLabel(normalized);

  function change(event) {
    value = levels[Number(event.currentTarget.value)] || 'medium';
  }
</script>

<div class="qwen-effort-field" class:drawer title={`Qwen3.8 리즈닝 · ${label}`}>
  {#if drawer}<span>리즈닝</span>{/if}
  <div class="qwen-effort-control">
    <input type="range" min="0" max="3" step="1" value={index} oninput={change} aria-label="Reasoning effort" aria-valuetext={label} />
    <output>{label}</output>
  </div>
  {#if drawer}<small>꺼짐 · Low · Medium · XHigh</small>{/if}
</div>
