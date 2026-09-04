<script>
  import { modelCapabilities, normalizeReasoningEffort, reasoningEffortLabel } from '../lib/model-capabilities.js';

  export let value = 'medium';
  export let drawer = false;
  export let modelType = 'qwen3.8';

  $: profile = modelCapabilities(modelType);
  $: levels = profile.reasoningLevels;
  $: normalized = normalizeReasoningEffort(modelType, value);
  $: index = Math.max(0, levels.indexOf(normalized));
  $: label = reasoningEffortLabel(normalized);
  $: modelLabel = profile.family === 'glm5.3' ? 'GLM-5.3' : 'Qwen3.8';

  function change(event) {
    value = levels[Number(event.currentTarget.value)] || levels[0];
  }
</script>

<div class="qwen-effort-field" class:drawer title={`${modelLabel} 리즈닝 · ${label}`}>
  {#if drawer}<span>리즈닝</span>{/if}
  <div class="qwen-effort-control">
    <input type="range" min="0" max={Math.max(0, levels.length - 1)} step="1" value={index} oninput={change} aria-label="Reasoning effort" aria-valuetext={label} />
    <output>{label}</output>
  </div>
  {#if drawer}<small>{levels.map(reasoningEffortLabel).join(' · ')}</small>{/if}
</div>
