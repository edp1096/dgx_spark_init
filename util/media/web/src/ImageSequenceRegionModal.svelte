<script>
  import { imageSequenceRegionOptions } from './lib/catalogs.js'

  export let openIndex = -1
  export let regions = []
  export let onSelect = () => {}
</script>

{#if openIndex >= 0}
  <div class="image-sequence-region-backdrop" role="presentation" onclick={(event) => { if (event.target === event.currentTarget) openIndex = -1 }}>
    <div class="image-sequence-region-modal" role="dialog" aria-modal="true" aria-label="변경 허용 영역 선택">
      <header><div><strong>변경 허용 영역</strong><small>초록색 영역만 새로 그립니다 · 왼쪽·오른쪽은 화면 기준</small></div><button type="button" aria-label="닫기" onclick={() => openIndex = -1}>×</button></header>
      <div class="image-sequence-region-grid">
        {#each imageSequenceRegionOptions as option}
          <button type="button" class:selected={regions[openIndex] === option.id} onclick={() => onSelect(openIndex, option.id)}>
            <i class={`image-sequence-region-map region-${option.id}`}><span></span></i>
            <b>{option.label}</b><small>{option.description}</small>
          </button>
        {/each}
      </div>
    </div>
  </div>
{/if}
