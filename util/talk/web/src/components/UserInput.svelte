<script>
  import { tick } from 'svelte';
  import { markdownView } from '../lib/markdown-view.js';

  export let content = '';
  export let render;
  export let label = '입력';
  let expanded = false;
  let disclosure;
  $: text = content || '';
  $: long = text.length > 800 || text.split('\n', 13).length > 12;
  $: preview = text.slice(0, 240).trimEnd();

  async function collapse() {
    expanded = false;
    await tick();
    disclosure?.scrollIntoView({ block: 'nearest', behavior: 'instant' });
    disclosure?.querySelector('summary')?.focus({ preventScroll: true });
  }
</script>

{#if long}
  <details class="bubble input-disclosure" bind:open={expanded} bind:this={disclosure}>
    <summary aria-expanded={expanded}>
      {label} {expanded ? '접기' : '펼치기'}<small>{text.length.toLocaleString()}자</small>
      {#if !expanded}<span class="input-preview">{preview}{text.length > 240 ? '…' : ''}</span>{/if}
    </summary>
    {#if expanded}
      <div class="prose" use:markdownView={render(text)}></div>
      <button type="button" class="input-collapse" onclick={collapse}>↑ {label} 접기</button>

    {/if}
  </details>
{:else}
  <div class="bubble prose">
    {#if label !== '입력'}<small>{label}</small>{/if}
    <div use:markdownView={render(text)}></div>
  </div>
{/if}

<style>
  .input-disclosure { min-width: 0; }
  summary { cursor: pointer; font-size: 12px; font-weight: 600; user-select: none; }
  summary small { margin-left: 10px; opacity: .65; font-weight: 400; }
  summary:focus-visible, .input-collapse:focus-visible { outline: 2px solid #839eff; outline-offset: 3px; border-radius: 3px; }
  .input-preview { display: -webkit-box; -webkit-box-orient: vertical; -webkit-line-clamp: 4; overflow: hidden; white-space: pre-line; margin: 8px 0 0; font-size: 13px; font-weight: 400; }
  .input-collapse { display: block; margin-top: 12px; padding: 4px 0; border: 0; background: transparent; color: inherit; cursor: pointer; font-size: 12px; }
</style>
