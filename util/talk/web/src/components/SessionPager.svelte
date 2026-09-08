<script>
  export let value;
  export let label = '대화';
  export let onPage = () => {};
</script>

{#if value.pages > 1}
  <div class="session-pagination" aria-label={`${label} 페이지`}>
    <button onclick={() => onPage(value.page - 1)} disabled={value.page === 0} aria-label={`${label} 이전 페이지`}>‹ 이전</button>
    <small aria-live="polite">{value.start + 1}–{value.end} / {value.total}</small>
    <button onclick={() => onPage(value.page + 1)} disabled={value.page === value.pages - 1} aria-label={`${label} 다음 페이지`}>{value.page === 0 ? '더보기 ›' : '다음 ›'}</button>
  </div>
{/if}

<style>
  .session-pagination { display: flex; align-items: center; justify-content: space-between; gap: 4px; margin: 4px 0 10px 8px; }
  button { border: 0; background: transparent; color: inherit; font-size: 11px; padding: 6px 4px; border-radius: 6px; }
  button:hover:not(:disabled) { background: var(--surface-hover, #80808020); }
  button:disabled { opacity: .35; }
  small { font-size: 10px; opacity: .7; white-space: nowrap; }
</style>
