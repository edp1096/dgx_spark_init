<script>
  import { tick } from 'svelte';
  export let value;
  export let label = '대화';
  export let onPage = () => {};
  let root, draftPage;
  $: draftPage = String(value.page + 1);
  function jump() {
    const text = draftPage.trim();
    if (!/^\d+$/.test(text)) { draftPage = String(value.page + 1); return; }
    move(Number(text) - 1);
  }
  function onKeydown(event) {
    if (event.isComposing) return;
    if (event.key === 'Enter') { event.preventDefault(); jump(); }
    else if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); draftPage = String(value.page + 1); }
  }
  async function move(page) {
    page = Math.max(0, Math.min(value.pages - 1, page));
    draftPage = String(page + 1);
    onPage(page);
    await tick();
    const nav = root?.closest('nav'), group = root?.closest('.chat-group');
    if (nav && group) nav.scrollTop += group.getBoundingClientRect().top - nav.getBoundingClientRect().top;
  }
</script>

{#if value.pages > 1}
  <div class="session-pagination" bind:this={root} aria-label={`${label} 페이지`}>
    <button type="button" onclick={() => move(value.page - 1)} disabled={value.page === 0} aria-label={`${label} 이전 페이지`} title="이전 페이지">‹</button>
    <div class="page-position" title={`${value.start + 1}–${value.end} / ${value.total}개 대화`}>
      <input type="text" inputmode="numeric" enterkeyhint="go" maxlength="9" aria-label={`${label} 페이지 이동`} title="페이지 번호를 입력하고 Enter를 누르세요" value={draftPage} oninput={event => draftPage = event.currentTarget.value} onkeydown={onKeydown} onblur={() => draftPage = String(value.page + 1)} />
      <span aria-label={`전체 ${value.pages}페이지`}>/ {value.pages}</span>
    </div>
    <button type="button" onclick={() => move(value.page + 1)} disabled={value.page === value.pages - 1} aria-label={`${label} 다음 페이지`} title="다음 페이지">›</button>
  </div>
{/if}

<style>
  .session-pagination { display: grid; grid-template-columns: 30px minmax(0, 1fr) 30px; align-items: center; gap: 5px; padding: 3px 7px 7px; }
  button, input { height: 30px; min-width: 0; border: 1px solid #80808035; border-radius: 7px; background: var(--sidebar-surface); color: inherit; }
  button { font-size: 20px; padding: 0; line-height: 1; }
  .page-position { display: flex; align-items: center; justify-content: center; gap: 6px; min-width: 0; font-size: 11px; font-variant-numeric: tabular-nums; }
  input { width: 52px; max-width: 50%; padding: 0 5px; font: inherit; text-align: center; }
  .page-position span { white-space: nowrap; }
  button:hover:not(:disabled) { background: #80808020; }
  button:disabled { opacity: .35; cursor: default; }
  button:focus-visible, input:focus-visible { outline: 2px solid #6584ed; outline-offset: 1px; }
</style>
