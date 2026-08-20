<script>
  export let label = '목록'
  export let total = 0
  export let page = 1
  export let pageSize = 10
  export let pageSizes = [8, 10, 20, 50, 100]
  export let onPageChange = () => {}
  export let onPageSizeChange = () => {}

  $: totalPages = Math.max(1, Math.ceil(total / pageSize))
  $: first = total ? (page - 1) * pageSize + 1 : 0
  $: last = Math.min(total, page * pageSize)
</script>

<div class="result-pagination" aria-label={`${label} 페이지 제어`}>
  <span class="result-range">{total ? `${first}–${last} / ${total}개` : '0개'}</span>
  <label class="page-size-control">
    <span>표시</span>
    <select aria-label={`${label} 페이지당 표시 개수`} value={pageSize} onchange={(event) => onPageSizeChange(Number(event.currentTarget.value))}>
      {#each pageSizes as size}<option value={size}>{size}개</option>{/each}
    </select>
  </label>
  <div class="page-buttons">
    <button type="button" aria-label="이전 페이지" disabled={page <= 1} onclick={() => onPageChange(page - 1)}>‹</button>
    <strong>{page} / {totalPages}</strong>
    <button type="button" aria-label="다음 페이지" disabled={page >= totalPages} onclick={() => onPageChange(page + 1)}>›</button>
  </div>
</div>
