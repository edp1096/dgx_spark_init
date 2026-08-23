<script>
  export let label = '목록'
  export let total = 0
  export let page = 1
  export let pageSize = 10
  export let pageSizes = [8, 10, 20, 50, 100]
  export let compact = false
  export let sortOrder = 'desc'
  export let onPageChange = () => {}
  export let onPageSizeChange = () => {}
  export let onSortOrderChange = () => {}

  $: totalPages = Math.max(1, Math.ceil(total / pageSize))
  $: first = total ? (page - 1) * pageSize + 1 : 0
  $: last = Math.min(total, page * pageSize)
  $: pageItems = buildPageItems(totalPages, page)

  function buildPageItems(count, current) {
    if (count <= 5) return Array.from({ length: count }, (_, index) => index + 1)
    if (current <= 3) return [1, 2, 3, 'end-gap', count]
    if (current >= count - 2) return [1, 'start-gap', count - 2, count - 1, count]
    return [1, 'start-gap', current, 'end-gap', count]
  }
</script>

<div class="result-pagination" class:compact aria-label={`${label} ${compact ? '하단 ' : ''}페이지 제어`}>
  <span class="result-range">{total ? `${first}–${last} / ${total}개` : '0개'}</span>
  {#if !compact}
    <label class="page-size-control">
      <span>표시</span>
      <select aria-label={`${label} 페이지당 표시 개수`} value={pageSize} onchange={(event) => onPageSizeChange(Number(event.currentTarget.value))}>
        {#each pageSizes as size}<option value={size}>{size}개</option>{/each}
      </select>
    </label>
    <label class="sort-order-control">
      <span>정렬</span>
      <select aria-label={`${label} 정렬 방식`} value={sortOrder} onchange={(event) => onSortOrderChange(event.currentTarget.value)}>
        <option value="desc">최신순</option>
        <option value="asc">오래된순</option>
      </select>
    </label>
  {/if}
  <div class="page-buttons">
    <button type="button" class="page-jump" aria-label="첫 페이지" title="첫 페이지" disabled={page <= 1} onclick={() => onPageChange(1)}>«</button>
    <button type="button" aria-label="이전 페이지" disabled={page <= 1} onclick={() => onPageChange(page - 1)}>‹</button>
    {#each pageItems as item}
      {#if typeof item === 'number'}
        <button type="button" class="page-number" class:active={item === page} aria-current={item === page ? 'page' : undefined} aria-label={`${item} 페이지`} onclick={() => onPageChange(item)}>{item}</button>
      {:else}
        <span class="page-ellipsis" aria-hidden="true">…</span>
      {/if}
    {/each}
    <button type="button" aria-label="다음 페이지" disabled={page >= totalPages} onclick={() => onPageChange(page + 1)}>›</button>
    <button type="button" class="page-jump" aria-label="마지막 페이지" title="마지막 페이지" disabled={page >= totalPages} onclick={() => onPageChange(totalPages)}>»</button>
  </div>
</div>
