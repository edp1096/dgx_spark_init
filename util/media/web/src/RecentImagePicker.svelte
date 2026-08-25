<script>
  import { onDestroy } from 'svelte'
  import { lockModalScroll } from './modalScroll.js'

  export let open = false
  export let title = '최근 결과에서 선택'
  export let jobs = []
  export let selectedRef = ''
  export let onSelect = () => {}
  export let onClose = () => {}
  export let zIndex = 70

  let releaseScroll = null
  let visibleJobs = []

  $: visibleJobs = jobs
    .filter((job) => job.kind === 'image' && job.status === 'completed' && job.output_url)
    .slice()
    .sort((a, b) => Date.parse(b.created_at || 0) - Date.parse(a.created_at || 0))

  $: {
    if (open && !releaseScroll) releaseScroll = lockModalScroll()
    else if (!open && releaseScroll) {
      releaseScroll()
      releaseScroll = null
    }
  }

  onDestroy(() => releaseScroll?.())

  function handleKeydown(event) {
    if (!open || event.key !== 'Escape') return
    event.stopImmediatePropagation()
    onClose()
  }
</script>

<svelte:window onkeydown={handleKeydown} />

{#if open}
  <div class="recent-image-picker-backdrop" style:z-index={zIndex} role="presentation" onclick={(event) => { if (event.target === event.currentTarget) onClose() }}>
    <div class="recent-image-picker" role="dialog" aria-modal="true" aria-label={title}>
      <header>
        <div><strong>{title}</strong><small>Spark Media에 저장된 결과를 원본 그대로 재사용합니다.</small></div>
        <button type="button" aria-label="닫기" onclick={onClose}>×</button>
      </header>
      <div class="recent-image-picker-grid">
        {#each visibleJobs as job (job.id)}
          {@const ref = `${job.id}:output:0`}
          <button type="button" class:selected={selectedRef === ref} title={job.prompt || '생성 이미지'} onclick={() => onSelect(job)}>
            <img src={job.output_url} alt={job.prompt || '생성 이미지'} loading="lazy">
            <span>{job.prompt || '프롬프트 없음'}</span>
            <small>{job.params?.width || '—'}×{job.params?.height || '—'} · {new Date(job.created_at).toLocaleString()}</small>
            {#if selectedRef === ref}<i>현재 선택</i>{/if}
          </button>
        {:else}
          <p class="recent-image-picker-empty">선택할 수 있는 완료 이미지가 없습니다.</p>
        {/each}
      </div>
      <footer><span>{visibleJobs.length}개 결과</span><button type="button" onclick={onClose}>닫기</button></footer>
    </div>
  </div>
{/if}

<style>
  .recent-image-picker-backdrop { position:fixed; z-index:70; inset:0; display:grid; place-items:center; padding:20px; background:#050708e8; backdrop-filter:blur(8px); }
  .recent-image-picker { display:grid; grid-template-rows:auto minmax(0,1fr) auto; width:min(1040px,96vw); height:min(820px,92vh); overflow:hidden; border:1px solid #4a5550; border-radius:14px; background:#11161a; box-shadow:0 24px 80px #000c; }
  header { position:static; display:flex; width:100%; height:auto; min-height:58px; align-items:center; justify-content:space-between; gap:12px; padding:11px 15px; border-bottom:1px solid #2d343a; background:#11161a; }
  header > div { display:grid; gap:3px; }
  header strong { color:#e3e8eb; font-size:14px; }
  header small { color:#76818a; font-size:9px; }
  header button { flex:0 0 auto; width:32px; height:32px; padding:0; border:1px solid #394148; border-radius:8px; color:#aab2b8; background:#191e22; font-size:18px; }
  .recent-image-picker-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); grid-auto-rows:max-content; align-content:start; align-items:start; gap:10px; overflow-y:auto; padding:14px; }
  .recent-image-picker-grid > button { position:relative; align-self:start; min-width:0; height:auto; overflow:hidden; padding:0 0 9px; border:1px solid #30383e; border-radius:10px; color:#c7ced2; background:#171c20; text-align:left; }
  .recent-image-picker-grid > button:hover { border-color:#60734f; }
  .recent-image-picker-grid > button.selected { border-color:#a8dc72; box-shadow:0 0 0 2px #a8dc7222; }
  img { display:block; width:100%; aspect-ratio:4/3; object-fit:cover; background:#090c0e; }
  span, small { display:block; overflow:hidden; margin:0 9px; text-overflow:ellipsis; white-space:nowrap; }
  .recent-image-picker-grid span { margin-top:8px; font-size:10px; font-weight:700; }
  .recent-image-picker-grid small { margin-top:3px; color:#727d84; font-size:8px; }
  .recent-image-picker-grid i { position:absolute; top:7px; right:7px; border-radius:999px; padding:4px 7px; color:#17200f; background:#b7ed75; font-size:8px; font-style:normal; font-weight:800; }
  .recent-image-picker-empty { grid-column:1/-1; margin:0; padding:80px 20px; color:#69737a; font-size:11px; text-align:center; }
  footer { display:flex; align-items:center; justify-content:space-between; padding:10px 14px; border-top:1px solid #2d343a; background:#11161a; }
  footer span { margin:0; color:#758078; font-size:9px; }
  footer button { min-width:84px; padding:8px 12px; border:1px solid #3a4248; border-radius:8px; color:#c2c9cd; background:#191e22; font-size:10px; }
  @media(max-width:700px) {
    .recent-image-picker-backdrop { padding:0; }
    .recent-image-picker { width:100vw; height:100dvh; border:0; border-radius:0; }
    header { min-height:52px; padding:9px 11px; }
    header small { display:none; }
    .recent-image-picker-grid { grid-template-columns:repeat(3,minmax(0,1fr)); gap:6px; padding:8px; }
    .recent-image-picker-grid span { margin:6px 6px 0; font-size:8px; }
    .recent-image-picker-grid small { margin:2px 6px 0; font-size:7px; }
    footer { padding:8px 10px; }
  }
</style>
