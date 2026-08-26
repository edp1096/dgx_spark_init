<script>
  import { onDestroy } from 'svelte'
  import { lockModalScroll } from './modalScroll.js'

  export let open = false
  export let jobs = []
  export let selectedID = ''
  export let onSelect = () => {}
  export let onClose = () => {}

  let releaseScroll = null
  let visibleJobs = []

  $: visibleJobs = jobs
    .filter((job) => job.kind === 'video' && job.status === 'completed' && job.output_url)
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

  function duration(job) {
    const frames = Number(job?.params?.num_frames || 0)
    const fps = Number(job?.params?.fps || 0)
    if (frames > 0 && fps > 0) return `${Math.max(0, (frames - 1) / fps).toFixed(1)}초`
    return '길이 정보 없음'
  }

  function handleKeydown(event) {
    if (!open || event.key !== 'Escape') return
    event.stopImmediatePropagation()
    onClose()
  }
</script>

<svelte:window onkeydown={handleKeydown} />

{#if open}
  <div class="backdrop" role="presentation" onclick={(event) => { if (event.target === event.currentTarget) onClose() }}>
    <div class="picker" role="dialog" aria-modal="true" aria-label="생성 영상 선택">
      <header>
        <div><strong>생성 영상 선택</strong><small>받아쓰기 원본으로 사용할 영상을 고르세요.</small></div>
        <button type="button" aria-label="닫기" onclick={onClose}>×</button>
      </header>
      <div class="grid">
        {#each visibleJobs as job (job.id)}
          <button type="button" class:selected={selectedID === job.id} title={job.prompt || '생성 영상'} onclick={() => onSelect(job)}>
            <video src={job.output_url} muted playsinline preload="metadata" aria-label={job.prompt || '생성 영상'}></video>
            <span>{job.prompt || '프롬프트 없음'}</span>
            <small>{job.params?.width || '—'}×{job.params?.height || '—'} · {duration(job)}</small>
            {#if selectedID === job.id}<i>현재 선택</i>{/if}
          </button>
        {:else}
          <p class="empty">선택할 수 있는 완료 영상이 없습니다.</p>
        {/each}
      </div>
      <footer><span>{visibleJobs.length}개 영상</span><button type="button" onclick={onClose}>닫기</button></footer>
    </div>
  </div>
{/if}

<style>
  .backdrop { position:fixed; z-index:72; inset:0; display:grid; place-items:center; padding:20px; background:#050708e8; backdrop-filter:blur(8px); }
  .picker { display:grid; grid-template-rows:auto minmax(0,1fr) auto; width:min(1040px,96vw); height:min(820px,92vh); overflow:hidden; border:1px solid #4a5550; border-radius:14px; background:#11161a; box-shadow:0 24px 80px #000c; }
  header { display:flex; min-height:58px; align-items:center; justify-content:space-between; gap:12px; padding:11px 15px; border-bottom:1px solid #2d343a; }
  header > div { display:grid; gap:3px; }
  header strong { color:#e3e8eb; font-size:14px; }
  header small { color:#76818a; font-size:9px; }
  header button { width:32px; height:32px; padding:0; border:1px solid #394148; border-radius:8px; color:#aab2b8; background:#191e22; font-size:18px; }
  .grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); grid-auto-rows:max-content; align-content:start; gap:10px; overflow-y:auto; padding:14px; }
  .grid > button { position:relative; min-width:0; overflow:hidden; padding:0 0 9px; border:1px solid #30383e; border-radius:10px; color:#c7ced2; background:#171c20; text-align:left; }
  .grid > button:hover { border-color:#60734f; }
  .grid > button.selected { border-color:#a8dc72; box-shadow:0 0 0 2px #a8dc7222; }
  video { display:block; width:100%; aspect-ratio:4/3; object-fit:cover; background:#090c0e; pointer-events:none; }
  .grid span,.grid small { display:block; overflow:hidden; margin:0 9px; text-overflow:ellipsis; white-space:nowrap; }
  .grid span { margin-top:8px; font-size:10px; font-weight:700; }
  .grid small { margin-top:3px; color:#727d84; font-size:8px; }
  .grid i { position:absolute; top:7px; right:7px; border-radius:999px; padding:4px 7px; color:#17200f; background:#b7ed75; font-size:8px; font-style:normal; font-weight:800; }
  .empty { grid-column:1/-1; margin:0; padding:80px 20px; color:#69737a; font-size:11px; text-align:center; }
  footer { display:flex; align-items:center; justify-content:space-between; padding:10px 14px; border-top:1px solid #2d343a; }
  footer span { color:#758078; font-size:9px; }
  footer button { min-width:84px; padding:8px 12px; border:1px solid #3a4248; border-radius:8px; color:#c2c9cd; background:#191e22; font-size:10px; }
  @media(max-width:700px) {
    .backdrop { padding:0; }
    .picker { width:100vw; height:100dvh; border:0; border-radius:0; }
    header { min-height:52px; padding:9px 11px; }
    header small { display:none; }
    .grid { grid-template-columns:repeat(2,minmax(0,1fr)); gap:6px; padding:8px; }
    .grid span { margin:6px 6px 0; font-size:8px; }
    .grid small { margin:2px 6px 0; font-size:7px; }
    footer { padding:8px 10px; }
  }
</style>
