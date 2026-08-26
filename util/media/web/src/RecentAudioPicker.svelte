<script>
  import { onDestroy } from 'svelte'
  import { lockModalScroll } from './modalScroll.js'

  export let open = false
  export let jobs = []
  export let selectedID = ''
  export let selectedIDs = []
  export let multiple = false
  export let onSelect = () => {}
  export let onClose = () => {}

  let releaseScroll = null
  let visibleJobs = []

  const selected = (id) => multiple ? selectedIDs.includes(id) : selectedID === id

  $: visibleJobs = jobs
    .filter((job) => job.kind === 'speech' && job.status === 'completed' && job.output_url)
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
  <div class="backdrop" role="presentation" onclick={(event) => { if (event.target === event.currentTarget) onClose() }}>
    <div class="picker" role="dialog" aria-modal="true" aria-label="생성 음성 선택">
      <header><div><strong>생성 음성 선택</strong><small>영상의 움직임과 길이를 이끌 음성을 미리 듣고 고르세요.</small></div><button type="button" aria-label="닫기" onclick={onClose}>×</button></header>
      <div class="list">
        {#each visibleJobs as job (job.id)}
          <article class:selected={selected(job.id)}>
            <div class="meta"><i>AUDIO</i><span><strong>{job.params?.speaker || '생성 음성'}</strong><small>{job.params?.language || '언어 자동'}{#if job.params?.seed >= 0} · seed {job.params.seed}{/if}</small></span>{#if selected(job.id)}<b>선택됨</b>{/if}</div>
            <p title={job.prompt}>{job.prompt || '원문 없음'}</p>
            {#if job.params?.instructions}<small class="instruction" title={job.params.instructions}>지시 · {job.params.instructions}</small>{/if}
            <audio controls preload="metadata" src={job.output_url}></audio>
            <button type="button" class="select" onclick={() => onSelect(job)}>{selected(job.id) ? (multiple ? '선택 해제' : '다시 선택') : (multiple ? '타임라인에 추가' : '이 음성 사용')}</button>
          </article>
        {:else}
          <p class="empty">선택할 수 있는 완료 음성이 없습니다.</p>
        {/each}
      </div>
      <footer><span>{visibleJobs.length}개 음성</span><button type="button" onclick={onClose}>닫기</button></footer>
    </div>
  </div>
{/if}

<style>
  .backdrop { position:fixed; z-index:72; inset:0; display:grid; place-items:center; padding:20px; background:#050708e8; backdrop-filter:blur(8px); }
  .picker { display:grid; grid-template-rows:auto minmax(0,1fr) auto; width:min(1040px,96vw); height:min(760px,92vh); overflow:hidden; border:1px solid #4a5550; border-radius:14px; background:#11161a; box-shadow:0 24px 80px #000c; }
  header { display:flex; min-height:58px; align-items:center; justify-content:space-between; gap:12px; padding:11px 15px; border-bottom:1px solid #2d343a; }
  header > div { display:grid; gap:3px; }
  header strong { color:#e3e8eb; font-size:14px; }
  header small { color:#76818a; font-size:9px; }
  header button { width:32px; height:32px; padding:0; border:1px solid #394148; border-radius:8px; color:#aab2b8; background:#191e22; font-size:18px; }
  .list { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); grid-auto-rows:max-content; align-content:start; gap:9px; overflow-y:auto; padding:14px; }
  article { display:grid; min-width:0; gap:8px; border:1px solid #30383e; border-radius:10px; padding:11px; background:#171c20; }
  article.selected { border-color:#a8dc72; box-shadow:0 0 0 2px #a8dc7222; }
  .meta { display:flex; align-items:center; gap:8px; min-width:0; }
  .meta > i { padding:4px 6px; border-radius:5px; color:#182010; background:#b7ed75; font:800 8px ui-monospace; font-style:normal; }
  .meta > span { display:grid; min-width:0; gap:2px; }
  .meta strong,.meta small { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .meta strong { color:#d9dfe1; font-size:10px; }
  .meta small { color:#737d83; font-size:8px; }
  .meta b { margin-left:auto; border-radius:999px; padding:4px 7px; color:#17200f; background:#b7ed75; font-size:8px; }
  article p { overflow:hidden; margin:0; color:#c5cbce; font-size:10px; line-height:1.45; text-overflow:ellipsis; white-space:nowrap; }
  .instruction { overflow:hidden; color:#727c82; font-size:8px; text-overflow:ellipsis; white-space:nowrap; }
  audio { width:100%; min-width:0; height:32px; }
  .select { border:1px solid #43513e; border-radius:7px; padding:7px; color:#c9ddb8; background:#1b241a; font-size:9px; }
  .empty { grid-column:1/-1; margin:0; padding:80px 20px; color:#69737a; font-size:11px; text-align:center; }
  footer { display:flex; align-items:center; justify-content:space-between; padding:10px 14px; border-top:1px solid #2d343a; }
  footer span { color:#758078; font-size:9px; }
  footer button { min-width:84px; padding:8px 12px; border:1px solid #3a4248; border-radius:8px; color:#c2c9cd; background:#191e22; font-size:10px; }
  @media(max-width:700px) { .backdrop { padding:0; } .picker { width:100vw; height:100dvh; border:0; border-radius:0; } header { min-height:52px; padding:9px 11px; } header small { display:none; } .list { grid-template-columns:repeat(2,minmax(0,1fr)); gap:7px; padding:8px; } article { gap:6px; padding:8px; } .meta > i { padding:3px 5px; font-size:7px; } .meta strong { font-size:9px; } .meta small,.instruction { font-size:7px; } .meta b { padding:3px 5px; font-size:7px; } footer { padding:8px 10px; } }
</style>
