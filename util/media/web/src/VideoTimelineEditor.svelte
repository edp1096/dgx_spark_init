<script>
  import { onDestroy } from 'svelte'
  import VideoConditionTimeline from './VideoConditionTimeline.svelte'
  import VideoAudioTimeline from './VideoAudioTimeline.svelte'
  import { lockModalScroll } from './modalScroll.js'

  export let open = false
  export let duration = 5
  export let fps = 24
  export let startImage = null
  export let endImage = null
  export let startStrength = 1
  export let endStrength = 1
  export let keyframes = []
  export let audioClips = []
  export let imageURL = (image) => image?.preview || image?.url || ''
  export let onMove = () => {}
  export let onMoveAudio = () => {}
  export let onUpdate = () => {}
  export let onRemove = () => {}
  export let onAdd = () => {}
  export let onSetStrength = () => {}
  export let onFile = () => {}
  export let onRecent = () => {}
  export let onRemote = () => {}
  export let onClear = () => {}
  export let onClose = () => {}
  export let overlayOpen = false

  let releaseScroll = null
  $: {
    if (open && !releaseScroll) releaseScroll = lockModalScroll()
    else if (!open && releaseScroll) { releaseScroll(); releaseScroll = null }
  }
  onDestroy(() => releaseScroll?.())

  function handleKeydown(event) {
    if (!open || overlayOpen || event.key !== 'Escape') return
    event.stopImmediatePropagation()
    onClose()
  }

  function keyframeCapacity() {
    return Math.max(0, Math.min(8, Math.round(Number(duration) * Math.max(1, Number(fps))) - 1))
  }
</script>

<svelte:window onkeydown={handleKeydown} />

{#if open}
  <div class="backdrop" role="presentation" onclick={(event) => { if (event.target === event.currentTarget) onClose() }}>
    <div class="editor" role="dialog" aria-modal="true" aria-label="키프레임 크게 편집">
      <header><div><strong>키프레임 크게 편집</strong><small>{duration.toFixed(2)}초 · {fps} FPS · 장면을 드래그하거나 정확한 시간을 입력하세요.</small></div><button type="button" aria-label="닫기" onclick={onClose}>×</button></header>
      <div class="content">
        {#if audioClips.length}<VideoAudioTimeline large {duration} clips={audioClips} onMove={onMoveAudio} />{/if}
        <VideoConditionTimeline large {duration} {fps} {startImage} {endImage} {keyframes} {imageURL} {onMove} />
        <div class="toolbar"><span>장면 {keyframes.length + 2}개 · 키프레임 {keyframes.length}/{keyframeCapacity()}</span><button type="button" disabled={keyframes.length >= keyframeCapacity()} onclick={onAdd}>+ 키프레임</button></div>
        <div class="cards">
          <article class="boundary-card">
            <div class="card-title"><strong>시작 이미지</strong><small>0초</small></div>
            {#if imageURL(startImage)}<img src={imageURL(startImage)} alt="시작 이미지">{:else}<div class="empty-image">START</div>{/if}
            <span title={startImage?.name || ''}>{startImage?.name || '이미지 미선택'}</span>
            <div class="source-actions"><label><input type="file" accept="image/png,image/jpeg,image/webp" onchange={(e) => onFile('start', e.currentTarget.files?.[0] || null)}><b>파일</b></label><button type="button" onclick={() => onRecent('start')}>최근 결과</button><button type="button" onclick={() => onRemote('start')}>URL</button>{#if startImage}<button type="button" class="danger" onclick={() => onClear('start')}>제거</button>{/if}</div>
            {#if startImage}<label class="number">반영 강도<input type="number" min="0" max="1" step="any" value={startStrength} onchange={(e) => onSetStrength('start', e.currentTarget.value)}></label>{/if}
          </article>
          {#each keyframes as keyframe, index (keyframe.id)}
            <article>
              <div class="card-title"><strong>키프레임 {index + 1}</strong><button type="button" aria-label="키프레임 제거" onclick={() => onRemove(keyframe.id)}>×</button></div>
              {#if imageURL(keyframe.image)}<img src={imageURL(keyframe.image)} alt="키프레임 {index + 1}">{:else}<div class="empty-image">K{index + 1}</div>{/if}
              <span title={keyframe.image?.name || ''}>{keyframe.image?.name || '이미지 미선택'}</span>
              <div class="source-actions"><label><input type="file" accept="image/png,image/jpeg,image/webp" onchange={(e) => onFile(`keyframe:${keyframe.id}`, e.currentTarget.files?.[0] || null)}><b>파일</b></label><button type="button" onclick={() => onRecent(`keyframe:${keyframe.id}`)}>최근 결과</button><button type="button" onclick={() => onRemote(`keyframe:${keyframe.id}`)}>URL</button></div>
              <div class="numbers"><label>위치<input type="number" min={1 / fps} max={Math.max(1 / fps, duration - 1 / fps)} step="any" value={Number(keyframe.time).toFixed(3)} onchange={(e) => onMove(keyframe.id, e.currentTarget.value)}></label><label>강도<input type="number" min="0" max="1" step="any" value={keyframe.strength} onchange={(e) => onUpdate(keyframe.id, 'strength', e.currentTarget.value)}></label></div>
            </article>
          {/each}
          <article class="boundary-card">
            <div class="card-title"><strong>마지막 이미지</strong><small>{duration.toFixed(2)}초</small></div>
            {#if imageURL(endImage)}<img src={imageURL(endImage)} alt="마지막 이미지">{:else}<div class="empty-image">END</div>{/if}
            <span title={endImage?.name || ''}>{endImage?.name || '이미지 미선택'}</span>
            <div class="source-actions"><label><input type="file" accept="image/png,image/jpeg,image/webp" onchange={(e) => onFile('end', e.currentTarget.files?.[0] || null)}><b>파일</b></label><button type="button" onclick={() => onRecent('end')}>최근 결과</button><button type="button" onclick={() => onRemote('end')}>URL</button>{#if endImage}<button type="button" class="danger" onclick={() => onClear('end')}>제거</button>{/if}</div>
            {#if endImage}<label class="number">반영 강도<input type="number" min="0" max="1" step="any" value={endStrength} onchange={(e) => onSetStrength('end', e.currentTarget.value)}></label>{/if}
          </article>
        </div>
      </div>
      <footer><small>변경 내용은 영상 생성 화면에 즉시 반영됩니다.</small><button type="button" onclick={onClose}>완료</button></footer>
    </div>
  </div>
{/if}

<style>
  .backdrop { position:fixed; z-index:65; inset:0; display:grid; place-items:center; padding:16px; background:#050708ed; backdrop-filter:blur(8px); }
  .editor { display:grid; grid-template-rows:auto minmax(0,1fr) auto; width:min(1320px,98vw); height:min(900px,96vh); overflow:hidden; border:1px solid #4a5550; border-radius:14px; background:#101519; box-shadow:0 24px 80px #000d; }
  header { display:flex; min-height:56px; align-items:center; justify-content:space-between; gap:12px; padding:10px 15px; border-bottom:1px solid #2d343a; }
  header > div { display:grid; gap:3px; }
  header strong { color:#e3e8eb; font-size:14px; }
  header small { color:#76818a; font-size:9px; }
  header button { width:32px; height:32px; padding:0; border:1px solid #394148; border-radius:8px; color:#aab2b8; background:#191e22; font-size:18px; }
  .content { overflow-y:auto; padding:14px; }
  .toolbar { display:flex; align-items:center; justify-content:space-between; gap:10px; margin:11px 0 8px; }
  .toolbar span { color:#7d8881; font-size:9px; }
  .toolbar button { border:1px solid #526347; border-radius:7px; padding:7px 10px; color:#cbe4b1; background:#1b2519; font-size:9px; }
  .toolbar button:disabled { opacity:.4; }
  .cards { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:9px; }
  article { display:grid; align-content:start; min-width:0; gap:7px; border:1px solid #30383e; border-radius:10px; padding:9px; background:#171c20; }
  article.boundary-card { border-color:#3e4a40; }
  .card-title { display:flex; min-height:25px; align-items:center; justify-content:space-between; gap:8px; }
  .card-title strong { color:#d8dfe1; font-size:10px; }
  .card-title small { color:#78827d; font-size:8px; }
  .card-title button { width:24px; height:24px; border:1px solid #54383b; border-radius:6px; padding:0; color:#d59096; background:transparent; }
  article > img,.empty-image { display:block; width:100%; aspect-ratio:16/9; border-radius:7px; object-fit:cover; background:#090d0f; }
  .empty-image { display:grid; place-items:center; border:1px dashed #3a444a; color:#68737a; font:750 10px ui-monospace; }
  article > span { overflow:hidden; color:#929ca2; font-size:8px; text-overflow:ellipsis; white-space:nowrap; }
  .source-actions { display:flex; gap:4px; }
  .source-actions label { position:relative; display:block; flex:1; margin:0; }
  .source-actions input { position:absolute; width:1px; height:1px; opacity:0; }
  .source-actions b,.source-actions button { display:grid; min-width:0; min-height:28px; place-items:center; flex:1; border:1px solid #3d4942; border-radius:6px; padding:4px 6px; color:#bdd3aa; background:#1a2219; font-size:8px; font-weight:700; white-space:nowrap; }
  .source-actions b { cursor:pointer; }
  .source-actions button.danger { flex:0 0 auto; border-color:#58383b; color:#d9979c; background:#281a1c; }
  .numbers { display:grid; grid-template-columns:1fr 1fr; gap:6px; }
  .numbers label,.number { display:grid; grid-template-columns:auto minmax(50px,1fr); align-items:center; gap:5px; margin:0; color:#9ca6a0; font-size:8px; }
  .numbers input,.number input { min-width:0; padding:7px; font-size:9px; }
  footer { display:flex; align-items:center; justify-content:space-between; gap:10px; padding:9px 14px; border-top:1px solid #2d343a; }
  footer small { color:#78827d; font-size:8px; }
  footer button { min-width:88px; border:1px solid #526347; border-radius:7px; padding:8px 12px; color:#d5eac2; background:#1c271b; font-size:9px; }
  @media(max-width:900px) { .cards { grid-template-columns:repeat(2,minmax(0,1fr)); } }
  @media(max-width:700px) { .backdrop { padding:0; } .editor { width:100vw; height:100dvh; border:0; border-radius:0; } header { min-height:50px; padding:8px 10px; } header small { display:none; } .content { padding:8px; } .cards { grid-template-columns:1fr; } footer { padding:7px 9px; } footer small { display:none; } }
</style>
