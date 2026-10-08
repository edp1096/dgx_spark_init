<script>
  import { onDestroy } from 'svelte'
  import { lockModalScroll } from './modalScroll.js'
  import RecentImagePicker from './RecentImagePicker.svelte'
  import RemoteImageModal from './RemoteImageModal.svelte'

  export let open = false
  export let jobs = []
  export let initialJob = null
  export let onSubmit = async () => {}
  export let onClose = () => {}

  let target = null
  let source = null
  let targetFace = 1
  let sourceFace = 1
  let picker = ''
  let remotePicker = ''
  let busy = false
  let error = ''
  let previousOpen = false
  let releaseScroll = null
  let objectURLs = []

  function serverImage(job) {
    if (!job?.output_url) return null
    return { server: true, ref: `${job.id}:output:0`, url: job.output_url, name: `생성 이미지 #${job.id.slice(0, 8)}` }
  }
  function localImage(file) {
    const url = URL.createObjectURL(file)
    objectURLs.push(url)
    return { file, url, name: file.name }
  }
  function clearObjectURLs() {
    objectURLs.forEach((url) => URL.revokeObjectURL(url))
    objectURLs = []
  }
  function reset() {
    clearObjectURLs()
    target = serverImage(initialJob)
    source = null
    targetFace = 1
    sourceFace = 1
    picker = ''
    remotePicker = ''
    error = ''
  }
  $: {
    if (open && !previousOpen) reset()
    previousOpen = open
  }
  $: {
    if (open && !releaseScroll) releaseScroll = lockModalScroll()
    else if (!open && releaseScroll) { releaseScroll(); releaseScroll = null }
  }
  onDestroy(() => { releaseScroll?.(); clearObjectURLs() })

  function chooseRecent(job) {
    const image = serverImage(job)
    if (picker === 'target') target = image
    else source = image
    picker = ''
  }
  function chooseRemote(file) {
    const image = localImage(file)
    if (remotePicker === 'target') target = image
    else source = image
  }
  function chooseFile(event, role) {
    const file = event.currentTarget.files?.[0]
    event.currentTarget.value = ''
    if (!file) return
    if (role === 'target') target = localImage(file)
    else source = localImage(file)
  }
  function appendImage(form, uploadField, reuseField, image) {
    if (image.server) form.append(reuseField, image.ref)
    else form.append(uploadField, image.file)
  }
  async function submit() {
    if (!target || !source || busy) return
    busy = true
    error = ''
    const form = new FormData()
    appendImage(form, 'target', 'reuse_target', target)
    appendImage(form, 'source', 'reuse_source', source)
    form.append('target_face_index', String(Math.max(0, Number(targetFace) - 1)))
    form.append('source_face_index', String(Math.max(0, Number(sourceFace) - 1)))
    try {
      await onSubmit(form)
      onClose()
    } catch (cause) {
      error = cause.message || '얼굴을 교체하지 못했습니다.'
    } finally {
      busy = false
    }
  }
  function keydown(event) {
    if (!open || event.key !== 'Escape' || busy || picker || remotePicker) return
    event.stopImmediatePropagation()
    onClose()
  }
</script>

<svelte:window onkeydown={keydown} />

{#if open}
  <div class="face-swap-backdrop" role="presentation" onclick={(event) => { if (event.target === event.currentTarget && !busy) onClose() }}>
    <div class="face-swap-modal" role="dialog" aria-modal="true" aria-label="얼굴 교체">
      <header><div><strong>얼굴 교체</strong><small>생성 모델로 다시 그리지 않고 ReActor가 얼굴 영역을 직접 교체합니다.</small></div><button type="button" aria-label="닫기" disabled={busy} onclick={onClose}>×</button></header>
      <div class="face-swap-content">
        <div class="face-swap-grid">
          <section>
            <div class="face-swap-heading"><strong>대상 이미지</strong><small>몸·의상·배경을 유지할 결과</small></div>
            {#if target}<div class="face-swap-selected"><img src={target.url} alt="얼굴 교체 대상"><span title={target.name}>{target.name}</span><button type="button" aria-label="대상 제거" onclick={() => target = null}>×</button></div>{:else}<div class="face-swap-empty">대상 이미지를 선택하세요.</div>{/if}
            <div class="face-swap-inputs"><label><input type="file" accept="image/png,image/jpeg,image/webp" onchange={(event) => chooseFile(event, 'target')}><span>파일</span></label><button type="button" onclick={() => picker = 'target'}>생성 결과</button><button type="button" onclick={() => remotePicker = 'target'}>URL</button></div>
            <label class="face-index">교체할 얼굴 번호 <input type="number" min="1" max="16" step="1" bind:value={targetFace}></label>
          </section>
          <section>
            <div class="face-swap-heading"><strong>가져올 얼굴</strong><small>정면에 가깝고 선명한 얼굴 권장</small></div>
            {#if source}<div class="face-swap-selected"><img src={source.url} alt="가져올 얼굴"><span title={source.name}>{source.name}</span><button type="button" aria-label="얼굴 원본 제거" onclick={() => source = null}>×</button></div>{:else}<div class="face-swap-empty">얼굴 원본을 선택하세요.</div>{/if}
            <div class="face-swap-inputs"><label><input type="file" accept="image/png,image/jpeg,image/webp" onchange={(event) => chooseFile(event, 'source')}><span>파일</span></label><button type="button" onclick={() => picker = 'source'}>생성 결과</button><button type="button" onclick={() => remotePicker = 'source'}>URL</button></div>
            <label class="face-index">가져올 얼굴 번호 <input type="number" min="1" max="16" step="1" bind:value={sourceFace}></label>
          </section>
        </div>
        <div class="face-swap-note"><strong>ReActor · INSwapper 128</strong><span>여러 얼굴이 있으면 왼쪽부터가 아니라 ReActor 검출 순서에 따라 번호가 정해질 수 있습니다. 실패하면 얼굴 번호를 바꾸어 재시도하세요. 원본 InsightFace 가중치는 비상업 연구용입니다.</span></div>
        {#if error}<p class="face-swap-error">{error}</p>{/if}
      </div>
      <footer><span>결과는 생성 이미지 목록에 새 항목으로 저장됩니다.</span><div><button type="button" disabled={busy} onclick={onClose}>닫기</button><button type="button" class="primary" disabled={busy || !target || !source} onclick={submit}>{busy ? '요청 중…' : '얼굴 교체 시작'}</button></div></footer>
    </div>
  </div>
{/if}

<RecentImagePicker open={Boolean(picker)} title={picker === 'target' ? '대상 이미지 선택' : '가져올 얼굴 선택'} {jobs} selectedRef={picker === 'target' ? (target?.ref || '') : (source?.ref || '')} onSelect={chooseRecent} onClose={() => picker = ''} zIndex={92} />
<RemoteImageModal open={Boolean(remotePicker)} title={remotePicker === 'target' ? '대상 이미지 URL' : '가져올 얼굴 URL'} onImport={chooseRemote} onClose={() => remotePicker = ''} zIndex={92} />

<style>
  .face-swap-backdrop{position:fixed;z-index:88;inset:0;display:grid;place-items:center;padding:18px;background:#050708e6;backdrop-filter:blur(8px)}
  .face-swap-modal{display:grid;grid-template-rows:auto minmax(0,1fr) auto;width:min(840px,96vw);max-height:min(780px,94dvh);overflow:hidden;border:1px solid #4a5750;border-radius:15px;background:#11161a;box-shadow:0 24px 90px #000c}
  header{position:static;display:flex;min-height:54px;align-items:center;justify-content:space-between;gap:12px;padding:9px 14px;border-bottom:1px solid #2d353a;background:#11161a}header>div{display:grid;gap:2px}header strong{font-size:14px}header small{color:#7b868c;font-size:9px}header button{width:30px;height:30px;padding:0;border:0;color:#aab3b8;background:transparent;font-size:20px}
  .face-swap-content{display:grid;gap:12px;overflow-y:auto;padding:14px}.face-swap-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.face-swap-grid section{padding:12px;border:1px solid #30393e;border-radius:11px;background:#141a1e}.face-swap-heading{display:grid;gap:3px;margin-bottom:9px}.face-swap-heading strong{font-size:11px}.face-swap-heading small{color:#768188;font-size:9px}
  .face-swap-selected{display:grid;grid-template-columns:96px minmax(0,1fr) 28px;align-items:center;gap:9px;overflow:hidden;border:1px solid #3a454a;border-radius:9px;background:#0d1215}.face-swap-selected img{width:96px;height:86px;object-fit:cover;background:#090c0e}.face-swap-selected span{overflow:hidden;color:#b7c0c4;font-size:9px;text-overflow:ellipsis;white-space:nowrap}.face-swap-selected button{width:24px;height:24px;padding:0;border:0;color:#aab2b6;background:transparent}
  .face-swap-empty{display:grid;min-height:86px;place-items:center;border:1px dashed #3b464c;border-radius:9px;color:#69747b;font-size:9px}.face-swap-inputs{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-top:8px}.face-swap-inputs button,.face-swap-inputs label{display:grid;height:32px;place-items:center;margin:0;border:1px solid #39444a;border-radius:7px;color:#bfc7cb;background:#192025;font-size:9px}.face-swap-inputs label{position:relative}.face-swap-inputs input{position:absolute;inset:0;opacity:0;cursor:pointer}.face-index{display:flex;align-items:center;justify-content:space-between;gap:9px;margin-top:9px;color:#9da8ad;font-size:9px}.face-index input{width:76px;padding:6px 8px;border:1px solid #3b454a;border-radius:7px;color:#e4e9eb;background:#0d1215}
  .face-swap-note{display:flex;gap:9px;padding:10px 12px;border-radius:8px;color:#8a969b;background:#182119;font-size:9px;line-height:1.5}.face-swap-note strong{flex:0 0 auto;color:#b8d49f}.face-swap-error{margin:0;padding:9px;border-radius:8px;color:#f0b2b5;background:#382126;font-size:10px}
  footer{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:10px 14px;border-top:1px solid #2d353a;background:#11161a}footer>span{color:#748087;font-size:8px}footer>div{display:flex;gap:7px}footer button{min-width:78px;padding:8px 11px;border:1px solid #3b454b;border-radius:8px;color:#c5ccd0;background:#1a2024;font-size:10px}footer button.primary{min-width:132px;border:0;color:#15200e;background:#b7ed75;font-weight:800}button:disabled{cursor:default;opacity:.45}
  @media(max-width:650px){.face-swap-backdrop{align-items:end;padding:0}.face-swap-modal{width:100vw;max-height:100dvh;border:0;border-radius:0}.face-swap-content{padding:9px}.face-swap-grid{grid-template-columns:1fr}.face-swap-grid section{padding:9px}.face-swap-note{display:grid}header small,footer>span{display:none}footer{justify-content:flex-end}}
</style>
