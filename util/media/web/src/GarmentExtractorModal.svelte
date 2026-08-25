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

  let source = null
  let references = []
  let selectedTargets = ['all']
  let feather = 1
  let busy = false
  let error = ''
  let picker = ''
  let remotePicker = ''
  let previousOpen = false
  let releaseScroll = null
  let objectURLs = []

  const targets = [
    ['all', '전체 의상'], ['upper', '상의'], ['lower', '하의'], ['dress', '원피스'],
    ['outer', '외투·상의'], ['shoes', '신발'], ['accessories', '모자·스카프·장신구']
  ]

  function toggleTarget(value) {
    if (value === 'all') {
      selectedTargets = ['all']
      return
    }
    const current = selectedTargets.filter((item) => item !== 'all')
    selectedTargets = current.includes(value) ? current.filter((item) => item !== value) : [...current, value]
    if (!selectedTargets.length) selectedTargets = ['all']
  }

  function serverImage(job) {
    if (!job?.output_url) return null
    return { server: true, ref: `${job.id}:output:0`, url: job.output_url, name: `생성 이미지 #${job.id.slice(0, 8)}`, job }
  }

  function localImage(file) {
    const preview = URL.createObjectURL(file)
    objectURLs.push(preview)
    return { file, url: preview, name: file.name }
  }

  function clearObjectURLs() {
    objectURLs.forEach((url) => URL.revokeObjectURL(url))
    objectURLs = []
  }

  function reset() {
    clearObjectURLs()
    source = serverImage(initialJob)
    references = []
    selectedTargets = ['all']
    feather = 1
    error = ''
    picker = ''
    remotePicker = ''
  }

  $: {
    if (open && !previousOpen) reset()
    previousOpen = open
  }

  $: {
    if (open && !releaseScroll) releaseScroll = lockModalScroll()
    else if (!open && releaseScroll) {
      releaseScroll()
      releaseScroll = null
    }
  }

  onDestroy(() => {
    releaseScroll?.()
    clearObjectURLs()
  })

  function addReference(image) {
    if (!image || references.length >= 4) return
    const key = image.ref || image.url
    if (key && (source?.ref === key || source?.url === key || references.some((item) => item.ref === key || item.url === key))) return
    references = [...references, image]
  }

  function chooseRecent(job) {
    const image = serverImage(job)
    if (picker === 'source') source = image
    else addReference(image)
    picker = ''
  }

  function useRemote(file) {
    const image = localImage(file)
    if (remotePicker === 'source') source = image
    else addReference(image)
  }

  function chooseFile(event, role) {
    const files = [...(event.currentTarget.files || [])]
    event.currentTarget.value = ''
    if (!files.length) return
    if (role === 'source') source = localImage(files[0])
    else files.slice(0, 4 - references.length).forEach((file) => addReference(localImage(file)))
  }

  function appendInput(form, uploadField, reuseField, image) {
    if (!image) return
    if (image.server) form.append(reuseField, image.ref)
    else form.append(uploadField, image.file)
  }

  async function submit() {
    if (!source || busy) return
    busy = true
    error = ''
    const form = new FormData()
    form.append('target', selectedTargets.join(','))
    form.append('feather', String(feather))
    appendInput(form, 'source', 'reuse_source', source)
    references.forEach((image) => appendInput(form, 'references', 'reuse_references', image))
    try {
      await onSubmit(form)
      onClose()
    } catch (cause) {
      error = cause.message || '의상을 추출하지 못했습니다.'
    } finally {
      busy = false
    }
  }

  function handleKeydown(event) {
    if (!open || event.key !== 'Escape' || busy || picker || remotePicker) return
    event.stopImmediatePropagation()
    onClose()
  }
</script>

<svelte:window onkeydown={handleKeydown} />

{#if open}
  <div class="garment-backdrop" role="presentation" onclick={(event) => { if (event.target === event.currentTarget && !busy) onClose() }}>
    <section class="garment-modal" role="dialog" aria-modal="true" aria-label="의상 추출">
      <header><div><strong>의상 추출</strong><small>대상만 고르면 의상 분할부터 투명 PNG·마스크 저장까지 자동 처리합니다.</small></div><button type="button" aria-label="닫기" disabled={busy} onclick={onClose}>×</button></header>
      <div class="garment-content">
        <section class="garment-source-section">
          <div class="garment-section-heading"><div><strong>원본 이미지</strong><small>의상이 가장 잘 보이는 이미지를 먼저 선택하세요.</small></div></div>
          {#if source}
            <div class="garment-selected-source"><img src={source.url} alt="의상 추출 원본"><div><strong title={source.name}>{source.name}</strong><small>주 이미지</small></div><button type="button" aria-label="원본 제거" onclick={() => source = null}>×</button></div>
          {:else}<div class="garment-empty">파일·생성 결과·URL 중 하나를 선택하세요.</div>{/if}
          <div class="garment-input-actions"><label><input type="file" accept="image/png,image/jpeg,image/webp" onchange={(event) => chooseFile(event, 'source')}><span>파일</span></label><button type="button" onclick={() => picker = 'source'}>생성 결과</button><button type="button" onclick={() => remotePicker = 'source'}>URL</button></div>
        </section>

        <section class="garment-reference-section">
          <div class="garment-section-heading"><div><strong>동일 의상 보완 이미지 <i>{references.length}/4</i></strong><small>선택 사항 · 여러 장이면 의상이 가장 온전히 보이는 원본을 자동 선택합니다.</small></div></div>
          {#if references.length}<div class="garment-reference-grid">{#each references as image, index}<article><img src={image.url} alt={`보완 이미지 ${index + 1}`}><span title={image.name}>{image.name}</span><button type="button" aria-label={`보완 이미지 ${index + 1} 제거`} onclick={() => references = references.filter((_, itemIndex) => itemIndex !== index)}>×</button></article>{/each}</div>
          {:else}<div class="garment-reference-empty">가림이 적은 같은 옷 사진이 있으면 추가하세요.</div>{/if}
          <div class="garment-input-actions"><label class:disabled={references.length >= 4}><input type="file" multiple disabled={references.length >= 4} accept="image/png,image/jpeg,image/webp" onchange={(event) => chooseFile(event, 'reference')}><span>파일 추가</span></label><button type="button" disabled={references.length >= 4} onclick={() => picker = 'reference'}>생성 결과</button><button type="button" disabled={references.length >= 4} onclick={() => remotePicker = 'reference'}>URL</button></div>
        </section>

        <section class="garment-options">
          <div class="garment-target-field"><span>추출 대상 <small>복수 선택 가능</small></span><div class="garment-targets">{#each targets as item}<button type="button" class:active={selectedTargets.includes(item[0])} aria-pressed={selectedTargets.includes(item[0])} onclick={() => toggleTarget(item[0])}>{item[1]}</button>{/each}</div></div>
          <label><span>가장자리 부드럽게 <b>{Number(feather).toFixed(1)}px</b></span><input type="range" min="0" max="4" step="0.5" bind:value={feather}></label>
        </section>
        <div class="garment-note"><strong>자동 보완 방식</strong><span>사진별 의상 영역·잘림·선명도를 비교해 가장 좋은 실제 픽셀을 선택합니다. 모든 사진에서 가려진 부분을 임의로 만들어내지는 않습니다.</span></div>
        {#if error}<p class="garment-error">{error}</p>{/if}
      </div>
      <footer><span>결과는 생성 이미지 목록에 투명 PNG로 저장되며 흑백 마스크도 함께 보존됩니다.</span><div><button type="button" disabled={busy} onclick={onClose}>닫기</button><button type="button" class="primary" disabled={busy || !source} onclick={submit}>{busy ? '요청 중…' : '자동 추출'}</button></div></footer>
    </section>
  </div>
{/if}

<RecentImagePicker open={Boolean(picker)} title={picker === 'source' ? '의상 추출 원본 선택' : '동일 의상 보완 이미지 추가'} {jobs} selectedRef={picker === 'source' ? (source?.ref || '') : ''} onSelect={chooseRecent} onClose={() => picker = ''} zIndex={92} />
<RemoteImageModal open={Boolean(remotePicker)} title={remotePicker === 'source' ? '의상 추출 원본 URL' : '동일 의상 보완 이미지 URL'} append={remotePicker === 'reference'} onImport={useRemote} onClose={() => remotePicker = ''} zIndex={92} />

<style>
  .garment-backdrop{position:fixed;z-index:88;inset:0;display:grid;place-items:center;padding:18px;background:#050708e6;backdrop-filter:blur(8px)}
  .garment-modal{display:grid;grid-template-rows:auto minmax(0,1fr) auto;width:min(820px,96vw);max-height:min(860px,94dvh);overflow:hidden;border:1px solid #4a5750;border-radius:15px;background:#11161a;box-shadow:0 24px 90px #000c}
  header{position:static;display:flex;width:100%;height:auto;min-height:54px;align-items:center;justify-content:space-between;gap:12px;padding:9px 14px;border-bottom:1px solid #2d353a;background:#11161a}
  header>div{display:grid;gap:2px}header strong{font-size:14px}header small{color:#7b868c;font-size:9px}header button{width:30px;height:30px;padding:0;border:0;color:#aab3b8;background:transparent;font-size:20px}
  .garment-content{display:grid;gap:12px;overflow-y:auto;padding:14px}.garment-content>section{padding:12px;border:1px solid #30393e;border-radius:11px;background:#141a1e}
  .garment-section-heading{display:flex;justify-content:space-between;margin-bottom:9px}.garment-section-heading>div{display:grid;gap:3px}.garment-section-heading strong{font-size:11px}.garment-section-heading strong i{color:#a8db72;font-size:9px;font-style:normal}.garment-section-heading small{color:#768188;font-size:9px}
  .garment-selected-source{display:grid;grid-template-columns:90px minmax(0,1fr) 28px;align-items:center;gap:10px;overflow:hidden;border:1px solid #3a454a;border-radius:9px;background:#0d1215}.garment-selected-source img{width:90px;height:68px;object-fit:cover;background:#090c0e}.garment-selected-source>div{display:grid;min-width:0;gap:3px}.garment-selected-source strong,.garment-selected-source small{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.garment-selected-source strong{font-size:10px}.garment-selected-source small{color:#86a46d;font-size:8px}.garment-selected-source button{width:24px;height:24px;padding:0;border:0;color:#aab2b6;background:transparent}
  .garment-empty,.garment-reference-empty{display:grid;min-height:68px;place-items:center;border:1px dashed #3b464c;border-radius:9px;color:#69747b;font-size:9px}.garment-reference-empty{min-height:54px}
  .garment-input-actions{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-top:8px}.garment-input-actions button,.garment-input-actions label{display:grid;height:32px;place-items:center;margin:0;border:1px solid #39444a;border-radius:7px;color:#bfc7cb;background:#192025;font-size:9px}.garment-input-actions label{position:relative}.garment-input-actions input{position:absolute;inset:0;opacity:0;cursor:pointer}.garment-input-actions button:disabled,.garment-input-actions label.disabled{cursor:default;opacity:.4}
  .garment-reference-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:7px}.garment-reference-grid article{position:relative;min-width:0;overflow:hidden;border:1px solid #374148;border-radius:8px;background:#0c1013}.garment-reference-grid img{display:block;width:100%;aspect-ratio:4/3;object-fit:cover}.garment-reference-grid span{display:block;overflow:hidden;padding:5px 6px;color:#9ea8ad;font-size:8px;text-overflow:ellipsis;white-space:nowrap}.garment-reference-grid button{position:absolute;top:4px;right:4px;width:22px;height:22px;padding:0;border:1px solid #ffffff30;border-radius:999px;color:#fff;background:#111b;font-size:13px}
  .garment-options{display:grid!important;grid-template-columns:1fr 1.3fr;align-items:end;gap:12px}.garment-options label{display:grid;gap:7px;margin:0;color:#aeb7bb;font-size:10px}.garment-options label span{display:flex;justify-content:space-between}.garment-options b{color:#a8db72}.garment-options input{padding:0}
  .garment-target-field{display:grid;gap:7px}.garment-target-field>span{display:flex;align-items:baseline;justify-content:space-between;color:#aeb7bb;font-size:10px}.garment-target-field small{color:#718078;font-size:8px}.garment-targets{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:5px}.garment-targets button{min-width:0;padding:7px 4px;border:1px solid #39444a;border-radius:7px;color:#9ca7ac;background:#192025;font-size:8px;white-space:nowrap}.garment-targets button.active{border-color:#789957;color:#efffdc;background:#314128;box-shadow:inset 0 0 0 1px #91b76633}
  .garment-note{display:flex;gap:8px;padding:9px 11px;border-radius:8px;color:#8a969b;background:#182119;font-size:9px;line-height:1.5}.garment-note strong{flex:0 0 auto;color:#b8d49f}.garment-error{margin:0;padding:9px;border-radius:8px;color:#f0b2b5;background:#382126;font-size:10px}
  footer{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:10px 14px;border-top:1px solid #2d353a;background:#11161a}footer>span{color:#748087;font-size:8px}footer>div{display:flex;gap:7px}footer button{min-width:78px;padding:8px 11px;border:1px solid #3b454b;border-radius:8px;color:#c5ccd0;background:#1a2024;font-size:10px}footer button.primary{border:0;color:#15200e;background:#b7ed75;font-weight:800}button:disabled{cursor:default;opacity:.45}
  @media(max-width:650px){.garment-backdrop{align-items:end;padding:0}.garment-modal{width:100vw;max-height:100dvh;border:0;border-radius:0}.garment-content{padding:9px}.garment-content>section{padding:9px}.garment-reference-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.garment-options{grid-template-columns:1fr!important}.garment-note{display:grid}header small,footer>span{display:none}footer{justify-content:flex-end}}
</style>
