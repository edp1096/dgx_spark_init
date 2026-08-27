<script>
  import { onDestroy, onMount } from 'svelte'
  import { api } from './api.js'
  import { lockModalScroll } from './modalScroll.js'
  import RecentImagePicker from './RecentImagePicker.svelte'

  export let onChanged = () => {}
  export let onOpenSettings = () => {}
  export let imageJobs = []

  let loras = []
  let status = { civitai_token_configured: false, hf_token_configured: false }
  let busy = ''
  let error = ''
  let message = ''
  let selectedLora = null
  let modalEditing = false
  let loraView = 'gallery'
  let releaseScroll = null
  let uploadFile = null
  let addCover = null
  let editCover = null
  let removeEditCover = false
  let coverPickerTarget = ''
  let form = { source: '', provider: 'auto', name: '', trigger_word: '', memo: '', base_model: '', recommended_strength: 1 }
  let edit = { name: '', trigger_word: '', memo: '', base_model: '', recommended_strength: 1 }

  async function refresh() {
    try {
      const [nextLoras, nextStatus] = await Promise.all([api.userLoras(), api.loraStatus()])
      loras = nextLoras
      status = nextStatus
    } catch (e) { error = e.message }
  }

  onMount(() => {
    loraView = localStorage.getItem('media-lora-view') === 'list' ? 'list' : 'gallery'
    refresh()
  })

  function setLoraView(view) {
    loraView = view
    localStorage.setItem('media-lora-view', view)
  }

  function unlockScroll() {
    releaseScroll?.()
    releaseScroll = null
  }

  $: {
    if (selectedLora && !releaseScroll) releaseScroll = lockModalScroll()
    else if (!selectedLora) unlockScroll()
  }

  onDestroy(unlockScroll)

  function releaseCover(selection) {
    if (selection?.preview?.startsWith('blob:')) URL.revokeObjectURL(selection.preview)
  }

  function setCover(target, selection) {
    if (target === 'edit') {
      releaseCover(editCover)
      editCover = selection
      removeEditCover = false
    } else {
      releaseCover(addCover)
      addCover = selection
    }
  }

  function chooseCoverFile(target, event) {
    const file = event.currentTarget.files?.[0]
    event.currentTarget.value = ''
    if (!file) return
    setCover(target, { file, name: file.name, preview: URL.createObjectURL(file) })
  }

  function chooseRecentCover(job) {
    if (!coverPickerTarget || !job?.output_url) return
    setCover(coverPickerTarget, { url: job.output_url, name: `생성 이미지 ${job.id.slice(0, 8)}`, preview: job.output_url })
    coverPickerTarget = ''
  }

  async function saveCover(filename, selection) {
    if (!selection) return null
    let blob = selection.file
    if (!blob && selection.url) {
      const response = await fetch(selection.url)
      if (!response.ok) throw new Error(`대표 이미지를 가져오지 못했습니다. HTTP ${response.status}`)
      blob = await response.blob()
    }
    const body = new FormData()
    body.append('file', blob, selection.name || 'lora-preview.png')
    return api.updateUserLoraPreview(filename, body)
  }

  function clearAddCover() {
    releaseCover(addCover)
    addCover = null
  }

  function loraPreview(lora) {
    return lora?.preview_available ? `/api/lora/${encodeURIComponent(lora.filename)}/preview?v=${encodeURIComponent(lora.preview_updated_at || 0)}` : ''
  }

  function detectedProvider() {
    if (form.provider !== 'auto') return form.provider
    const source = form.source.trim().toLowerCase()
    return /^\d+$/.test(source) || source.includes('civitai.') ? 'civitai' : 'huggingface'
  }

  async function importLora() {
    if (!form.source.trim()) return
    busy = 'import'; error = ''; message = ''
    try {
      const imported = await api.importUserLora({ ...form, source: form.source.trim(), name: form.name.trim(), trigger_word: form.trigger_word.trim(), recommended_strength: Number(form.recommended_strength) })
      await saveCover(imported.filename, addCover)
      clearAddCover()
      form = { ...form, source: '', name: '', trigger_word: '', memo: '', base_model: '', recommended_strength: 1 }
      message = `${imported.name || imported.filename} LoRA를 등록했습니다.`
      await refresh()
      await onChanged()
    } catch (e) { error = e.message } finally { busy = '' }
  }

  async function uploadLora() {
    if (!uploadFile) return
    busy = 'upload'; error = ''; message = ''
    const body = new FormData()
    body.append('file', uploadFile)
    body.append('name', form.name.trim())
    body.append('trigger_word', form.trigger_word.trim())
    body.append('memo', form.memo.trim())
    body.append('base_model', form.base_model.trim())
    body.append('recommended_strength', String(Number(form.recommended_strength)))
    try {
      const imported = await api.uploadUserLora(body)
      await saveCover(imported.filename, addCover)
      clearAddCover()
      uploadFile = null
      form = { ...form, source: '', name: '', trigger_word: '', memo: '', base_model: '', recommended_strength: 1 }
      message = `${imported.name || imported.filename} LoRA를 등록했습니다.`
      await refresh(); await onChanged()
    } catch (e) { error = e.message } finally { busy = '' }
  }

  function beginEdit(lora) {
    releaseCover(editCover)
    editCover = null
    removeEditCover = false
    selectedLora = lora
    modalEditing = true
    edit = { name: lora.name || lora.filename.replace(/\.safetensors$/i, ''), trigger_word: lora.trigger_word || '', memo: lora.memo || '', base_model: lora.base_model || '', recommended_strength: Number(lora.recommended_strength ?? 1) }
  }

  function openDetails(lora) {
    selectedLora = lora
    modalEditing = false
  }

  function closeDetails() {
    releaseCover(editCover)
    editCover = null
    removeEditCover = false
    coverPickerTarget = ''
    selectedLora = null
    modalEditing = false
  }

  async function saveEdit(lora) {
    busy = `edit:${lora.filename}`; error = ''; message = ''
    try {
      const updated = await api.updateUserLora(lora.filename, { name: edit.name.trim(), trigger_word: edit.trigger_word.trim(), memo: edit.memo.trim(), base_model: edit.base_model.trim(), recommended_strength: Number(edit.recommended_strength) })
      if (editCover) await saveCover(lora.filename, editCover)
      else if (removeEditCover && lora.preview_available) {
        const response = await api.deleteUserLoraPreview(lora.filename)
        if (!response.ok) throw new Error((await response.text()) || `HTTP ${response.status}`)
      }
      releaseCover(editCover); editCover = null; removeEditCover = false
      await refresh()
      selectedLora = loras.find((item) => item.filename === lora.filename) || updated
      modalEditing = false; message = 'LoRA 정보를 수정했습니다.'; await onChanged()
    } catch (e) { error = e.message } finally { busy = '' }
  }

  async function deleteLora(lora) {
    if (!confirm(`${lora.name || lora.filename} LoRA를 생성 모델에서 삭제할까요?`)) return
    busy = `delete:${lora.filename}`; error = ''; message = ''
    try {
      const response = await api.deleteUserLora(lora.filename)
      if (!response.ok) throw new Error((await response.text()) || `HTTP ${response.status}`)
      if (selectedLora?.filename === lora.filename) closeDetails()
      message = 'LoRA를 삭제했습니다.'; await refresh(); await onChanged()
    } catch (e) { error = e.message } finally { busy = '' }
  }

  const sizeLabel = (size) => Number(size || 0) >= 1073741824 ? `${(Number(size) / 1073741824).toFixed(2)} GB` : `${(Number(size || 0) / 1048576).toFixed(1)} MB`
  const providerLabel = (provider) => provider === 'civitai' ? 'Civitai' : provider === 'huggingface' ? 'Hugging Face' : provider === 'upload' ? '직접 업로드' : '로컬'
</script>

<section class="lora-studio">
  <div class="section-title"><div><span>05</span><h2>LoRA 관리</h2></div></div>
  <p class="lora-intro">Civitai와 Hugging Face에서 LoRA를 등록하고 생성에 사용할 이름·트리거·기본 강도를 관리합니다. 학습은 추후 Ostris ai-toolkit에서 별도로 진행합니다.</p>
  {#if error}<div class="error"><span>{error}</span><button onclick={() => error = ''}>×</button></div>{/if}
  {#if message}<div class="success">{message}</div>{/if}

  <div class="lora-manager-layout">
    <section class="lora-card lora-add-card">
      <div class="lora-card-heading"><div><h3>LoRA 추가</h3><small>URL로 받거나 가지고 있는 파일을 직접 등록</small></div></div>
      <label>출처
        <input bind:value={form.source} placeholder="Civitai 주소·버전 ID 또는 Hugging Face 주소·owner/repo">
        <small>Hugging Face 저장소에 safetensors가 여러 개면 원하는 파일의 페이지 주소를 입력하세요.</small>
      </label>
      <label class="lora-direct-file">직접 업로드
        <input type="file" accept=".safetensors,application/octet-stream" onchange={(event) => uploadFile = event.currentTarget.files?.[0] || null}>
        <span><i>FILE</i><b title={uploadFile?.name || 'safetensors 파일 선택'}>{uploadFile?.name || 'safetensors 파일 선택'}</b></span>
      </label>
      <div class="fields">
        <label>공급자<select bind:value={form.provider}><option value="auto">자동 감지</option><option value="civitai">Civitai</option><option value="huggingface">Hugging Face</option></select></label>
        <label>표시 이름 · 선택<input bind:value={form.name} maxlength="128" placeholder="원본 파일명 사용"></label>
      </div>
      <div class="fields">
        <label>트리거 · 선택<input bind:value={form.trigger_word} maxlength="512" placeholder="Civitai는 메타데이터 자동 사용"></label>
        <label>기본 강도<input type="number" min="-2" max="2" step="any" bind:value={form.recommended_strength}></label>
      </div>
      <label>기반 모델 · 선택<input bind:value={form.base_model} maxlength="128" list="lora-base-models" placeholder="예: Krea 2 Turbo, LTX-2.5"><small>비워두면 Civitai는 메타데이터를 사용하고, 직접 업로드는 미지정으로 등록됩니다.</small></label>
      <datalist id="lora-base-models"><option value="Krea 2 Turbo"></option><option value="LTX-2.5"></option></datalist>
      <label>메모 · 선택<textarea rows="2" maxlength="2000" bind:value={form.memo} placeholder="용도, 권장 프롬프트, 조합 주의사항 등을 기록하세요."></textarea></label>
      <div class="lora-cover-field">
        <span class="lora-field-label">대표 이미지 · 선택</span>
        <div class="lora-cover-control">
          <div class="lora-cover-preview" class:empty={!addCover}>
            {#if addCover}<img src={addCover.preview} alt="등록할 LoRA 대표 이미지">{:else}<span>IMG</span>{/if}
          </div>
          <div><strong title={addCover?.name || '대표 이미지 없음'}>{addCover?.name || '대표 이미지 없음'}</strong><small>직접 올리거나 생성 이미지에서 복사해 보관합니다.</small><div class="lora-cover-actions"><label class="quiet lora-cover-file">파일 선택<input type="file" accept="image/*" onchange={(event) => chooseCoverFile('add', event)}></label><button type="button" class="quiet" onclick={() => coverPickerTarget = 'add'}>생성 이미지</button>{#if addCover}<button type="button" class="quiet danger" onclick={clearAddCover}>비우기</button>{/if}</div></div>
        </div>
      </div>
      <div class="lora-credential-status"><span class:ready={status.civitai_token_configured}><i></i>Civitai {status.civitai_token_configured ? '설정됨' : '미설정'}</span><span class:ready={status.hf_token_configured}><i></i>Hugging Face {status.hf_token_configured ? '설정됨' : '미설정'}</span><button type="button" class="quiet" onclick={onOpenSettings}>설정 열기</button></div>
      <div class="lora-add-actions"><button type="button" class="quiet" disabled={Boolean(busy) || !uploadFile} onclick={uploadLora}>{busy === 'upload' ? '업로드 중…' : '파일 추가'}</button><button type="button" class="primary" disabled={Boolean(busy) || !form.source.trim()} onclick={importLora}>{busy === 'import' ? `${detectedProvider() === 'civitai' ? 'Civitai' : 'Hugging Face'} 다운로드 중…` : 'URL에서 추가'}</button></div>
      <small class="lora-storage-note">다운로드 키는 설정 탭에서 한 번만 등록합니다. safetensors · 파일당 최대 2 GiB.</small>
    </section>

    <section class="lora-card lora-list-card">
      <div class="lora-card-heading"><div><h3>등록된 LoRA</h3><small>{loras.length}개 · Spark Media 공유 저장소</small></div><div class="lora-list-tools"><div class="lora-view-toggle"><button type="button" class:active={loraView === 'gallery'} onclick={() => setLoraView('gallery')}>갤러리</button><button type="button" class:active={loraView === 'list'} onclick={() => setLoraView('list')}>리스트</button></div><button type="button" class="quiet" onclick={refresh}>새로고침</button></div></div>
      <div class="registered-loras" class:list-view={loraView === 'list'}>
        {#each loras as lora}
          <article>
            <button type="button" class="lora-item-open" onclick={() => openDetails(lora)}>
              <span class="lora-file-visual" class:has-image={lora.preview_available}>{#if lora.preview_available}<img src={loraPreview(lora)} alt={`${lora.name || lora.filename} 대표 이미지`} loading="lazy">{:else}<i>LORA</i><b>{providerLabel(lora.provider)}</b>{/if}</span>
              <span class="lora-item-main"><strong title={lora.name || lora.filename}>{lora.name || lora.filename}</strong><small title={lora.filename}>{lora.filename}</small></span>
              <span class="lora-item-summary"><i>강도 {Number(lora.recommended_strength ?? 1).toFixed(2)}</i><i>{sizeLabel(lora.size)}</i>{#if lora.memo}<i class="has-memo">메모</i>{/if}</span>
            </button>
            <div class="lora-item-actions"><button type="button" class="quiet" onclick={() => openDetails(lora)}>정보</button><button type="button" class="quiet" onclick={() => beginEdit(lora)}>수정</button><button type="button" class="job-delete" disabled={busy === `delete:${lora.filename}`} onclick={() => deleteLora(lora)}>삭제</button></div>
          </article>
        {:else}<div class="empty">등록된 사용자 LoRA가 없습니다.</div>{/each}
      </div>
    </section>
  </div>
</section>

<svelte:window onkeydown={(event) => { if (selectedLora && event.key === 'Escape' && !busy) closeDetails() }} />

{#if selectedLora}
  <div class="lora-detail-backdrop" role="presentation" onclick={(event) => { if (event.target === event.currentTarget && !busy) closeDetails() }}>
    <div class="lora-detail-modal" role="dialog" aria-modal="true" aria-label="LoRA 상세 정보">
      <header><div><strong>{modalEditing ? 'LoRA 정보 수정' : selectedLora.name || selectedLora.filename}</strong><small title={selectedLora.filename}>{selectedLora.filename}</small></div><button type="button" aria-label="닫기" disabled={Boolean(busy)} onclick={closeDetails}>×</button></header>
      <div class="lora-detail-content">
        {#if modalEditing}
          <div class="lora-modal-edit-fields">
            <div class="lora-modal-cover wide">
              <div class="lora-cover-preview large" class:empty={!editCover && (!selectedLora.preview_available || removeEditCover)}>{#if editCover}<img src={editCover.preview} alt="새 대표 이미지">{:else if selectedLora.preview_available && !removeEditCover}<img src={loraPreview(selectedLora)} alt="현재 대표 이미지">{:else}<span>IMG</span>{/if}</div>
              <div><strong>대표 이미지</strong><small>{editCover ? editCover.name : selectedLora.preview_available && !removeEditCover ? '현재 등록된 이미지' : '대표 이미지 없음'}</small><div class="lora-cover-actions"><label class="quiet lora-cover-file">파일 선택<input type="file" accept="image/*" onchange={(event) => chooseCoverFile('edit', event)}></label><button type="button" class="quiet" onclick={() => coverPickerTarget = 'edit'}>생성 이미지</button>{#if editCover || (selectedLora.preview_available && !removeEditCover)}<button type="button" class="quiet danger" onclick={() => { releaseCover(editCover); editCover = null; removeEditCover = true }}>비우기</button>{/if}</div></div>
            </div>
            <label>표시 이름<input bind:value={edit.name} maxlength="128"></label>
            <label>기본 강도<input type="number" min="-2" max="2" step="any" bind:value={edit.recommended_strength}></label>
            <label class="wide">기반 모델<input bind:value={edit.base_model} maxlength="128" list="lora-base-models" placeholder="예: Krea 2 Turbo, LTX-2.5"></label>
            <label class="wide">트리거<input bind:value={edit.trigger_word} maxlength="512" placeholder="없음"></label>
            <label class="wide">메모<textarea rows="9" maxlength="2000" bind:value={edit.memo} placeholder="용도, 권장 프롬프트, 조합 주의사항 등을 기록하세요."></textarea><small>{edit.memo.length}/2000</small></label>
          </div>
        {:else}
          {#if selectedLora.preview_available}<div class="lora-detail-image"><img src={loraPreview(selectedLora)} alt={`${selectedLora.name || selectedLora.filename} 대표 이미지`}></div>{/if}
          <dl class="lora-detail-facts">
            <div><dt>공급자</dt><dd>{providerLabel(selectedLora.provider)}</dd></div><div><dt>기본 강도</dt><dd>{Number(selectedLora.recommended_strength ?? 1).toFixed(2)}</dd></div>
            <div><dt>파일 크기</dt><dd>{sizeLabel(selectedLora.size)}</dd></div><div><dt>기반 모델</dt><dd>{selectedLora.base_model || '—'}</dd></div>
            <div class="wide"><dt>파일명</dt><dd>{selectedLora.filename}</dd></div>
          </dl>
          <section class="lora-detail-section"><strong>트리거</strong><p>{selectedLora.trigger_word || '없음'}</p></section>
          <section class="lora-detail-section memo"><strong>메모</strong><p>{selectedLora.memo || '작성된 메모가 없습니다.'}</p></section>
        {/if}
      </div>
      <footer><div>{#if selectedLora.source}<a href={selectedLora.source} target="_blank" rel="noreferrer">출처 ↗</a>{/if}</div><div>{#if modalEditing}<button type="button" class="quiet" disabled={Boolean(busy)} onclick={() => modalEditing = false}>취소</button><button type="button" class="primary" disabled={busy === `edit:${selectedLora.filename}`} onclick={() => saveEdit(selectedLora)}>{busy === `edit:${selectedLora.filename}` ? '저장 중…' : '저장'}</button>{:else}<button type="button" class="quiet" onclick={closeDetails}>닫기</button><button type="button" class="primary" onclick={() => beginEdit(selectedLora)}>수정</button>{/if}</div></footer>
    </div>
  </div>
{/if}

<RecentImagePicker open={Boolean(coverPickerTarget)} zIndex={100} title="LoRA 대표 이미지 선택" jobs={imageJobs} selectedRef="" onSelect={chooseRecentCover} onClose={() => coverPickerTarget = ''} />
