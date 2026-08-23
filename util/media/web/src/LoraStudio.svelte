<script>
  import { onMount } from 'svelte'
  import { api } from './api.js'

  let datasets = []
  let jobs = []
  let loras = []
  let selected = ''
  let newDataset = ''
  let uploads = []
  let defaultCaption = ''
  let busy = ''
  let error = ''
  let message = ''
  let form = {
    name: '', trigger_word: '', steps: 1500, rank: 32, alpha: 32,
    learning_rate: 0.0001, resolutions: [512, 768, 1024],
    caption_dropout: 0.05, save_every: 250, sample_prompt: ''
  }

  const current = () => datasets.find((dataset) => dataset.name === selected)
  const activeJob = () => jobs.find((job) => job.status === 'queued' || job.status === 'running')
  const completeCaptions = () => current() && current().images > 1 && current().images === current().captioned

  async function refresh() {
    try {
      const [nextDatasets, nextJobs, nextLoras] = await Promise.all([api.loraDatasets(), api.loraJobs(), api.userLoras()])
      datasets = nextDatasets
      jobs = nextJobs
      loras = nextLoras
      if (!datasets.some((dataset) => dataset.name === selected)) selected = datasets[0]?.name || ''
    } catch (e) { error = e.message }
  }

  onMount(() => {
    refresh()
    const timer = setInterval(refresh, 2000)
    return () => clearInterval(timer)
  })

  async function createDataset() {
    if (!newDataset.trim()) return
    busy = 'dataset'; error = ''; message = ''
    try {
      const dataset = await api.createLoraDataset(newDataset.trim())
      newDataset = ''; selected = dataset.name; message = '데이터셋을 만들었습니다.'
      await refresh()
    } catch (e) { error = e.message } finally { busy = '' }
  }

  async function deleteDataset() {
    if (!selected || !confirm(`${selected} 데이터셋과 이미지·캡션을 모두 삭제할까요?`)) return
    busy = 'dataset-delete'; error = ''; message = ''
    try {
      const response = await api.deleteLoraDataset(selected)
      if (!response.ok) throw new Error((await response.text()) || `HTTP ${response.status}`)
      selected = ''; message = '데이터셋을 삭제했습니다.'; await refresh()
    } catch (e) { error = e.message } finally { busy = '' }
  }

  async function uploadImages() {
    if (!selected || !uploads.length) return
    busy = 'upload'; error = ''; message = ''
    const body = new FormData()
    for (const image of uploads) body.append('images', image)
    body.append('default_caption', defaultCaption)
    try {
      await api.uploadLoraImages(selected, body)
      uploads = []; message = '학습 이미지를 추가했습니다.'
      await refresh()
    } catch (e) { error = e.message } finally { busy = '' }
  }

  async function saveCaption(item) {
    busy = `caption:${item.name}`; error = ''; message = ''
    try {
      await api.saveLoraCaption(selected, item.name, item.caption)
      message = `${item.name} 캡션을 저장했습니다.`; await refresh()
    } catch (e) { error = e.message } finally { busy = '' }
  }

  async function deleteImage(item) {
    if (!confirm(`${item.name}을 데이터셋에서 삭제할까요?`)) return
    busy = `delete:${item.name}`; error = ''; message = ''
    try {
      const response = await api.deleteLoraImage(selected, item.name)
      if (!response.ok) throw new Error((await response.text()) || `HTTP ${response.status}`)
      message = '이미지를 삭제했습니다.'; await refresh()
    } catch (e) { error = e.message } finally { busy = '' }
  }

  async function prependTrigger() {
    const trigger = form.trigger_word.trim()
    if (!trigger || !current()?.items.length) return
    busy = 'trigger'; error = ''; message = ''
    try {
      for (const item of current().items) {
        const caption = item.caption.trim()
        if (caption.toLowerCase().includes(trigger.toLowerCase())) continue
        await api.saveLoraCaption(selected, item.name, caption ? `${trigger}, ${caption}` : trigger)
      }
      message = '모든 캡션에 트리거 단어를 반영했습니다.'; await refresh()
    } catch (e) { error = e.message } finally { busy = '' }
  }

  function toggleResolution(value) {
    form.resolutions = form.resolutions.includes(value)
      ? form.resolutions.filter((item) => item !== value)
      : [...form.resolutions, value].sort((a, b) => a - b)
  }

  async function startTraining() {
    if (!selected || !form.name.trim()) return
    busy = 'train'; error = ''; message = ''
    try {
      await api.startLoraTraining({ ...form, name: form.name.trim(), dataset: selected })
      message = 'Krea 2 Turbo LoRA 학습을 시작했습니다.'; await refresh()
    } catch (e) { error = e.message } finally { busy = '' }
  }

  async function cancelTraining(job) {
    if (!confirm(`${job.name} 학습을 중지할까요?`)) return
    busy = `cancel:${job.id}`; error = ''; message = ''
    try {
      await api.cancelLoraTraining(job.id); message = '학습 중지를 요청했습니다.'; await refresh()
    } catch (e) { error = e.message } finally { busy = '' }
  }

  async function deleteLora(lora) {
    if (!confirm(`${lora.name || lora.filename} LoRA를 생성 모델에서 삭제할까요?`)) return
    busy = `lora-delete:${lora.filename}`; error = ''; message = ''
    try {
      const response = await api.deleteUserLora(lora.filename)
      if (!response.ok) throw new Error((await response.text()) || `HTTP ${response.status}`)
      message = '사용자 LoRA를 삭제했습니다.'; await refresh()
    } catch (e) { error = e.message } finally { busy = '' }
  }

  function progress(job) {
    return Math.min(100, Math.max(0, Number(job.step || 0) * 100 / Math.max(1, Number(job.total_steps || 1))))
  }
</script>

<section class="lora-studio">
  <div class="section-title"><div><span>LORA</span><h2>LoRA 제작소</h2></div></div>
  <p class="lora-intro">현재는 Ostris ai-toolkit으로 Krea 2 Turbo 이미지 LoRA를 만듭니다. 완성된 LoRA는 생성 모델에 자동 등록됩니다.</p>
  {#if error}<div class="error"><span>{error}</span><button onclick={() => error = ''}>×</button></div>{/if}
  {#if message}<div class="success">{message}</div>{/if}

  <div class="lora-layout">
    <section class="lora-card dataset-panel">
      <div class="lora-card-heading"><div><h3>1. 데이터셋</h3><small>모든 이미지에 설명 캡션이 필요합니다.</small></div></div>
      <div class="lora-inline">
        <input bind:value={newDataset} maxlength="64" placeholder="새 데이터셋 이름">
        <button type="button" class="quiet" disabled={busy === 'dataset' || !newDataset.trim()} onclick={createDataset}>만들기</button>
      </div>
      <label>데이터셋<select bind:value={selected}><option value="">선택</option>{#each datasets as dataset}<option value={dataset.name}>{dataset.name} · {dataset.captioned}/{dataset.images} 캡션</option>{/each}</select></label>
      {#if selected}<button type="button" class="job-delete dataset-delete" disabled={busy === 'dataset-delete'} onclick={deleteDataset}>현재 데이터셋 삭제</button>{/if}
      {#if selected}
        <div class="lora-upload">
          <label>공통 기본 캡션<input bind:value={defaultCaption} placeholder="예: portrait photo, studio lighting"></label>
          <label class="module-file">학습 이미지<input type="file" accept="image/*" multiple onchange={(e) => uploads = [...e.currentTarget.files]}><span class="module-file-display"><i>IMG</i><b>{uploads.length ? `${uploads.length}장 선택됨` : '이미지 여러 장 선택'}</b></span></label>
          <button type="button" class="primary" disabled={busy === 'upload' || !uploads.length} onclick={uploadImages}>{busy === 'upload' ? '업로드 중…' : '이미지 추가'}</button>
        </div>
      {/if}
    </section>

    <section class="lora-card train-panel">
      <div class="lora-card-heading"><div><h3>2. 학습 설정</h3><small>Krea 2 Turbo · 공식 training adapter</small></div></div>
      <div class="fields">
        <label>LoRA 이름<input bind:value={form.name} maxlength="64" placeholder="my-character-v1"></label>
        <label>트리거 단어<input bind:value={form.trigger_word} maxlength="128" placeholder="예: ohwx_person"></label>
      </div>
      <button type="button" class="quiet lora-trigger" disabled={!form.trigger_word.trim() || !current()?.items.length || busy === 'trigger'} onclick={prependTrigger}>트리거를 모든 캡션 앞에 추가</button>
      <div class="fields three">
        <label>스텝<input type="number" min="100" max="10000" step="50" bind:value={form.steps}></label>
        <label>Rank<select bind:value={form.rank}><option value={8}>8</option><option value={16}>16</option><option value={32}>32</option><option value={64}>64</option><option value={128}>128</option></select></label>
        <label>Alpha<input type="number" min="1" max="128" bind:value={form.alpha}></label>
      </div>
      <div class="fields">
        <label>학습률<input type="number" min="0.000001" max="0.01" step="0.00001" bind:value={form.learning_rate}></label>
        <label>캡션 드롭아웃<input type="number" min="0" max="0.5" step="0.01" bind:value={form.caption_dropout}></label>
      </div>
      <fieldset class="lora-resolutions"><legend>학습 해상도</legend>{#each [512, 768, 1024, 1280] as size}<label><input type="checkbox" checked={form.resolutions.includes(size)} onchange={() => toggleResolution(size)}>{size}</label>{/each}</fieldset>
      <label>검증 프롬프트<textarea rows="3" bind:value={form.sample_prompt} placeholder="학습 중 같은 조건으로 확인할 장면"></textarea></label>
      <button type="button" class="primary" disabled={Boolean(activeJob()) || busy === 'train' || !completeCaptions() || !form.name.trim() || !form.resolutions.length} onclick={startTraining}>{activeJob() ? '다른 학습이 진행 중입니다' : busy === 'train' ? '시작 중…' : 'LoRA 학습 시작'}</button>
      {#if current() && !completeCaptions()}<small class="module-caution">이미지 2장 이상과 모든 이미지의 캡션을 준비하세요.</small>{/if}
    </section>
  </div>

  {#if current()?.items.length}
    <section class="lora-card caption-panel">
      <div class="lora-card-heading"><div><h3>3. 이미지와 캡션</h3><small>{current().captioned}/{current().images}개 완료</small></div></div>
      <div class="caption-grid">
        {#each current().items as item}
          <article>
            <img src={`/api/lora${item.url}`} alt={item.name}>
            <textarea rows="4" bind:value={item.caption} placeholder="이미지의 인물·스타일·구도 설명"></textarea>
            <div><button type="button" class="quiet" disabled={busy === `caption:${item.name}`} onclick={() => saveCaption(item)}>캡션 저장</button><button type="button" class="job-delete" disabled={busy === `delete:${item.name}`} onclick={() => deleteImage(item)}>삭제</button></div>
          </article>
        {/each}
      </div>
    </section>
  {/if}

  <div class="lora-layout lower">
    <section class="lora-card">
      <div class="lora-card-heading"><div><h3>학습 작업</h3><small>학습 컨테이너를 종료하지 않고 여기서 중지할 수 있습니다.</small></div></div>
      <div class="training-jobs">
        {#each jobs as job}
          <article>
            <div><strong>{job.name}</strong><small>{job.dataset} · {job.status}</small></div>
            <div class="training-progress"><span style={`width:${progress(job)}%`}></span></div>
            <small>{job.step || 0}/{job.total_steps || 0}{#if job.loss !== undefined} · loss {Number(job.loss).toFixed(4)}{/if}</small>
            {#if job.error}<em>{job.error}</em>{/if}
            {#if job.status === 'queued' || job.status === 'running'}<button type="button" class="job-delete" disabled={busy === `cancel:${job.id}`} onclick={() => cancelTraining(job)}>중지</button>{/if}
          </article>
        {:else}<div class="empty">아직 학습 작업이 없습니다.</div>{/each}
      </div>
    </section>
    <section class="lora-card">
      <div class="lora-card-heading"><div><h3>등록된 사용자 LoRA</h3><small>생성 스튜디오용 공유 저장소</small></div></div>
      <div class="registered-loras">
        {#each loras as lora}<article><strong>{lora.name || lora.filename}</strong><small>{lora.trigger_word || '트리거 없음'} · rank {lora.rank || '—'}{#if lora.recommended_strength !== undefined} · 권장 강도 {Number(lora.recommended_strength).toFixed(2)}{/if} · {(Number(lora.size || 0) / 1048576).toFixed(1)} MB</small><button type="button" class="job-delete" disabled={busy === `lora-delete:${lora.filename}`} onclick={() => deleteLora(lora)}>삭제</button></article>{:else}<div class="empty">학습이 끝난 LoRA가 여기에 등록됩니다.</div>{/each}
      </div>
    </section>
  </div>
</section>
