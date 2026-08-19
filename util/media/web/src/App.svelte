<script>
  import { onMount } from 'svelte'
  import { api } from './api.js'

  let tab = 'image'
  let config = null
  let settings = null
  let savedMessage = ''
  let jobs = []
  let engineStates = { image: 'offline', speech: 'offline', recognition: 'offline', video: 'offline', prompt: 'offline' }
  let busy = false
  let error = ''
  let refs = []
  let imageForm = { prompt: '', width: 1024, height: 1024, seed: -1 }
  let speechForm = { text: '', instructions: '', language: 'Korean', speaker: 'Sohee', seed: -1 }
  let recognitionForm = { language: 'Auto', context: '' }
  let recognitionFile = null
  let videoForm = { prompt: '', width: 768, height: 512, num_frames: 121, fps: 24, seed: -1, image_strength: 1 }
  let videoImage = null
  let videoEnhanceEnabled = true
  let videoEnhancedPrompt = ''
  let videoEnhancedSource = ''
  let videoEnhancedImageKey = ''
  let enhancingPrompt = false
  let deletingJob = ''

  const engineMeta = {
    image: ['image', 'Klein'],
    video: ['video', 'LTX'],
    speech: ['speech', 'TTS'],
    recognition: ['recognition', 'ASR']
  }

  const activeJobs = () => jobs.filter((j) => j.status === 'queued' || j.status === 'running')
  async function refresh() {
    try {
      const [nextJobs, nextEngines] = await Promise.all([api.jobs(), api.engines()])
      jobs = nextJobs
      engineStates = Object.fromEntries(nextEngines.map((item) => [item.kind, item.status]))
    } catch (e) { error = e.message }
  }

  onMount(() => {
    api.config().then((value) => {
      config = value
      settings = structuredClone(value)
      imageForm.width = value.image.default_width
      imageForm.height = value.image.default_height
      speechForm.language = value.speech.default_language
      speechForm.speaker = value.speech.default_speaker
      recognitionForm.language = value.recognition.default_language
      videoForm.width = value.video.default_width
      videoForm.height = value.video.default_height
      videoForm.num_frames = value.video.default_frames
      videoForm.fps = value.video.default_fps
      videoEnhanceEnabled = value.prompt_enhancement.default_enabled
    }).catch((e) => error = e.message)
    refresh()
    const timer = setInterval(refresh, 1500)
    return () => clearInterval(timer)
  })

  function addRefs(files) {
    const incoming = [...files].filter((f) => f.type.startsWith('image/'))
    refs = [...refs, ...incoming].slice(0, config?.image.max_reference_images || 4)
  }

  async function generateImage() {
    busy = true; error = ''
    try {
      const form = new FormData()
      Object.entries(imageForm).forEach(([key, value]) => form.append(key, value))
      refs.forEach((file) => form.append('references', file))
      await api.image(form); imageForm.prompt = ''; refs = []; await refresh()
    } catch (e) { error = e.message } finally { busy = false }
  }

  async function generateSpeech() {
    busy = true; error = ''
    try {
      const form = new FormData()
      form.append('text', speechForm.text)
      form.append('instructions', speechForm.instructions)
      form.append('language', speechForm.language); form.append('speaker', speechForm.speaker); form.append('seed', speechForm.seed)
      await api.speech(form); speechForm.text = ''; await refresh()
    } catch (e) { error = e.message } finally { busy = false }
  }

  async function recognizeSpeech() {
    if (!recognitionFile) return
    busy = true; error = ''
    try {
      const form = new FormData()
      form.append('audio', recognitionFile)
      form.append('language', recognitionForm.language)
      form.append('context', recognitionForm.context)
      await api.recognition(form); recognitionFile = null; await refresh()
    } catch (e) { error = e.message } finally { busy = false }
  }

  async function generateVideo() {
    if (videoEnhancementActive() && !videoEnhancementCurrent()) {
      await enhanceVideoPrompt()
      return
    }
    busy = true; error = ''
    try {
      const form = new FormData()
      Object.entries(videoForm).forEach(([key, value]) => form.append(key, key === 'prompt' && videoEnhancementActive() ? videoEnhancedPrompt : value))
      form.append('original_prompt', videoForm.prompt)
      if (videoImage) form.append('image', videoImage)
      await api.video(form)
      videoForm.prompt = ''; videoImage = null; resetVideoEnhancement(); await refresh()
    } catch (e) { error = e.message } finally { busy = false }
  }

  function videoImageKey() {
    return videoImage ? `${videoImage.name}:${videoImage.size}:${videoImage.lastModified}` : ''
  }

  function videoEnhancementCurrent() {
    return videoEnhancedPrompt.trim() !== '' && videoEnhancedSource === videoForm.prompt.trim() && videoEnhancedImageKey === videoImageKey()
  }

  function videoEnhancementActive() {
    return videoEnhanceEnabled && !(videoImage && !config?.prompt_enhancement.vision_enabled)
  }

  function resetVideoEnhancement() {
    videoEnhancedPrompt = ''
    videoEnhancedSource = ''
    videoEnhancedImageKey = ''
  }

  function selectVideoImage(file) {
    videoImage = file || null
    resetVideoEnhancement()
  }

  async function enhanceVideoPrompt() {
    const original = videoForm.prompt.trim()
    if (!original) return
    enhancingPrompt = true; error = ''
    try {
      const form = new FormData()
      form.append('prompt', original)
      form.append('mode', videoImage ? 'i2v' : 't2v')
      if (videoImage) form.append('image', videoImage)
      const result = await api.enhancePrompt(form)
      videoEnhancedPrompt = result.enhanced_prompt
      videoEnhancedSource = original
      videoEnhancedImageKey = videoImageKey()
    } catch (e) { error = e.message }
    finally { enhancingPrompt = false }
  }

  async function deleteJob(job) {
    if (!confirm(`이 ${job.status === 'failed' ? '실패한 작업' : '작업'}과 저장 파일을 삭제할까요?`)) return
    deletingJob = job.id; error = ''
    try { await api.deleteJob(job.id); await refresh() }
    catch (e) { error = e.message }
    finally { deletingJob = '' }
  }

  async function clearFinishedJobs() {
    const count = jobs.filter((job) => job.status !== 'queued' && job.status !== 'running').length
    if (!count || !confirm(`완료·실패 작업 ${count}개와 저장 파일을 모두 삭제할까요?`)) return
    deletingJob = 'all'; error = ''
    try { await api.deleteFinishedJobs(); await refresh() }
    catch (e) { error = e.message }
    finally { deletingJob = '' }
  }

  function openSettings() {
    settings = structuredClone(config)
    savedMessage = ''
    error = ''
    tab = 'settings'
  }

  async function saveSettings() {
    busy = true; error = ''; savedMessage = ''
    try {
      const result = await api.saveConfig(settings)
      config = result.config
      settings = structuredClone(result.config)
      imageForm.width = config.image.default_width
      imageForm.height = config.image.default_height
      speechForm.language = config.speech.default_language
      speechForm.speaker = config.speech.default_speaker
      recognitionForm.language = config.recognition.default_language
      videoForm.width = config.video.default_width
      videoForm.height = config.video.default_height
      videoForm.num_frames = config.video.default_frames
      videoForm.fps = config.video.default_fps
      videoEnhanceEnabled = config.prompt_enhancement.default_enabled
      savedMessage = result.restart_required
        ? '저장했습니다. Listen 주소 또는 데이터 폴더 변경은 Media 재시작 후 적용됩니다.'
        : '저장했습니다. API 연결과 생성 기본값이 즉시 적용됐습니다.'
      await refresh()
    } catch (e) { error = e.message } finally { busy = false }
  }

</script>

<svelte:head><meta name="theme-color" content="#101318"></svelte:head>

<header>
  <div><span class="mark">✦</span><h1>생성 스튜디오</h1><span class="phase">IMAGE · VIDEO · VOICE · TRANSCRIBE</span></div>
  <div class="engine-strip">
    {#if engineMeta[tab]}
      <span class:running={engineStates[engineMeta[tab][0]] === 'online'}><i></i>{engineMeta[tab][1]} API · {engineStates[engineMeta[tab][0]] || 'offline'}</span>
    {/if}
    {#if tab === 'video'}
      <span class:running={engineStates.prompt === 'online'}><i></i>Enhancer API · {engineStates.prompt || 'offline'}</span>
    {/if}
  </div>
</header>

<main>
  <nav>
    <button class:active={tab === 'image'} onclick={() => tab = 'image'}>이미지</button>
    <button class:active={tab === 'video'} onclick={() => tab = 'video'}>영상</button>
    <button class:active={tab === 'speech'} onclick={() => tab = 'speech'}>음성</button>
    <button class:active={tab === 'recognition'} onclick={() => tab = 'recognition'}>인식</button>
    <button class:active={tab === 'history'} onclick={() => tab = 'history'}>기록 <b>{jobs.length}</b></button>
    <button class:active={tab === 'settings'} onclick={openSettings}>설정</button>
  </nav>

  {#if error}<div class="error"><span>{error}</span><button onclick={() => error = ''}>×</button></div>{/if}

  {#if tab === 'image'}
    <section class="workspace">
      <form onsubmit={(e) => { e.preventDefault(); generateImage() }}>
        <div class="section-title"><div><span>01</span><h2>이미지 생성과 편집</h2></div></div>
        <label>프롬프트<textarea bind:value={imageForm.prompt} rows="7" placeholder="만들고 싶은 장면이나 참조 이미지의 변경 내용을 입력하세요." required></textarea></label>
        <div class="drop" role="button" tabindex="0" ondragover={(e) => e.preventDefault()} ondrop={(e) => { e.preventDefault(); addRefs(e.dataTransfer.files) }}>
          <input type="file" accept="image/*" multiple onchange={(e) => addRefs(e.currentTarget.files)}>
          <strong>{refs.length ? `참조 이미지 ${refs.length}개` : '참조 이미지 놓기'}</strong>
          <small>선택 사항 · 최대 {config?.image.max_reference_images || 4}개 · 클릭하거나 드래그</small>
          {#if refs.length}<div class="chips">{#each refs as file, i}<button type="button" onclick={() => refs = refs.filter((_, n) => n !== i)}>{file.name} ×</button>{/each}</div>{/if}
        </div>
        <div class="fields three">
          <label>너비<input type="number" min="256" max="2048" step="16" bind:value={imageForm.width}></label>
          <label>높이<input type="number" min="256" max="2048" step="16" bind:value={imageForm.height}></label>
          <label>시드<input type="number" bind:value={imageForm.seed}></label>
        </div>
        <button class="primary" disabled={busy || activeJobs().some((j) => j.kind === 'image')}>{busy ? '요청 중…' : '이미지 만들기'}</button>
      </form>
      <aside><h3>최근 이미지</h3><div class="gallery">
        {#each jobs.filter((j) => j.kind === 'image').slice(0, 8) as job}
          <article class:pending={job.status !== 'completed'}>{#if job.output_url}<img src={job.output_url} alt={job.prompt}>{:else}<div class="placeholder"><span>{job.status}</span></div>{/if}<p>{job.prompt}</p>{#if job.error}<em>{job.error}</em>{/if}{#if job.status === 'completed' || job.status === 'failed'}<button class="job-delete" disabled={deletingJob === job.id} onclick={() => deleteJob(job)}>삭제</button>{/if}</article>
        {:else}<div class="empty">첫 이미지가 여기에 나타납니다.</div>{/each}
      </div></aside>
    </section>
  {:else if tab === 'video'}
    <section class="workspace">
      <form onsubmit={(e) => { e.preventDefault(); generateVideo() }}>
        <div class="section-title"><div><span>02</span><h2>LTX 2.5 영상 생성</h2></div></div>
        <label>원본 프롬프트<textarea bind:value={videoForm.prompt} rows="5" placeholder="장면과 움직임을 자연스럽게 입력하세요." required></textarea></label>
        <label class="file-field">시작 이미지 <small>선택 사항 · image-to-video</small><input type="file" accept="image/png,image/jpeg,image/webp" onchange={(e) => selectVideoImage(e.currentTarget.files?.[0])}><span>{videoImage?.name || '시작 이미지 선택'}</span></label>
        <div class="enhancer-control">
          <div>
            <strong>프롬프트 향상</strong>
            <small>{videoImage && !config?.prompt_enhancement.vision_enabled ? '현재 E2B 번들은 이미지를 볼 수 없어 I2V에서는 원문을 그대로 사용합니다.' : 'LTX 캡션 형식의 영어 프롬프트로 확장합니다.'}</small>
          </div>
          <div class="segmented compact">
            <button type="button" class:active={videoEnhanceEnabled} onclick={() => videoEnhanceEnabled = true}>자동</button>
            <button type="button" class:active={!videoEnhanceEnabled} onclick={() => videoEnhanceEnabled = false}>꺼짐</button>
          </div>
        </div>
        {#if videoEnhancementActive()}
          <div class="enhanced-prompt">
            <div><span>향상된 프롬프트</span><button type="button" class="quiet" disabled={enhancingPrompt || !videoForm.prompt.trim()} onclick={enhanceVideoPrompt}>{enhancingPrompt ? '향상 중…' : videoEnhancementCurrent() ? '다시 향상' : '미리 향상'}</button></div>
            {#if videoEnhancedPrompt}
              <textarea bind:value={videoEnhancedPrompt} rows="8" aria-label="향상된 프롬프트"></textarea>
              <small>{videoImage ? '시작 이미지를 분석해 확장했습니다.' : '텍스트 기반 T2V 확장입니다.'} 생성 전에 직접 수정할 수 있습니다.</small>
            {:else}
              <p>영상 만들기를 누르면 먼저 프롬프트를 향상하여 보여줍니다. 내용을 확인하거나 수정한 뒤 다시 누르면 생성합니다.</p>
            {/if}
          </div>
        {/if}
        <div class="fields three">
          <label>너비<input type="number" min="256" max="1920" step="64" bind:value={videoForm.width}></label>
          <label>높이<input type="number" min="256" max="1920" step="64" bind:value={videoForm.height}></label>
          <label>프레임 <small>8k+1</small><input type="number" min="9" max="481" step="8" bind:value={videoForm.num_frames}></label>
          <label>FPS<input type="number" min="1" max="60" step="1" bind:value={videoForm.fps}></label>
          <label>시드 <small>-1은 무작위</small><input type="number" min="-1" bind:value={videoForm.seed}></label>
          <label>이미지 강도<input type="number" min="0" max="1" step="0.05" bind:value={videoForm.image_strength}></label>
        </div>
        <button class="primary" disabled={busy || enhancingPrompt || activeJobs().some((j) => j.kind === 'video')}>{enhancingPrompt ? '프롬프트 향상 중…' : busy ? '요청 중…' : videoEnhancementActive() && !videoEnhancementCurrent() ? '프롬프트 향상 후 확인' : '영상 만들기'}</button>
      </form>
      <aside><h3>최근 영상</h3><div class="video-list">
        {#each jobs.filter((j) => j.kind === 'video').slice(0, 8) as job}
          <article class:pending={job.status !== 'completed'}>{#if job.output_url}<!-- svelte-ignore a11y_media_has_caption --><video controls preload="metadata" src={job.output_url}></video>{:else}<div class="video-placeholder"><span>{job.status}</span></div>{/if}<p>{job.prompt}</p><small>{job.params?.width}×{job.params?.height} · {job.params?.num_frames} frames · {job.params?.fps} fps{#if job.params?.seed >= 0} · seed {job.params.seed}{/if}</small>{#if job.error}<em>{job.error}</em>{/if}{#if job.status === 'completed' || job.status === 'failed'}<button class="job-delete" disabled={deletingJob === job.id} onclick={() => deleteJob(job)}>삭제</button>{/if}</article>
        {:else}<div class="empty">첫 영상이 여기에 나타납니다.</div>{/each}
      </div></aside>
    </section>
  {:else if tab === 'speech'}
    <section class="workspace">
      <form onsubmit={(e) => { e.preventDefault(); generateSpeech() }}>
        <div class="section-title"><div><span>03</span><h2>CustomVoice 음성 생성</h2></div></div>
        <label>읽을 문장<textarea bind:value={speechForm.text} rows="7" placeholder="음성으로 변환할 문장을 입력하세요." required></textarea></label>
        <label>연기 지시 <small>선택 사항 · 1.7B instruction control</small><textarea bind:value={speechForm.instructions} rows="3" placeholder="예: 기쁘고 활기찬 목소리로, 중요한 단어는 힘주어 말해 주세요."></textarea></label>
        <div class="fields three">
          <label>언어<select bind:value={speechForm.language}><option>Korean</option><option>English</option><option>Chinese</option><option>Japanese</option><option>Auto</option></select></label>
          <label>화자<select bind:value={speechForm.speaker}><option>Sohee</option><option>Vivian</option><option>Serena</option><option>Ryan</option><option>Aiden</option><option>Ono_Anna</option></select></label>
          <label>시드 <small>-1은 무작위</small><input type="number" min="-1" bind:value={speechForm.seed}></label>
        </div>
        <button class="primary" disabled={busy || activeJobs().some((j) => j.kind === 'speech')}>{busy ? '요청 중…' : '음성 만들기'}</button>
      </form>
      <aside><h3>최근 음성</h3><div class="audio-list">
        {#each jobs.filter((j) => j.kind === 'speech').slice(0, 10) as job}<article><div><span>{job.params?.speaker}{#if job.params?.seed >= 0} · seed {job.params.seed}{/if}</span><p>{job.prompt}</p></div>{#if job.params?.instructions}<small class="instruction">지시 · {job.params.instructions}</small>{/if}{#if job.output_url}<audio controls src={job.output_url}></audio>{:else}<small>{job.status}</small>{/if}{#if job.error}<em>{job.error}</em>{/if}{#if job.status === 'completed' || job.status === 'failed'}<button class="job-delete" disabled={deletingJob === job.id} onclick={() => deleteJob(job)}>삭제</button>{/if}</article>{:else}<div class="empty">첫 음성이 여기에 나타납니다.</div>{/each}
      </div></aside>
    </section>
  {:else if tab === 'recognition'}
    <section class="workspace">
      <form onsubmit={(e) => { e.preventDefault(); recognizeSpeech() }}>
        <div class="section-title"><div><span>04</span><h2>음성 받아쓰기</h2></div></div>
        <label class="file-field">음성 파일<input type="file" accept="audio/*,.wav,.flac,.ogg,.mp3" onchange={(e) => recognitionFile = e.currentTarget.files?.[0] || null}><span>{recognitionFile?.name || '음성 파일 선택'}</span></label>
        <div class="fields">
          <label>언어<select bind:value={recognitionForm.language}><option>Auto</option><option>Korean</option><option>English</option><option>Chinese</option><option>Japanese</option></select></label>
          <label>최대 크기<input value={`${config?.recognition.max_upload_mb || 500} MB`} disabled></label>
        </div>
        <label>문맥·전문용어<textarea bind:value={recognitionForm.context} rows="4" placeholder="선택 사항 · 인명, 제품명, 전문용어 등을 입력하세요."></textarea></label>
        <button class="primary" disabled={busy || !recognitionFile || activeJobs().some((j) => j.kind === 'recognition')}>{busy ? '요청 중…' : '음성을 텍스트로 변환'}</button>
      </form>
      <aside><h3>최근 받아쓰기</h3><div class="audio-list">
        {#each jobs.filter((j) => j.kind === 'recognition').slice(0, 10) as job}<article><div><span>{job.params?.detected_language || job.params?.language}</span><p>{job.prompt}</p></div>{#if job.params?.text}<p class="transcript">{job.params.text}</p>{:else}<small>{job.status}</small>{/if}{#if job.output_url}<a href={job.output_url} target="_blank">텍스트 열기 ↗</a>{/if}{#if job.error}<em>{job.error}</em>{/if}{#if job.status === 'completed' || job.status === 'failed'}<button class="job-delete" disabled={deletingJob === job.id} onclick={() => deleteJob(job)}>삭제</button>{/if}</article>{:else}<div class="empty">첫 받아쓰기가 여기에 나타납니다.</div>{/each}
      </div></aside>
    </section>
  {:else if tab === 'settings' && settings}
    <form class="settings" onsubmit={(e) => { e.preventDefault(); saveSettings() }}>
      <div class="section-title"><div><span>SET</span><h2>연결 및 기본 설정</h2></div></div>
      {#if savedMessage}<div class="success">{savedMessage}</div>{/if}

      <section class="settings-card">
        <h3>Media 앱</h3>
        <p>Listen 주소와 데이터 폴더는 저장되지만 실행 중인 서버에는 재시작 후 적용됩니다.</p>
        <div class="fields">
          <label>Listen 주소<input bind:value={settings.listen} required></label>
          <label>데이터 폴더<input bind:value={settings.data_dir} required></label>
        </div>
      </section>

      <section class="settings-card">
        <h3>API 연결</h3>
        <div class="endpoint-list">
          {#each [['image', 'Klein 이미지'], ['video', 'LTX 영상'], ['speech', 'Qwen3 TTS'], ['recognition', 'Qwen3 ASR'], ['prompt', 'Gemma 프롬프트 향상']] as item}
            <label><span>{item[1]} <small class:online={engineStates[item[0]] === 'online'}>{engineStates[item[0]] || 'offline'}</small></span><input type="url" bind:value={settings.engines[item[0]].endpoint} required></label>
          {/each}
        </div>
      </section>

      <div class="settings-grid">
        <section class="settings-card">
          <h3>이미지</h3>
          <label>모델<input bind:value={settings.image.model} required></label>
          <div class="fields three">
            <label>기본 너비<input type="number" min="256" step="16" bind:value={settings.image.default_width}></label>
            <label>기본 높이<input type="number" min="256" step="16" bind:value={settings.image.default_height}></label>
            <label>참조 이미지 수<input type="number" min="1" max="16" bind:value={settings.image.max_reference_images}></label>
          </div>
        </section>

        <section class="settings-card">
          <h3>영상</h3>
          <label>모델<input bind:value={settings.video.model} required></label>
          <div class="fields">
            <label>기본 너비<input type="number" min="256" step="64" bind:value={settings.video.default_width}></label>
            <label>기본 높이<input type="number" min="256" step="64" bind:value={settings.video.default_height}></label>
            <label>기본 프레임<input type="number" min="9" step="8" bind:value={settings.video.default_frames}></label>
            <label>기본 FPS<input type="number" min="1" max="60" bind:value={settings.video.default_fps}></label>
          </div>
          <div class="prompt-settings">
            <label>향상 모델<input bind:value={settings.prompt_enhancement.model} required></label>
            <div class="fields three">
              <label>기본 사용<select bind:value={settings.prompt_enhancement.default_enabled}><option value={true}>자동</option><option value={false}>꺼짐</option></select></label>
              <label>최대 토큰<input type="number" min="64" max="2048" bind:value={settings.prompt_enhancement.max_tokens}></label>
              <label>이미지 인식<select bind:value={settings.prompt_enhancement.vision_enabled}><option value={false}>꺼짐</option><option value={true}>사용</option></select></label>
            </div>
            <small>현재 Huihui LiteRT 번들은 이미지 인식이 되지 않으므로 이미지 인식은 꺼짐을 유지하세요.</small>
          </div>
        </section>

        <section class="settings-card">
          <h3>음성 생성</h3>
          <label>CustomVoice 모델<input bind:value={settings.speech.custom_voice_model} required></label>
          <div class="fields">
            <label>기본 언어<input bind:value={settings.speech.default_language} required></label>
            <label>기본 화자<input bind:value={settings.speech.default_speaker} required></label>
          </div>
        </section>

        <section class="settings-card">
          <h3>음성 인식</h3>
          <label>ASR 모델<input bind:value={settings.recognition.model} required></label>
          <div class="fields">
            <label>기본 언어<input bind:value={settings.recognition.default_language} required></label>
            <label>최대 업로드 MB<input type="number" min="1" bind:value={settings.recognition.max_upload_mb}></label>
          </div>
        </section>
      </div>
      <button class="primary settings-save" disabled={busy}>{busy ? '저장 중…' : '설정 저장'}</button>
    </form>
  {:else}
    <section class="history"><div class="section-title"><div><span>05</span><h2>생성 기록</h2></div>{#if jobs.some((job) => job.status !== 'queued' && job.status !== 'running')}<button class="quiet danger" disabled={deletingJob === 'all'} onclick={clearFinishedJobs}>모두 비우기</button>{/if}</div>
      {#each jobs as job}<article><span class="kind">{job.kind}</span><div><strong>{job.prompt}</strong><small>{new Date(job.created_at).toLocaleString()} · {job.status}</small>{#if job.error}<em>{job.error}</em>{/if}</div><div class="job-actions">{#if job.output_url}<a href={job.output_url} target="_blank">열기 ↗</a>{/if}{#if job.status === 'completed' || job.status === 'failed'}<button class="job-delete" disabled={deletingJob === job.id} onclick={() => deleteJob(job)}>삭제</button>{/if}</div></article>{:else}<div class="empty">아직 생성 기록이 없습니다.</div>{/each}
    </section>
  {/if}
</main>
