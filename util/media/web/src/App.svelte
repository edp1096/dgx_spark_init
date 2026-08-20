<script>
  import { onMount } from 'svelte'
  import { api } from './api.js'
  import ResultPagination from './ResultPagination.svelte'

  let tab = 'image'
  let config = null
  let settings = null
  let savedMessage = ''
  let jobs = []
  let engineStates = { image: 'offline', speech: 'offline', recognition: 'offline', video: 'offline', prompt: 'offline', media: 'offline' }
  let busy = false
  let error = ''
  let refs = []
  let imageForm = { prompt: '', width: 1024, height: 1024, seed: -1 }
  let speechForm = { text: '', instructions: '', language: 'Korean', speaker: 'Sohee', seed: -1 }
  let recognitionForm = {
    source: 'file', url: '', language: 'Auto', context: '',
    output_formats: ['srt', 'txt'], translation_mode: 'none', target_language: 'Korean',
    media_part: '', media_source: ''
  }
  let recognitionFile = null
  let recognitionOptions = null
  let loadingRecognitionOptions = false
  let videoForm = { prompt: '', width: 768, height: 512, fps: 24, seed: -1, image_strength: 1 }
  let videoDurationSeconds = 5
  let settingsVideoDurationSeconds = 5
  let videoImage = null
  let videoEnhanceEnabled = true
  let videoEnhancedPrompt = ''
  let videoEnhancedSource = ''
  let videoEnhancedImageKey = ''
  let enhancingPrompt = false
  let deletingJob = ''
  let storage = null
  let cleaningStorage = false
  let subtitleView = 'gallery'
  let imageView = 'gallery'
  let videoView = 'gallery'
  let expandedVideoJobs = new Set()
  let expandedSubtitleJobs = new Set()
  let refreshSequence = 0
  const pageSizeOptions = [8, 10, 20, 50, 100]
  let listPageSizes = { image: 8, video: 8, speech: 10, recognition: 10, history: 20 }
  let listPages = { image: 1, video: 1, speech: 1, recognition: 1, history: 1 }

  const engineMeta = {
    image: ['image', 'Klein'],
    video: ['video', 'LTX'],
    speech: ['speech', 'TTS'],
    recognition: ['media', 'Media']
  }
  const outputLabels = { srt: 'SRT', vtt: 'VTT', timestamped_txt: '타임코드 TXT', txt: '일반 TXT' }
  const kindLabels = { image: '이미지', video: '영상', speech: '음성', recognition: '자막' }
  const languageCodes = { Korean: 'ko', Japanese: 'ja', English: 'en', Chinese: 'zh' }
  const translationLanguages = [
    'Korean', 'Japanese', 'English', 'Chinese', 'Traditional Chinese',
    'Spanish', 'French', 'German', 'Italian', 'Portuguese', 'Russian',
    'Arabic', 'Hindi', 'Vietnamese', 'Thai', 'Indonesian', 'Turkish',
    'Dutch', 'Polish', 'Ukrainian'
  ]

  function captionLanguage(job) {
    const language = job.params?.translation_mode === 'none'
      ? job.params?.detected_language || job.params?.language
      : job.params?.target_language
    return languageCodes[language] || 'und'
  }

  function formatBytes(value) {
    const bytes = Number(value) || 0
    if (bytes >= 1073741824) return `${(bytes / 1073741824).toFixed(2)} GB`
    if (bytes >= 1048576) return `${(bytes / 1048576).toFixed(1)} MB`
    if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${bytes} B`
  }

  function isAudioMedia(job) {
    const media = job.params?.media || {}
    return media.media_type === 'audio' || String(media.content_type || '').startsWith('audio/')
  }

  function mediaToggleLabel(job, expanded) {
    if (!job.media_url) return expanded ? '자막 접기' : '자막 보기'
    const kind = isAudioMedia(job) ? '음성' : '영상'
    return expanded ? `${kind}·자막 접기` : `${kind}·자막 보기`
  }

  function mediaSummary(job) {
    const media = job.params?.media
    if (!media) return ''
    const dimensions = !isAudioMedia(job) && media.width && media.height ? `${media.width}×${media.height} · ` : ''
    return `${dimensions}${formatDuration(media.duration)} · ${formatBytes(media.size)}`
  }

  function durationFromFrames(frames, fps) {
    return Math.round(Math.max(0, (Number(frames) - 1) / Math.max(1, Number(fps))) * 1000) / 1000
  }

  function framesForDuration(seconds, fps) {
    const rawFrames = Math.max(0, Number(seconds) || 0) * Math.max(1, Number(fps) || 1)
    return Math.max(9, Math.round(rawFrames / 8) * 8 + 1)
  }

  function formatDuration(seconds) {
    const total = Math.max(0, Number(seconds) || 0)
    const hours = Math.floor(total / 3600)
    const minutes = Math.floor((total % 3600) / 60)
    const secs = Math.round((total % 60) * 10) / 10
    const secondText = Number.isInteger(secs) ? String(secs).padStart(2, '0') : secs.toFixed(1).padStart(4, '0')
    if (hours) return `${hours}:${String(minutes).padStart(2, '0')}:${secondText}`
    return `${minutes}:${secondText}`
  }

  function videoJobDuration(job) {
    return (Math.max(1, Number(job.params?.num_frames) || 1) - 1) / Math.max(1, Number(job.params?.fps) || 1)
  }

  function recognitionProgressText(job) {
    const params = job.params || {}
    if (params.stage === 'media') {
      const labels = {
        starting: '미디어 준비 시작 중', resuming: '저장된 원본에서 작업 재개 중', receiving: '파일 전송 중', resolving: '영상 페이지 분석 중',
        storing: '미디어 저장·재생 형식 정리 중', extracting_audio: '음성 추출·분할 중', complete: '미디어 준비 마무리 중'
      }
      if (params.media_stage === 'downloading') {
        const percent = Number(params.media_percent) || 0
        const amount = params.media_total_bytes ? ` · ${formatBytes(params.media_downloaded_bytes)} / ${formatBytes(params.media_total_bytes)}` : ''
        const eta = params.media_eta_seconds ? ` · 약 ${params.media_eta_seconds}초 남음` : ''
        return `미디어 다운로드 ${percent.toFixed(1)}%${amount}${eta}`
      }
      return labels[params.media_stage] || '미디어 준비 중'
    }
    if (params.stage === 'recognition') return params.segments ? `음성 인식 ${params.progress || 0}/${params.segments} 구간` : '음성 인식 준비 중'
    if (params.stage === 'translation') return `자막 번역 ${params.translation_progress || 0}/${params.translation_total || 0} 배치`
    if (params.stage === 'finalizing') return '자막 파일 생성 중'
    return job.status
  }

  function recognitionProgressPercent(job) {
    const params = job.params || {}
    if (params.stage === 'media' && params.media_stage === 'downloading') return Math.min(100, Math.max(0, Number(params.media_percent) || 0))
    if (params.stage === 'recognition' && params.segments) return Math.min(100, (Number(params.progress) || 0) * 100 / params.segments)
    if (params.stage === 'translation' && params.translation_total) return Math.min(100, (Number(params.translation_progress) || 0) * 100 / params.translation_total)
    return 0
  }

  const activeJobs = () => jobs.filter((j) => j.status === 'queued' || j.status === 'running')

  function jobsForList(key) {
    return key === 'history' ? jobs : jobs.filter((job) => job.kind === key)
  }

  function pagedJobs(key) {
    const start = (listPages[key] - 1) * listPageSizes[key]
    return jobsForList(key).slice(start, start + listPageSizes[key])
  }

  function clampListPages() {
    const next = { ...listPages }
    for (const key of Object.keys(next)) {
      const lastPage = Math.max(1, Math.ceil(jobsForList(key).length / listPageSizes[key]))
      next[key] = Math.min(Math.max(1, next[key]), lastPage)
    }
    listPages = next
  }

  function setListPage(key, page) {
    const lastPage = Math.max(1, Math.ceil(jobsForList(key).length / listPageSizes[key]))
    listPages = { ...listPages, [key]: Math.min(Math.max(1, page), lastPage) }
  }

  function setListPageSize(key, pageSize) {
    const size = pageSizeOptions.includes(pageSize) ? pageSize : listPageSizes[key]
    listPageSizes = { ...listPageSizes, [key]: size }
    listPages = { ...listPages, [key]: 1 }
    localStorage.setItem(`media-${key}-page-size`, String(size))
  }

  function showNewestListPage(key) {
    listPages = { ...listPages, [key]: 1 }
  }

  async function refresh() {
    const sequence = ++refreshSequence
    try {
      const [nextJobs, nextEngines] = await Promise.all([api.jobs(), api.engines()])
      if (sequence !== refreshSequence) return
      jobs = [...nextJobs].sort((a, b) => {
        const createdDifference = Date.parse(b.created_at || 0) - Date.parse(a.created_at || 0)
        return createdDifference || String(b.id).localeCompare(String(a.id))
      })
      clampListPages()
      engineStates = Object.fromEntries(nextEngines.map((item) => [item.kind, item.status]))
    } catch (e) {
      if (sequence === refreshSequence) error = e.message
    }
  }

  onMount(() => {
    subtitleView = localStorage.getItem('media-subtitle-view') === 'list' ? 'list' : 'gallery'
    imageView = localStorage.getItem('media-image-view') === 'list' ? 'list' : 'gallery'
    videoView = localStorage.getItem('media-video-view') === 'list' ? 'list' : 'gallery'
    for (const key of Object.keys(listPageSizes)) {
      const storedSize = Number(localStorage.getItem(`media-${key}-page-size`))
      if (pageSizeOptions.includes(storedSize)) listPageSizes = { ...listPageSizes, [key]: storedSize }
    }
    api.config().then((value) => {
      config = value
      settings = structuredClone(value)
      imageForm.width = value.image.default_width
      imageForm.height = value.image.default_height
      speechForm.language = value.speech.default_language
      speechForm.speaker = value.speech.default_speaker
      recognitionForm.language = value.recognition.default_language
      recognitionForm.output_formats = [...value.recognition.default_output_formats]
      recognitionForm.translation_mode = value.recognition.default_translation_mode
      recognitionForm.target_language = value.recognition.default_translation_language
      videoForm.width = value.video.default_width
      videoForm.height = value.video.default_height
      videoForm.fps = value.video.default_fps
      videoDurationSeconds = durationFromFrames(value.video.default_frames, value.video.default_fps)
      videoEnhanceEnabled = value.prompt_enhancement.default_enabled
    }).catch((e) => error = e.message)
    refresh()
    const timer = setInterval(refresh, 1500)
    return () => clearInterval(timer)
  })

  function setSubtitleView(view) {
    subtitleView = view
    localStorage.setItem('media-subtitle-view', view)
  }

  function setImageView(view) {
    imageView = view
    localStorage.setItem('media-image-view', view)
  }

  function setVideoView(view) {
    videoView = view
    localStorage.setItem('media-video-view', view)
  }

  function toggleExpandedJob(kind, id) {
    const current = kind === 'video' ? expandedVideoJobs : expandedSubtitleJobs
    const next = new Set(current)
    if (next.has(id)) next.delete(id)
    else next.add(id)
    if (kind === 'video') expandedVideoJobs = next
    else expandedSubtitleJobs = next
  }

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
      await api.image(form); imageForm.prompt = ''; refs = []; showNewestListPage('image'); await refresh()
    } catch (e) { error = e.message } finally { busy = false }
  }

  async function generateSpeech() {
    busy = true; error = ''
    try {
      const form = new FormData()
      form.append('text', speechForm.text)
      form.append('instructions', speechForm.instructions)
      form.append('language', speechForm.language); form.append('speaker', speechForm.speaker); form.append('seed', speechForm.seed)
      await api.speech(form); speechForm.text = ''; showNewestListPage('speech'); await refresh()
    } catch (e) { error = e.message } finally { busy = false }
  }

  async function recognizeSpeech() {
    if ((recognitionForm.source === 'file' && !recognitionFile) || (recognitionForm.source === 'url' && !recognitionForm.url.trim())) return
    busy = true; error = ''
    try {
      const form = new FormData()
      if (recognitionForm.source === 'file') form.append('media', recognitionFile)
      else form.append('url', recognitionForm.url.trim())
      if (recognitionForm.source === 'url' && recognitionForm.media_part) form.append('media_part', recognitionForm.media_part)
      if (recognitionForm.source === 'url' && recognitionForm.media_source) form.append('media_source', recognitionForm.media_source)
      form.append('language', recognitionForm.language)
      form.append('context', recognitionForm.context)
      form.append('output_formats', recognitionForm.output_formats.join(','))
      form.append('translation_mode', recognitionForm.translation_mode)
      form.append('target_language', recognitionForm.target_language)
      await api.recognition(form)
      showNewestListPage('recognition')
      recognitionFile = null
      recognitionForm.url = ''
      resetRecognitionOptions()
      await refresh()
    } catch (e) { error = e.message } finally { busy = false }
  }

  function resetRecognitionOptions() {
    recognitionOptions = null
    recognitionForm.media_part = ''
    recognitionForm.media_source = ''
  }

  function selectedRecognitionPart() {
    return recognitionOptions?.parts?.find((part) => part.id === recognitionForm.media_part) || recognitionOptions?.parts?.[0]
  }

  function selectRecognitionPart(partID) {
    recognitionForm.media_part = partID
    recognitionForm.media_source = ''
  }

  async function loadRecognitionOptions() {
    const url = recognitionForm.url.trim()
    if (!url) return
    loadingRecognitionOptions = true
    error = ''
    try {
      const options = await api.mediaOptions(url)
      if (url !== recognitionForm.url.trim()) return
      recognitionOptions = options
      recognitionForm.media_part = options.parts?.[0]?.id || ''
      recognitionForm.media_source = ''
    } catch (e) {
      error = e.message
      resetRecognitionOptions()
    } finally {
      loadingRecognitionOptions = false
    }
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
      form.append('num_frames', framesForDuration(videoDurationSeconds, videoForm.fps))
      form.append('original_prompt', videoForm.prompt)
      if (videoImage) form.append('image', videoImage)
      await api.video(form)
      showNewestListPage('video')
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
    settingsVideoDurationSeconds = durationFromFrames(config.video.default_frames, config.video.default_fps)
    savedMessage = ''
    error = ''
    tab = 'settings'
    storage = null
    api.storage().then((value) => storage = value).catch((e) => error = e.message)
  }

  async function cleanupTemporaryStorage() {
    const amount = formatBytes(storage?.reclaimable_bytes || 0)
    if (!confirm(`실행 중인 작업을 제외한 임시 파일 ${amount}을(를) 삭제할까요?`)) return
    cleaningStorage = true; error = ''; savedMessage = ''
    try {
      const result = await api.cleanupTemporaryStorage()
      storage = await api.storage()
      savedMessage = `임시 폴더 ${result.removed_directories}개, ${formatBytes(result.removed_bytes)}을(를) 정리했습니다.`
    } catch (e) { error = e.message }
    finally { cleaningStorage = false }
  }

  async function saveSettings() {
    busy = true; error = ''; savedMessage = ''
    try {
      settings.video.default_frames = framesForDuration(settingsVideoDurationSeconds, settings.video.default_fps)
      const result = await api.saveConfig(settings)
      config = result.config
      settings = structuredClone(result.config)
      imageForm.width = config.image.default_width
      imageForm.height = config.image.default_height
      speechForm.language = config.speech.default_language
      speechForm.speaker = config.speech.default_speaker
      recognitionForm.language = config.recognition.default_language
      recognitionForm.output_formats = [...config.recognition.default_output_formats]
      recognitionForm.translation_mode = config.recognition.default_translation_mode
      recognitionForm.target_language = config.recognition.default_translation_language
      videoForm.width = config.video.default_width
      videoForm.height = config.video.default_height
      videoForm.fps = config.video.default_fps
      videoDurationSeconds = durationFromFrames(config.video.default_frames, config.video.default_fps)
      settingsVideoDurationSeconds = durationFromFrames(config.video.default_frames, config.video.default_fps)
      videoEnhanceEnabled = config.prompt_enhancement.default_enabled
      savedMessage = result.restart_required
        ? '저장했습니다. Listen 주소 또는 데이터 폴더 변경은 Media 재시작 후 적용됩니다.'
        : '저장했습니다. API 연결과 생성 기본값이 즉시 적용됐습니다.'
      await refresh()
    } catch (e) { error = e.message } finally { busy = false }
  }

</script>

<datalist id="translation-languages">
  {#each translationLanguages as language}<option value={language}></option>{/each}
</datalist>

<svelte:head><meta name="theme-color" content="#101318"></svelte:head>

<header>
  <div><span class="mark">✦</span><h1>생성 스튜디오</h1><span class="phase">IMAGE · VIDEO · VOICE · SUBTITLE</span></div>
  <div class="engine-strip">
    {#if engineMeta[tab]}
      <span class:running={engineStates[engineMeta[tab][0]] === 'online'}><i></i>{engineMeta[tab][1]} API · {engineStates[engineMeta[tab][0]] || 'offline'}</span>
    {/if}
    {#if tab === 'video'}
      <span class:running={engineStates.prompt === 'online'}><i></i>Enhancer API · {engineStates.prompt || 'offline'}</span>
    {/if}
    {#if tab === 'recognition'}
      <span class:running={engineStates.recognition === 'online'}><i></i>ASR API · {engineStates.recognition || 'offline'}</span>
      {#if recognitionForm.translation_mode !== 'none'}<span class:running={engineStates.prompt === 'online'}><i></i>Translator API · {engineStates.prompt || 'offline'}</span>{/if}
    {/if}
  </div>
</header>

<main>
  <nav>
    <button class:active={tab === 'image'} onclick={() => tab = 'image'}>이미지</button>
    <button class:active={tab === 'video'} onclick={() => tab = 'video'}>영상</button>
    <button class:active={tab === 'speech'} onclick={() => tab = 'speech'}>음성</button>
    <button class:active={tab === 'recognition'} onclick={() => tab = 'recognition'}>자막</button>
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
      <aside>
        <div class="results-heading">
          <h3>최근 이미지</h3>
          <div class="view-switch" aria-label="최근 이미지 보기 방식">
            <button type="button" class:active={imageView === 'gallery'} onclick={() => setImageView('gallery')}>갤러리</button>
            <button type="button" class:active={imageView === 'list'} onclick={() => setImageView('list')}>리스트</button>
          </div>
        </div>
        <ResultPagination label="최근 이미지" total={jobsForList('image').length} page={listPages.image} pageSize={listPageSizes.image} pageSizes={pageSizeOptions} onPageChange={(page) => setListPage('image', page)} onPageSizeChange={(size) => setListPageSize('image', size)} />
        <div class="gallery image-results" class:list-view={imageView === 'list'}>
        {#each pagedJobs('image') as job (job.id)}
          <article class:pending={job.status !== 'completed'}>
            {#if imageView === 'list'}
              {#if job.output_url}<a class="image-list-thumb" href={job.output_url} target="_blank" aria-label="생성 이미지 열기"><img src={job.output_url} alt={job.prompt}></a>{:else}<div class="image-list-thumb placeholder"><span>{job.status}</span></div>{/if}
              <div class="image-list-content"><span>{job.params?.width || '—'}×{job.params?.height || '—'}{#if job.params?.seed >= 0} · seed {job.params.seed}{/if}</span><p>{job.prompt}</p>{#if job.error}<em>{job.error}</em>{/if}</div>
            {:else}
              {#if job.output_url}<img src={job.output_url} alt={job.prompt}>{:else}<div class="placeholder"><span>{job.status}</span></div>{/if}<p>{job.prompt}</p>{#if job.error}<em>{job.error}</em>{/if}
            {/if}
            {#if job.status === 'completed' || job.status === 'failed'}<button class="job-delete" disabled={deletingJob === job.id} onclick={() => deleteJob(job)}>삭제</button>{/if}
          </article>
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
          <label class="duration-field"><span>길이 (초) <small>{framesForDuration(videoDurationSeconds, videoForm.fps)} 프레임 · 8k+1</small></span><input aria-label="영상 길이 초" type="number" min="0.1" step="0.1" bind:value={videoDurationSeconds}></label>
          <label>FPS<input type="number" min="1" max="60" step="1" bind:value={videoForm.fps}></label>
          <label><span>시드 <small>-1은 무작위</small></span><input type="number" min="-1" bind:value={videoForm.seed}></label>
          <label>이미지 강도<input type="number" min="0" max="1" step="0.05" bind:value={videoForm.image_strength}></label>
        </div>
        <button class="primary" disabled={busy || enhancingPrompt || activeJobs().some((j) => j.kind === 'video')}>{enhancingPrompt ? '프롬프트 향상 중…' : busy ? '요청 중…' : videoEnhancementActive() && !videoEnhancementCurrent() ? '프롬프트 향상 후 확인' : '영상 만들기'}</button>
      </form>
      <aside>
        <div class="results-heading">
          <h3>최근 영상</h3>
          <div class="view-switch" aria-label="최근 영상 보기 방식">
            <button type="button" class:active={videoView === 'gallery'} onclick={() => setVideoView('gallery')}>갤러리</button>
            <button type="button" class:active={videoView === 'list'} onclick={() => setVideoView('list')}>리스트</button>
          </div>
        </div>
        <ResultPagination label="최근 영상" total={jobsForList('video').length} page={listPages.video} pageSize={listPageSizes.video} pageSizes={pageSizeOptions} onPageChange={(page) => setListPage('video', page)} onPageSizeChange={(size) => setListPageSize('video', size)} />
        <div class="video-list" class:list-view={videoView === 'list'}>
        {#each pagedJobs('video') as job (job.id)}
          <article class:pending={job.status !== 'completed'}>
            {#if videoView === 'list'}
              <div class="video-list-thumb" class:empty-thumb={!job.output_url}>{#if job.output_url}<!-- svelte-ignore a11y_media_has_caption --><video preload="metadata" muted playsinline src={job.output_url}></video>{:else}<span>{job.status}</span>{/if}</div>
              <div class="video-list-content"><small>{job.params?.width}×{job.params?.height} · {formatDuration(videoJobDuration(job))} · {job.params?.fps} fps{#if job.params?.seed >= 0} · seed {job.params.seed}{/if}</small><p>{job.prompt}</p>{#if job.error}<em>{job.error}</em>{/if}{#if job.output_url}<button type="button" class="list-preview-toggle" aria-expanded={expandedVideoJobs.has(job.id)} onclick={() => toggleExpandedJob('video', job.id)}>{expandedVideoJobs.has(job.id) ? '영상 접기' : '영상 보기'}</button>{/if}</div>
              {#if job.output_url && expandedVideoJobs.has(job.id)}<div class="video-list-expanded"><!-- svelte-ignore a11y_media_has_caption --><video controls preload="metadata" src={job.output_url}></video></div>{/if}
            {:else}
              {#if job.output_url}<!-- svelte-ignore a11y_media_has_caption --><video controls preload="metadata" src={job.output_url}></video>{:else}<div class="video-placeholder"><span>{job.status}</span></div>{/if}<p>{job.prompt}</p><small>{job.params?.width}×{job.params?.height} · {formatDuration(videoJobDuration(job))} · {job.params?.fps} fps{#if job.params?.seed >= 0} · seed {job.params.seed}{/if}</small>{#if job.error}<em>{job.error}</em>{/if}
            {/if}
            {#if job.status === 'completed' || job.status === 'failed'}<button class="job-delete" disabled={deletingJob === job.id} onclick={() => deleteJob(job)}>삭제</button>{/if}
          </article>
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
          <label><span>시드 <small>-1은 무작위</small></span><input type="number" min="-1" bind:value={speechForm.seed}></label>
        </div>
        <button class="primary" disabled={busy || activeJobs().some((j) => j.kind === 'speech')}>{busy ? '요청 중…' : '음성 만들기'}</button>
      </form>
      <aside><div class="results-heading"><h3>최근 음성</h3></div>
        <ResultPagination label="최근 음성" total={jobsForList('speech').length} page={listPages.speech} pageSize={listPageSizes.speech} pageSizes={pageSizeOptions} onPageChange={(page) => setListPage('speech', page)} onPageSizeChange={(size) => setListPageSize('speech', size)} />
        <div class="audio-list">
        {#each pagedJobs('speech') as job (job.id)}<article><div><span>{job.params?.speaker}{#if job.params?.seed >= 0} · seed {job.params.seed}{/if}</span><p>{job.prompt}</p></div>{#if job.params?.instructions}<small class="instruction">지시 · {job.params.instructions}</small>{/if}{#if job.output_url}<audio controls src={job.output_url}></audio>{:else}<small>{job.status}</small>{/if}{#if job.error}<em>{job.error}</em>{/if}{#if job.status === 'completed' || job.status === 'failed'}<button class="job-delete" disabled={deletingJob === job.id} onclick={() => deleteJob(job)}>삭제</button>{/if}</article>{:else}<div class="empty">첫 음성이 여기에 나타납니다.</div>{/each}
      </div></aside>
    </section>
  {:else if tab === 'recognition'}
    <section class="workspace">
      <form onsubmit={(e) => { e.preventDefault(); recognizeSpeech() }}>
        <div class="section-title"><div><span>04</span><h2>자막과 스크립트</h2></div></div>
        <div class="segmented source-selector">
          <button type="button" class:active={recognitionForm.source === 'file'} onclick={() => recognitionForm.source = 'file'}>파일 업로드</button>
          <button type="button" class:active={recognitionForm.source === 'url'} onclick={() => recognitionForm.source = 'url'}>영상 링크</button>
        </div>
        {#if recognitionForm.source === 'file'}
          <label class="file-field">영상·음성 파일<input type="file" accept="audio/*,video/*,.mkv,.mp4,.webm,.mov,.m4v,.avi,.wav,.flac,.ogg,.mp3,.m4a,.aac" onchange={(e) => recognitionFile = e.currentTarget.files?.[0] || null}><span>{recognitionFile?.name || '영상 또는 음성 파일 선택'}</span></label>
          <small class="form-note">긴 파일은 메모리에 올리지 않고 작업 폴더로 스트리밍 업로드합니다.</small>
        {:else}
          <label>영상 페이지 주소<input type="url" bind:value={recognitionForm.url} oninput={resetRecognitionOptions} placeholder="https://www.youtube.com/watch?v=…" required></label>
          <small class="form-note">영상을 호스트 저장소에 보관하고 음성을 분리합니다. 직접 추출 실패 시 Chromium·Firefox 해석기를 사용합니다.</small>
          <button type="button" class="quiet media-options-load" disabled={loadingRecognitionOptions || !recognitionForm.url.trim()} onclick={loadRecognitionOptions}>{loadingRecognitionOptions ? '영상 내부 선택지 조회 중…' : '영상 내부 선택지 조회'}</button>
          {#if recognitionOptions}
            <div class="media-options">
              {#if recognitionOptions.parts?.length}
                {#if recognitionOptions.parts.length > 1}
                  <div class="media-option-row"><strong>파트</strong><div class="media-option-buttons">
                    {#each recognitionOptions.parts as part (part.id)}<button type="button" class:active={recognitionForm.media_part === part.id} onclick={() => selectRecognitionPart(part.id)}>{part.label}</button>{/each}
                  </div></div>
                {/if}
                <div class="media-option-row"><strong>영상 출처</strong><div class="media-option-buttons">
                  <button type="button" class:active={!recognitionForm.media_source} onclick={() => recognitionForm.media_source = ''}>자동 · StreamTape 우선</button>
                  {#each selectedRecognitionPart()?.sources || [] as source (source.id)}<button type="button" class:active={recognitionForm.media_source === source.id} onclick={() => recognitionForm.media_source = source.id}>{source.label}</button>{/each}
                </div></div>
              {:else}
                <small>이 주소에는 선택 가능한 파트나 별도 영상 출처가 없습니다. 기본 방식으로 처리합니다.</small>
              {/if}
            </div>
          {/if}
        {/if}
        <div class="fields">
          <label>언어<select bind:value={recognitionForm.language}><option>Auto</option><option>Korean</option><option>English</option><option>Chinese</option><option>Japanese</option></select></label>
          <label>구간 길이<input value={`${config?.recognition.segment_seconds || 180}초`} disabled></label>
        </div>
        <label>문맥·전문용어<textarea bind:value={recognitionForm.context} rows="4" placeholder="선택 사항 · 인명, 제품명, 전문용어 등을 입력하세요."></textarea></label>
        <fieldset class="format-options">
          <legend>결과 형식 <small>복수 선택 가능</small></legend>
          <label><input type="checkbox" value="srt" bind:group={recognitionForm.output_formats}>SRT 자막</label>
          <label><input type="checkbox" value="vtt" bind:group={recognitionForm.output_formats}>VTT 자막</label>
          <label><input type="checkbox" value="timestamped_txt" bind:group={recognitionForm.output_formats}>타임코드 TXT</label>
          <label><input type="checkbox" value="txt" bind:group={recognitionForm.output_formats}>일반 TXT</label>
        </fieldset>
        <div class="fields">
          <label>번역<select bind:value={recognitionForm.translation_mode}><option value="none">번역 안 함</option><option value="translated">번역문만</option><option value="bilingual">원문과 번역문</option></select></label>
          <label>번역 언어<input list="translation-languages" bind:value={recognitionForm.target_language} disabled={recognitionForm.translation_mode === 'none'} placeholder="Korean"></label>
        </div>
        <button class="primary" disabled={busy || recognitionForm.output_formats.length === 0 || (recognitionForm.source === 'file' ? !recognitionFile : !recognitionForm.url.trim()) || activeJobs().some((j) => j.kind === 'recognition')}>{busy ? '요청 중…' : '자막 만들기'}</button>
      </form>
      <aside>
        <div class="results-heading">
          <h3>최근 자막</h3>
          <div class="view-switch" aria-label="최근 자막 보기 방식">
            <button type="button" class:active={subtitleView === 'gallery'} onclick={() => setSubtitleView('gallery')}>갤러리</button>
            <button type="button" class:active={subtitleView === 'list'} onclick={() => setSubtitleView('list')}>리스트</button>
          </div>
        </div>
        <ResultPagination label="최근 자막" total={jobsForList('recognition').length} page={listPages.recognition} pageSize={listPageSizes.recognition} pageSizes={pageSizeOptions} onPageChange={(page) => setListPage('recognition', page)} onPageSizeChange={(size) => setListPageSize('recognition', size)} />
        <div class="audio-list subtitle-results" class:list-view={subtitleView === 'list'}>
        {#each pagedJobs('recognition') as job (job.id)}
          <article>
            {#if subtitleView === 'list'}
              <div class="subtitle-list-thumb" class:empty-thumb={!job.media_url || isAudioMedia(job)}>
                {#if job.media_url && !isAudioMedia(job)}
                  <!-- svelte-ignore a11y_media_has_caption -->
                  <video preload="metadata" muted playsinline src={job.media_url}></video>
                {:else}<span>{job.media_url && isAudioMedia(job) ? 'AUDIO' : job.status}</span>{/if}
              </div>
            {/if}
            <div class="subtitle-result-title"><span>{job.params?.detected_language || job.params?.language}{#if job.params?.segments} · {job.params.segments}구간{/if}{#if job.params?.media_part} · 파트 {job.params.media_part}{/if}{#if job.params?.media_source} · {job.params.media_source}{/if}</span><p>{job.prompt}</p></div>
            {#if subtitleView === 'gallery' && job.media_url}
              <div class="subtitle-player">
                {#if isAudioMedia(job)}
                  <audio controls preload="metadata" src={job.media_url}></audio>
                {:else}
                  <video controls preload="metadata">
                    <source src={job.media_url}>
                    {#if job.caption_url}<track kind="subtitles" src={job.caption_url} srclang={captionLanguage(job)} label={job.params?.translation_mode === 'none' ? '원문' : job.params?.target_language || '번역'} default>{/if}
                  </video>
                {/if}
                {#if job.params?.media}<small>{mediaSummary(job)}</small>{/if}
              </div>
            {/if}
            {#if subtitleView === 'gallery' && job.params?.text}
              <details class="transcript-details"><summary>자막 미리보기</summary><p class="transcript">{job.params.text}</p></details>
            {:else if subtitleView === 'list' && (job.media_url || job.params?.text)}
              <button type="button" class="subtitle-list-preview-toggle" aria-expanded={expandedSubtitleJobs.has(job.id)} onclick={() => toggleExpandedJob('subtitle', job.id)}>{mediaToggleLabel(job, expandedSubtitleJobs.has(job.id))}</button>
              {#if expandedSubtitleJobs.has(job.id)}<div class="subtitle-list-expanded">
                {#if job.media_url}
                  <div class="subtitle-player">
                    {#if isAudioMedia(job)}
                      <audio controls preload="metadata" src={job.media_url}></audio>
                    {:else}
                      <video controls preload="metadata">
                        <source src={job.media_url}>
                        {#if job.caption_url}<track kind="subtitles" src={job.caption_url} srclang={captionLanguage(job)} label={job.params?.translation_mode === 'none' ? '원문' : job.params?.target_language || '번역'} default>{/if}
                      </video>
                    {/if}
                    {#if job.params?.media}<small>{mediaSummary(job)}</small>{/if}
                  </div>
                {/if}
                {#if job.params?.text}<details class="subtitle-expanded-transcript"><summary>자막 보기</summary><p class="transcript">{job.params.text}</p></details>{/if}
              </div>{/if}
            {:else if !job.params?.text}<small class="recognition-progress-text">{recognitionProgressText(job)}</small>{/if}
            {#if job.status === 'queued' || job.status === 'running'}<div class="recognition-progress" aria-label={recognitionProgressText(job)}><i style={`width: ${recognitionProgressPercent(job)}%`}></i></div>{/if}
            {#if job.outputs}<div class="output-links">{#each Object.entries(job.outputs) as output}<a href={output[1]} target="_blank">{outputLabels[output[0]] || output[0]} ↗</a>{/each}</div>{:else if job.output_url}<a href={job.output_url} target="_blank">결과 열기 ↗</a>{/if}
            {#if job.error}<em>{job.error}</em>{/if}
            {#if job.status === 'completed' || job.status === 'failed'}<button class="job-delete" disabled={deletingJob === job.id} onclick={() => deleteJob(job)}>삭제</button>{/if}
          </article>
        {:else}<div class="empty">첫 자막 작업이 여기에 나타납니다.</div>{/each}
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
          {#each [['image', 'Klein 이미지'], ['video', 'LTX 영상'], ['speech', 'Qwen3 TTS'], ['recognition', 'Qwen3 ASR'], ['prompt', 'Gemma 프롬프트·번역'], ['media', '미디어 접근·FFmpeg']] as item}
            <label><span>{item[1]} <small class:online={engineStates[item[0]] === 'online'}>{engineStates[item[0]] || 'offline'}</small></span><input type="url" bind:value={settings.engines[item[0]].endpoint} required></label>
          {/each}
        </div>
      </section>

      <section class="settings-card storage-card">
        <div class="storage-heading">
          <div><h3>저장소 관리</h3><p>실행 중인 작업은 정리 대상에서 제외됩니다.</p></div>
          <button type="button" class="quiet danger" disabled={cleaningStorage || !storage?.reclaimable_directories} onclick={cleanupTemporaryStorage}>{cleaningStorage ? '정리 중…' : '찌꺼기 정리'}</button>
        </div>
        <div class="storage-stats">
          <span><small>임시 파일</small><strong>{storage ? formatBytes(storage.temporary_bytes) : '확인 중…'}</strong></span>
          <span><small>정리 가능</small><strong>{storage ? `${storage.reclaimable_directories}개 · ${formatBytes(storage.reclaimable_bytes)}` : '확인 중…'}</strong></span>
          <span><small>사용 중</small><strong>{storage ? `${storage.active_directories}개` : '확인 중…'}</strong></span>
        </div>
        <div class="fields storage-policy">
          <label>시작 시 자동 정리<select bind:value={settings.storage.cleanup_on_startup}><option value={true}>사용</option><option value={false}>꺼짐</option></select></label>
          <label>자동 정리 보존 시간<input type="number" min="1" max="8760" bind:value={settings.storage.temp_retention_hours}><small>이 시간보다 오래된 중단 작업만 앱 시작 시 정리합니다.</small></label>
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
            <label class="duration-field"><span>기본 길이 (초) <small>{framesForDuration(settingsVideoDurationSeconds, settings.video.default_fps)} 프레임 · 8k+1</small></span><input aria-label="기본 영상 길이 초" type="number" min="0.1" step="0.1" bind:value={settingsVideoDurationSeconds}></label>
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
          <h3>자막</h3>
          <label>ASR 모델<input bind:value={settings.recognition.model} required></label>
          <div class="fields">
            <label>기본 언어<input bind:value={settings.recognition.default_language} required></label>
            <label>최대 업로드 MB<input type="number" min="1" bind:value={settings.recognition.max_upload_mb}></label>
            <label>구간 길이(초)<input type="number" min="5" max="180" bind:value={settings.recognition.segment_seconds}></label>
            <label>기본 번역<select bind:value={settings.recognition.default_translation_mode}><option value="none">번역 안 함</option><option value="translated">번역문만</option><option value="bilingual">원문과 번역문</option></select></label>
          </div>
          <label>기본 번역 언어<input list="translation-languages" bind:value={settings.recognition.default_translation_language} required></label>
          <fieldset class="format-options settings-formats">
            <legend>기본 결과 형식</legend>
            <label><input type="checkbox" value="srt" bind:group={settings.recognition.default_output_formats}>SRT</label>
            <label><input type="checkbox" value="vtt" bind:group={settings.recognition.default_output_formats}>VTT</label>
            <label><input type="checkbox" value="timestamped_txt" bind:group={settings.recognition.default_output_formats}>타임코드 TXT</label>
            <label><input type="checkbox" value="txt" bind:group={settings.recognition.default_output_formats}>일반 TXT</label>
          </fieldset>
        </section>
      </div>
      <button class="primary settings-save" disabled={busy}>{busy ? '저장 중…' : '설정 저장'}</button>
    </form>
  {:else}
    <section class="history"><div class="section-title"><div><span>05</span><h2>생성 기록</h2></div>{#if jobs.some((job) => job.status !== 'queued' && job.status !== 'running')}<button class="quiet danger" disabled={deletingJob === 'all'} onclick={clearFinishedJobs}>모두 비우기</button>{/if}</div>
      <ResultPagination label="생성 기록" total={jobsForList('history').length} page={listPages.history} pageSize={listPageSizes.history} pageSizes={pageSizeOptions} onPageChange={(page) => setListPage('history', page)} onPageSizeChange={(size) => setListPageSize('history', size)} />
      {#each pagedJobs('history') as job (job.id)}<article><span class="kind">{kindLabels[job.kind] || job.kind}</span><div><strong>{job.prompt}</strong><small>{new Date(job.created_at).toLocaleString()} · {job.status}</small>{#if job.error}<em>{job.error}</em>{/if}</div><div class="job-actions">{#if job.output_url}<a href={job.output_url} target="_blank">열기 ↗</a>{/if}{#if job.status === 'completed' || job.status === 'failed'}<button class="job-delete" disabled={deletingJob === job.id} onclick={() => deleteJob(job)}>삭제</button>{/if}</div></article>{:else}<div class="empty">아직 생성 기록이 없습니다.</div>{/each}
    </section>
  {/if}
</main>
