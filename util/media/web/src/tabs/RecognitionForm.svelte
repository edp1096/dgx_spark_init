<script>
  import { recognitionLanguages } from '../lib/catalogs.js'
  import { videoJobDuration } from '../lib/mediaPresentation.js'
  import { formatDuration } from '../lib/videoTiming.js'

  export let form
  export let file = null
  export let fileInput = null
  export let sourceVideoJob = null
  export let options = null
  export let selectedPart = null
  export let loadingOptions = false
  export let config = null
  export let busy = false
  export let activeJobs = []
  export let onReset = () => {}
  export let onSubmit = () => {}
  export let onClearSourceVideo = () => {}
  export let onURL = () => {}
  export let onLoadOptions = () => {}
  export let onFile = () => {}
  export let onOpenVideoPicker = () => {}
  export let onClearFile = () => {}
  export let onSelectPart = () => {}

  $: submitDisabled = busy
    || form.output_formats.length === 0
    || (form.source === 'file'
      ? !file
      : form.source === 'video_job'
        ? !sourceVideoJob
        : !form.url.trim())
</script>

<form class="mobile-create-pane" onsubmit={(event) => { event.preventDefault(); onSubmit() }}>
  <div class="section-title"><div><span>04</span><h2>자막 받아쓰기</h2></div><div class="image-title-actions"><button type="button" class="quiet image-create-reset" disabled={busy} title="받아쓰기 설정을 모두 비웁니다." onclick={onReset}>초기화</button></div></div>
  <section class="recognition-source-panel">
    <div class="recognition-source-heading"><div><strong>입력 소스</strong><small>링크·로컬 파일·생성 영상 중 하나를 사용하세요.</small></div></div>
    {#if sourceVideoJob}
      <div class="recognition-video-source"><span><i>VIDEO</i><strong>생성 영상 {sourceVideoJob.id.slice(0, 8)}</strong><small>{sourceVideoJob.params?.width}×{sourceVideoJob.params?.height} · {formatDuration(videoJobDuration(sourceVideoJob))}</small></span><button type="button" aria-label="생성 영상 선택 해제" onclick={onClearSourceVideo}>×</button></div>
    {/if}
    <div class="recognition-source-bar">
      <div class="recognition-url-input" class:active={form.source === 'url'}><i>URL</i><input aria-label="영상 링크" type="url" value={form.url} oninput={onURL} placeholder="영상 페이지 URL"></div>
      <button type="button" class="quiet media-options-load" disabled={loadingOptions || !form.url.trim()} onclick={onLoadOptions}>{loadingOptions ? '조회 중…' : '조회'}</button>
      <label class="recognition-file-button" class:active={form.source === 'file'} title={file?.name || '영상·음성 파일 선택'}><input bind:this={fileInput} type="file" accept="audio/*,video/*,.mkv,.mp4,.webm,.mov,.m4v,.avi,.wav,.flac,.ogg,.mp3,.m4a,.aac" onchange={onFile}><i>FILE</i><span>{file?.name || '파일 선택'}</span></label>
      <button type="button" class="recognition-video-list-button" class:active={form.source === 'video_job'} onclick={onOpenVideoPicker}><i>VIDEO</i><span>영상 목록</span></button>
      {#if file}<button type="button" class="recognition-file-clear" aria-label="선택 파일 해제" title="선택 파일 해제" onclick={onClearFile}>×</button>{/if}
    </div>
    <small class="recognition-source-note">{form.source === 'video_job' && sourceVideoJob ? '생성 영상 파일을 서버 내부에서 바로 연결합니다.' : form.source === 'file' && file ? '선택한 파일을 작업 폴더로 바로 전송합니다.' : '링크 영상을 보관하고 음성을 분리합니다. 필요하면 브라우저 해석기를 사용합니다.'}</small>
    {#if options}
      <div class="media-options">
        {#if options.parts?.length}
          {#if options.parts.length > 1}
            <div class="media-option-row"><strong>파트</strong><div class="media-option-buttons">{#each options.parts as part (part.id)}<button type="button" class:active={form.media_part === part.id} onclick={() => onSelectPart(part.id)}>{part.label}</button>{/each}</div></div>
          {/if}
          <div class="media-option-row"><strong>영상 출처</strong><div class="media-option-buttons"><button type="button" class:active={!form.media_source} onclick={() => form.media_source = ''}>자동 · StreamTape 우선</button>{#each selectedPart?.sources || [] as source (source.id)}<button type="button" class:active={form.media_source === source.id} onclick={() => form.media_source = source.id}>{source.label}</button>{/each}</div></div>
        {:else}
          <small>별도 파트나 출처 선택 없이 기본 방식으로 처리합니다.</small>
        {/if}
      </div>
    {/if}
  </section>
  <div class="fields">
    <label>언어<select bind:value={form.language}>{#each recognitionLanguages as option}<option value={option[0]}>{option[1]}</option>{/each}</select></label>
    <label>구간 길이<input value={`${config?.recognition.segment_seconds || 180}초`} disabled></label>
  </div>
  <label>컨텍스트·전문용어<textarea bind:value={form.context} rows="4" placeholder="선택 사항 · 인명, 제품명, 전문용어 등을 입력하세요."></textarea></label>
  <fieldset class="format-options">
    <legend>결과 형식 <small>복수 선택 가능</small></legend>
    <label><input type="checkbox" value="srt" bind:group={form.output_formats}>SRT 자막</label>
    <label><input type="checkbox" value="vtt" bind:group={form.output_formats}>VTT 자막</label>
    <label><input type="checkbox" value="timestamped_txt" bind:group={form.output_formats}>타임코드 TXT</label>
    <label><input type="checkbox" value="txt" bind:group={form.output_formats}>일반 TXT</label>
  </fieldset>
  <div class="fields">
    <label>번역<select bind:value={form.translation_mode}><option value="none">번역 안 함</option><option value="translated">번역문만</option><option value="bilingual">원문과 번역문</option></select></label>
    <label>번역 언어<input list="translation-languages" bind:value={form.target_language} disabled={form.translation_mode === 'none'} placeholder="Korean"></label>
  </div>
  <button class="primary" disabled={submitDisabled}>{busy ? '등록 중…' : activeJobs.some((job) => job.kind === 'recognition') ? '자막 큐에 추가' : '자막 만들기'}</button>
</form>
