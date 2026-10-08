<script>
  import PromptComposer from '../PromptComposer.svelte'
  import VideoAudioTimeline from '../VideoAudioTimeline.svelte'
  import VideoConditionTimeline from '../VideoConditionTimeline.svelte'
  import { videoResolutionPresets } from '../lib/catalogs.js'
  import { framesForDuration, snapDimension } from '../lib/videoTiming.js'

  export let activeJobs
  export let addVideoKeyframe
  export let applyVideoResolutionPreset
  export let busy
  export let config
  export let createVideoPromptFromScenes
  export let creatingVideoPrompt
  export let currentVideoResolutionPreset
  export let enhanceVideoPrompt
  export let enhancingPrompt
  export let generateVideo
  export let moveVideoAudio
  export let moveVideoKeyframe
  export let normalizeVideoTiming
  export let openPromptExamples = () => {}
  export let removeVideoAudio
  export let removeVideoKeyframe
  export let resetVideoCreation
  export let resetVideoEnhancement
  export let setVideoConditionImage
  export let showImage
  export let updateVideoKeyframe
  export let videoAccelerationPreview
  export let videoAdvancedOpen
  export let videoAudioClips
  export let videoAudioJob
  export let videoAudioPickerOpen
  export let videoConditioningDisabledReason
  export let videoDurationSeconds
  export let videoEndImage
  export let videoEndStrength
  export let videoEnhanceEnabled
  export let videoEnhancedPrompt
  export let videoEnhancementIsActive
  export let videoEnhancementIsCurrent
  export let videoForm
  export let videoImage
  export let videoImagePickerTarget
  export let videoImagePreview
  export let videoKeyframeCapacity
  export let videoKeyframes
  export let videoPromptCreationMessage
  export let videoPromptPreset
  export let videoRemoteImageTarget
  export let videoTimelineEditorOpen
</script>

<form class="mobile-create-pane" onsubmit={(e) => { e.preventDefault(); generateVideo() }}>
  <div class="section-title"><div><span>02</span><h2>영상 생성</h2></div><div class="image-title-actions"><button type="button" class="quiet header-prompt-tool" onclick={() => openPromptExamples('video')}>예제{#if videoPromptPreset}<b>선택됨</b>{/if}</button><PromptComposer compact storageKey="spark-media-prompt-composer-video" onApply={(prompt, mode) => { const currentPrompt = videoForm.prompt.trimEnd(); videoForm.prompt = mode === 'append' && currentPrompt ? `${currentPrompt}\n${prompt}` : prompt; videoPromptPreset = ''; resetVideoEnhancement() }} /><a class="quiet portrait-lab-open" href="/tools/portrait-lab/" target="_blank" rel="noreferrer">P Lab ↗</a><button type="button" class="quiet image-create-reset" disabled={busy} title="영상 생성 설정을 모두 비웁니다." onclick={resetVideoCreation}>초기화</button></div></div>
  <label>원본 프롬프트 <small>선택 사항 · 비워두면 음성·장면 이미지로 자동 작성</small><textarea bind:value={videoForm.prompt} rows="5" placeholder="직접 입력하거나 비워두고 음성·키프레임으로 자동 작성하세요."></textarea></label>
  <section class="video-audio-panel">
    <div class="video-audio-heading"><div><strong>음성으로 영상 생성</strong><small>여러 음성을 추가하고 영상 안의 시작 위치를 지정합니다.</small></div><button type="button" onclick={() => videoAudioPickerOpen = true}>{videoAudioClips.length ? `음성 추가 · ${videoAudioClips.length}` : '음성 목록'}</button></div>
    {#if videoAudioClips.length}
      <VideoAudioTimeline duration={videoDurationSeconds} clips={videoAudioClips} onMove={moveVideoAudio} />
      <div class="video-audio-sources">
        {#each videoAudioClips as clip, index (clip.id)}
          <div class="video-audio-source">
            <div><i>A{index + 1}</i><span><strong>{clip.job.params?.speaker || '생성 음성'}</strong><small>{clip.job.prompt}</small></span></div>
            <audio controls preload="metadata" src={clip.job.output_url}></audio>
            <label><span>시작</span><input type="number" min="0" max={Math.max(0, videoDurationSeconds - Math.min(clip.duration || 0, videoDurationSeconds))} step="0.01" value={clip.start} onchange={(event) => moveVideoAudio(clip.id, event.currentTarget.value)}><b>초</b></label>
            <button type="button" aria-label={`음성 ${index + 1} 제거`} onclick={() => removeVideoAudio(clip.id)}>×</button>
          </div>
        {/each}
      </div>
    {:else}
      <button type="button" class="video-audio-empty" onclick={() => videoAudioPickerOpen = true}><i>AUDIO</i><span><strong>음성을 선택하세요</strong><small>음성 목록에서 미리 듣고 영상 생성에 연결합니다.</small></span></button>
    {/if}
  </section>
  <section class="video-conditioning">
    <div class="video-conditioning-heading"><div><strong>장면 이미지</strong><small>장면은 시간순으로 가로 나열됩니다.</small></div><div class="video-conditioning-heading-actions"><button type="button" title="타임라인을 큰 화면에서 편집" onclick={() => videoTimelineEditorOpen = true}>크게 편집</button><button type="button" title="선택한 음성·장면 이미지로 LTX 영상 프롬프트 만들기" disabled={creatingVideoPrompt || (!videoAudioJob && !videoImage && !videoEndImage && !videoKeyframes.some((item) => item.image))} onclick={() => createVideoPromptFromScenes(false)}>{creatingVideoPrompt ? '분석 중…' : '프롬프트'}</button><button type="button" disabled={videoKeyframes.length >= videoKeyframeCapacity()} onclick={addVideoKeyframe}>+ 키프레임</button></div></div>
    {#if videoPromptCreationMessage}<small class="video-prompt-creation-message">{videoPromptCreationMessage}</small>{/if}
    <VideoConditionTimeline duration={(framesForDuration(videoDurationSeconds, videoForm.fps) - 1) / Number(videoForm.fps || 1)} fps={videoForm.fps} startImage={videoImage} endImage={videoEndImage} keyframes={videoKeyframes} imageURL={videoImagePreview} onMove={moveVideoKeyframe} />
    <div class="video-scene-cards">
      <article class="video-boundary-card" class:has-image={Boolean(videoImage)}>
        <div class="video-condition-heading"><strong>시작 이미지</strong><small>0초 · 선택 사항</small></div>
        {#if videoImage}
          <button type="button" class="video-condition-preview" title="클릭하여 크게 보기" onclick={(event) => showImage(event, videoImagePreview(videoImage), '영상 시작 이미지')}><img src={videoImagePreview(videoImage)} alt="영상 시작 이미지"></button>
          <span class="video-condition-name" title={videoImage.name}>{videoImage.name}</span>
        {:else}<div class="video-condition-empty">첫 장면을 고정하려면 이미지를 선택하세요.</div>{/if}
        <div class="video-condition-actions"><label><input type="file" accept="image/png,image/jpeg,image/webp" onchange={(e) => setVideoConditionImage('start', e.currentTarget.files?.[0] || null)}><span>파일</span></label><button type="button" onclick={() => videoImagePickerTarget = 'start'}>최근 결과</button><button type="button" onclick={() => videoRemoteImageTarget = 'start'}>URL</button>{#if videoImage}<button type="button" class="danger" onclick={() => setVideoConditionImage('start', null)}>제거</button>{/if}</div>
        {#if videoImage}<label class="video-condition-strength">반영 강도<input type="number" min="0" max="1" step="any" bind:value={videoForm.image_strength}></label>{/if}
      </article>
      {#each videoKeyframes as keyframe, index (keyframe.id)}
          <article class="video-keyframe-card">
            <div class="video-condition-heading"><strong>키프레임 {index + 1}</strong><button type="button" aria-label="키프레임 제거" onclick={() => removeVideoKeyframe(keyframe.id)}>×</button></div>
            {#if keyframe.image}<button type="button" class="video-keyframe-preview" title="클릭하여 크게 보기" onclick={(event) => showImage(event, videoImagePreview(keyframe.image), `영상 키프레임 ${index + 1}`)}><img src={videoImagePreview(keyframe.image)} alt="영상 키프레임 {index + 1}"></button>{:else}<div class="video-keyframe-empty">IMG</div>{/if}
            <div class="video-keyframe-controls">
              <span class="video-condition-name" title={keyframe.image?.name || '이미지 미선택'}>{keyframe.image?.name || '이미지 미선택'}</span>
              <div class="video-condition-actions"><label><input type="file" accept="image/png,image/jpeg,image/webp" onchange={(e) => setVideoConditionImage(`keyframe:${keyframe.id}`, e.currentTarget.files?.[0] || null)}><span>파일</span></label><button type="button" onclick={() => videoImagePickerTarget = `keyframe:${keyframe.id}`}>최근 결과</button><button type="button" onclick={() => videoRemoteImageTarget = `keyframe:${keyframe.id}`}>URL</button></div>
              <div class="video-keyframe-numbers"><label>위치 (초)<input type="number" min={1 / Number(videoForm.fps || 1)} max={Math.max(1 / Number(videoForm.fps || 1), (framesForDuration(videoDurationSeconds, videoForm.fps) - 2) / videoForm.fps)} step="any" value={Number(keyframe.time).toFixed(3)} onchange={(event) => moveVideoKeyframe(keyframe.id, event.currentTarget.value)}></label><label>반영 강도<input type="number" min="0" max="1" step="any" value={keyframe.strength} onchange={(event) => updateVideoKeyframe(keyframe.id, 'strength', event.currentTarget.value)}></label></div>
            </div>
          </article>
      {/each}
      <article class="video-boundary-card" class:has-image={Boolean(videoEndImage)}>
        <div class="video-condition-heading"><strong>마지막 이미지</strong><small>{((framesForDuration(videoDurationSeconds, videoForm.fps) - 1) / videoForm.fps).toFixed(1)}초 · 선택 사항</small></div>
        {#if videoEndImage}
          <button type="button" class="video-condition-preview" title="클릭하여 크게 보기" onclick={(event) => showImage(event, videoImagePreview(videoEndImage), '영상 마지막 이미지')}><img src={videoImagePreview(videoEndImage)} alt="영상 마지막 이미지"></button>
          <span class="video-condition-name" title={videoEndImage.name}>{videoEndImage.name}</span>
        {:else}<div class="video-condition-empty">도착 장면을 고정하려면 이미지를 선택하세요.</div>{/if}
        <div class="video-condition-actions"><label><input type="file" accept="image/png,image/jpeg,image/webp" onchange={(e) => setVideoConditionImage('end', e.currentTarget.files?.[0] || null)}><span>파일</span></label><button type="button" onclick={() => videoImagePickerTarget = 'end'}>최근 결과</button><button type="button" onclick={() => videoRemoteImageTarget = 'end'}>URL</button>{#if videoEndImage}<button type="button" class="danger" onclick={() => setVideoConditionImage('end', null)}>제거</button>{/if}</div>
        {#if videoEndImage}<label class="video-condition-strength">반영 강도<input type="number" min="0" max="1" step="any" bind:value={videoEndStrength}></label>{/if}
      </article>
    </div>
  </section>
  <div class="enhanced-prompt image-enhancer-panel" class:inactive={!videoEnhancementIsActive}>
    <div class="image-enhancer-panel-header">
      <div class="enhancer-panel-title"><strong title="연결된 Gemma 4 12B 모델이 LTX 캡션 형식의 영어 프롬프트로 정리·확장합니다.">프롬프트 향상</strong></div>
      <div class="enhancer-panel-actions">
        <button type="button" class="quiet enhancer-run" disabled={!videoEnhancementIsActive || enhancingPrompt || !videoForm.prompt.trim()} onclick={enhanceVideoPrompt}>{enhancingPrompt ? '향상 중…' : videoEnhancementIsCurrent ? '다시 향상' : '미리 향상'}</button>
        <div class="segmented compact">
          <button type="button" class:active={videoEnhanceEnabled} onclick={() => videoEnhanceEnabled = true}>켜짐</button>
          <button type="button" class:active={!videoEnhanceEnabled} onclick={() => videoEnhanceEnabled = false}>꺼짐</button>
        </div>
      </div>
    </div>
    {#if videoEnhancedPrompt.trim()}
      <textarea bind:value={videoEnhancedPrompt} rows="8" aria-label="향상된 영상 프롬프트"></textarea>
      <small>{videoEnhancementIsActive ? `${videoImage ? '시작 이미지를 분석한' : '텍스트 기반'} 실제 생성 프롬프트입니다. 생성 전에 직접 수정할 수 있습니다.` : '꺼짐 · 기존 결과는 보존되며 실제 생성에는 원문을 사용합니다.'}</small>
    {:else}
      <small>{videoEnhancementIsActive ? '생성 시작 시 자동으로 향상합니다. 먼저 확인하려면 미리 향상을 누르세요.' : videoImage && !config?.prompt_enhancement.vision_enabled ? '현재 향상 모델은 이미지를 볼 수 없어 I2V에서는 원문을 사용합니다.' : '꺼짐 · 실제 생성에는 원문을 사용합니다.'}</small>
    {/if}
  </div>
  <section class="video-output-settings">
    <div class="video-preset-heading"><div><strong>출력 설정</strong><small>{framesForDuration(videoDurationSeconds, videoForm.fps)}프레임 · {videoAccelerationPreview()}</small></div><button type="button" class="quiet" onclick={() => videoAdvancedOpen = !videoAdvancedOpen}>{videoAdvancedOpen ? '간단히' : '고급 설정'}</button></div>
    <div class="video-resolution-presets" aria-label="영상 해상도 프리셋">
      {#each videoResolutionPresets as preset}<button type="button" class:active={currentVideoResolutionPreset() === preset.id} onclick={() => applyVideoResolutionPreset(preset)}><strong>{preset.label}</strong><small>{preset.width}×{preset.height} · {preset.hint}</small></button>{/each}
      <button type="button" class:active={currentVideoResolutionPreset() === 'custom'} onclick={() => videoAdvancedOpen = true}><strong>직접</strong><small>{videoForm.width}×{videoForm.height}</small></button>
    </div>
    <label class="duration-field video-duration-main"><span>길이 (초) <small>{framesForDuration(videoDurationSeconds, videoForm.fps)} 프레임 · 8k+1</small></span><input aria-label="영상 길이 초" type="number" min={8 / Number(videoForm.fps || 24)} step="any" bind:value={videoDurationSeconds} onchange={normalizeVideoTiming}></label>
    {#if videoAdvancedOpen}
      <div class="fields three video-advanced-fields">
        <label>너비<input type="number" min="256" max="1920" step="any" bind:value={videoForm.width} onchange={() => videoForm.width = snapDimension(videoForm.width, 64, 256, 1920)}></label>
        <label>높이<input type="number" min="256" max="1920" step="any" bind:value={videoForm.height} onchange={() => videoForm.height = snapDimension(videoForm.height, 64, 256, 1920)}></label>
        <label>FPS<input type="number" min="1" max="60" step="any" bind:value={videoForm.fps} onchange={normalizeVideoTiming}></label>
        <label><span>시드 <small>-1은 무작위</small></span><input type="number" min="-1" bind:value={videoForm.seed}></label>
      </div>
    {/if}
  </section>
  {#if videoConditioningDisabledReason()}<small class="video-conditioning-error">{videoConditioningDisabledReason()}</small>{/if}
  <button class="primary" disabled={busy || enhancingPrompt || Boolean(videoConditioningDisabledReason())}>{creatingVideoPrompt ? '프롬프트 자동 작성 중…' : enhancingPrompt ? '프롬프트 처리 중…' : busy ? '요청 중…' : activeJobs().some((j) => j.kind === 'image' || j.kind === 'video' || j.kind === 'speech') ? '영상 큐에 추가' : '생성 시작'}</button>
</form>
