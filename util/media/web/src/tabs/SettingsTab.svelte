<script>
  import { checkpointDisplayChoices, imageModeChoices, imageModeMeta, recognitionLanguages } from '../lib/catalogs.js'
  import { formatBytes } from '../lib/mediaPresentation.js'
  import { framesForDuration, snapDimension } from '../lib/videoTiming.js'

  export let settings
  export let settingsSection = 'connection'
  export let savedMessage = ''
  export let engineStates = {}
  export let civitaiToken = ''
  export let hfToken = ''
  export let imageCheckpointStatus = null
  export let videoModelStatus = null
  export let savingDownloadCredentials = false
  export let preparingImageCheckpoints = false
  export let convertingImageCheckpoints = false
  export let preparingVideoModels = false
  export let checkpointSelection = {}
  export let nvfp4Selection = {}
  export let removeBF16Sources = false
  export let storage = null
  export let cleaningStorage = false
  export let settingsVideoDurationSeconds = 0
  export let busy = false
  export let saveSettings = () => {}
  export let saveDownloadCredentials = () => {}
  export let displayCheckpointReady = () => false
  export let checkpointVisible = () => false
  export let setCheckpointVisible = () => {}
  export let prepareImageCheckpoints = () => {}
  export let convertImageCheckpointsNVFP4 = () => {}
  export let prepareVideoModels = () => {}
  export let cleanupTemporaryStorage = () => {}
  export let snapVideoDuration = (value) => value
</script>

<form class="settings" onsubmit={(e) => { e.preventDefault(); saveSettings() }}>
      <div class="section-title"><div><span>SET</span><h2>연결 및 기본 설정</h2></div></div>
      {#if savedMessage}<div class="success">{savedMessage}</div>{/if}

      <div class="settings-tabs" role="tablist" aria-label="설정 구역">
        <button type="button" role="tab" aria-selected={settingsSection === 'connection'} class:active={settingsSection === 'connection'} onclick={() => settingsSection = 'connection'}><span>연결</span><small>앱·API 주소</small></button>
        <button type="button" role="tab" aria-selected={settingsSection === 'defaults'} class:active={settingsSection === 'defaults'} onclick={() => settingsSection = 'defaults'}><span>생성 기본값</span><small>이미지·영상·음성·자막</small></button>
        <button type="button" role="tab" aria-selected={settingsSection === 'metadata'} class:active={settingsSection === 'metadata'} onclick={() => settingsSection = 'metadata'}><span>이미지 정보</span><small>EXIF 제작자 정보</small></button>
        <button type="button" role="tab" aria-selected={settingsSection === 'storage'} class:active={settingsSection === 'storage'} onclick={() => settingsSection = 'storage'}><span>저장소</span><small>용량·자동 정리</small></button>
      </div>

      {#if settingsSection === 'connection'}
      <div class="settings-section connection-settings">
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
          {#each [['video', 'LTX 영상'], ['speech', 'Qwen3 TTS'], ['recognition', 'Qwen3 ASR'], ['prompt', '프롬프트·번역'], ['upscale', 'SeedVR2 고화질'], ['media', '미디어 접근·FFmpeg']] as item}
            <label><span>{item[1]} <small class:online={engineStates[item[0]] === 'online'}>{engineStates[item[0]] || 'offline'}</small></span><input type="url" bind:value={settings.engines[item[0]].endpoint} required></label>
          {/each}
        </div>
      </section>

      <section class="settings-card">
        <h3>다운로드 인증</h3>
        <p>Civitai와 Hugging Face 인증 정보를 한 번 저장하면 Krea 체크포인트·LoRA·LTX 모델 다운로드에서 함께 사용합니다. 저장된 값은 화면으로 다시 전송되지 않습니다.</p>
        <div class="fields">
          <label>Civitai API 키
            <input type="password" autocomplete="new-password" bind:value={civitaiToken} placeholder={imageCheckpointStatus?.token_configured ? '저장됨 · 변경할 때만 입력' : 'API key'}>
          </label>
          <label>Hugging Face read 토큰
            <input type="password" autocomplete="new-password" bind:value={hfToken} placeholder={videoModelStatus?.token_configured ? '저장됨 · 변경할 때만 입력' : 'hf_…'}>
          </label>
        </div>
        <button type="button" class="primary" disabled={savingDownloadCredentials || (!civitaiToken.trim() && !hfToken.trim())} onclick={saveDownloadCredentials}>{savingDownloadCredentials ? '저장 중…' : '인증 정보 저장'}</button>
      </section>

      <section class="settings-card">
        <h3>Krea 모델 준비</h3>
        <p>저장된 Civitai 키와 Hugging Face 토큰을 사용해 Identity Edit용 인코더와 Ray·Moody·Turbo Edit 체크포인트를 영구 모델 캐시에 준비합니다.</p>
        {#if imageCheckpointStatus}
          <div class="storage-stats">
            <span><small>상태</small><strong>{imageCheckpointStatus.ready ? '모두 준비됨' : imageCheckpointStatus.preparing ? '다운로드 중' : '준비 필요'}</strong></span>
            <span><small>완료</small><strong>{imageCheckpointStatus.variants?.filter((item) => item.ready).length || 0}/{imageCheckpointStatus.variants?.length || 4}</strong></span>
            <span><small>현재</small><strong>{imageCheckpointStatus.current || '대기'}</strong></span>
          </div>
          {#if imageCheckpointStatus.preparing && imageCheckpointStatus.total_bytes > 0}
            <div class="progress-track"><span style={`width:${Math.min(100, imageCheckpointStatus.downloaded_bytes / imageCheckpointStatus.total_bytes * 100)}%`}></span></div>
            <small>{formatBytes(imageCheckpointStatus.downloaded_bytes)} / {formatBytes(imageCheckpointStatus.total_bytes)}</small>
          {/if}
          {#if imageCheckpointStatus.error}<small class="model-setup-error">{imageCheckpointStatus.error}</small>{/if}
          {#if imageCheckpointStatus.identity_runtime}
            <div class="model-variant-list">
              <label>
                <input type="checkbox" checked disabled>
                <span>Identity · ConvRot INT8<small>{imageCheckpointStatus.identity_runtime.convrot_ready ? '준비됨' : '컨테이너 시작 시 Hugging Face에서 자동 준비'}</small></span>
                <a href={imageCheckpointStatus.identity_runtime.convrot_source} target="_blank" rel="noreferrer">출처</a>
              </label>
              <label>
                <input type="checkbox" checked disabled>
                <span>Identity · Heretic INT8 ConvRot<small>{imageCheckpointStatus.identity_runtime.heretic_ready ? '준비됨' : imageCheckpointStatus.identity_runtime.heretic_downloaded_bytes ? `${formatBytes(imageCheckpointStatus.identity_runtime.heretic_downloaded_bytes)} / ${formatBytes(imageCheckpointStatus.identity_runtime.heretic_size_bytes)}` : formatBytes(imageCheckpointStatus.identity_runtime.heretic_size_bytes)}</small></span>
                <a href={imageCheckpointStatus.identity_runtime.heretic_source} target="_blank" rel="noreferrer">출처</a>
              </label>
            </div>
          {/if}
          <h4>이미지 탭 모델 표시</h4>
          <p>체크한 모델만 이미지 생성의 체크포인트 목록에 표시됩니다. 공식 INT8과 NVFP4는 항상 표시됩니다.</p>
          <div class="model-variant-list checkpoint-visibility-list">
            <label><input type="checkbox" checked disabled><span>Krea 2 Turbo · INT8<small>기본 · 항상 표시</small></span></label>
            <label><input type="checkbox" checked disabled><span>Krea 2 Turbo · NVFP4<small>고속 · 항상 표시</small></span></label>
            {#each checkpointDisplayChoices as choice}
              <label class:unavailable={!displayCheckpointReady(choice[0])}><input type="checkbox" checked={checkpointVisible(choice[0])} onchange={(event) => setCheckpointVisible(choice[0], event.currentTarget.checked)}><span>{choice[1]}<small>{displayCheckpointReady(choice[0]) ? '준비됨' : '모델 준비 필요'}</small></span></label>
            {/each}
          </div>
          <hr>
          <h4>다운로드 대상</h4>
          <p>아래에서 선택한 원본 파일을 `선택 모델 준비` 버튼으로 다운로드합니다.</p>
          <div class="model-variant-list">
            {#each imageCheckpointStatus.variants || [] as variant}
              <label>
                <input type="checkbox" checked={checkpointSelection[variant.id]} onchange={(event) => checkpointSelection = { ...checkpointSelection, [variant.id]: event.currentTarget.checked }}>
                <span>{variant.label}<small>{variant.ready ? '준비됨' : variant.downloaded_bytes ? `${formatBytes(variant.downloaded_bytes)} / ${formatBytes(variant.size_bytes)}` : formatBytes(variant.size_bytes)}</small></span>
                <a href={variant.source} target="_blank" rel="noreferrer">출처</a>
              </label>
            {/each}
          </div>
        {/if}
        <button type="button" class="primary" disabled={preparingImageCheckpoints || imageCheckpointStatus?.preparing || (!civitaiToken.trim() && !imageCheckpointStatus?.token_configured)} onclick={prepareImageCheckpoints}>
          {imageCheckpointStatus?.preparing ? '모델 준비 중…' : imageCheckpointStatus?.ready && imageCheckpointStatus?.identity_runtime?.heretic_ready ? '파일 다시 확인' : '선택 모델 준비'}
        </button>
        {#if imageCheckpointStatus?.nvfp4_conversion}
          {@const conversion = imageCheckpointStatus.nvfp4_conversion}
          <hr>
          <h4>V2·V4 NVFP4 변환</h4>
          <p>BF16 원본을 받은 뒤 GB10 네이티브 NVFP4로 변환하고, 실제 512px 생성을 통과한 파일만 선택 가능하게 만듭니다.</p>
          <div class="model-variant-list">
            {#each conversion.variants || [] as variant}
              <label>
                <input type="checkbox" checked={nvfp4Selection[variant.id]} onchange={(event) => nvfp4Selection = { ...nvfp4Selection, [variant.id]: event.currentTarget.checked }}>
                <span>{variant.id === 'ray-v2' ? 'Ray Artshoot V2' : 'Ray Artshoot V4'}<small>{variant.validated ? `검증 완료 · ${formatBytes(variant.converted_size_bytes)}` : variant.converted_ready ? '변환됨 · 생성 검증 필요' : variant.source_ready ? 'BF16 준비됨 · 변환 대기' : `BF16 ${formatBytes(variant.source_size_bytes)}`}</small></span>
                <a href={variant.source} target="_blank" rel="noreferrer">출처</a>
              </label>
            {/each}
          </div>
          {#if conversion.preparing}
            <div class="progress-track"><span style={`width:${conversion.total ? Math.min(100, conversion.done / conversion.total * 100) : 0}%`}></span></div>
            <small>{conversion.current} · {conversion.stage === 'download' ? 'BF16 다운로드' : conversion.stage === 'unload' ? '메모리 정리' : conversion.stage === 'convert' ? 'NVFP4 변환' : conversion.stage === 'validate' ? '생성 검증' : '준비'} · {conversion.stage === 'download' ? `${formatBytes(conversion.done)} / ${formatBytes(conversion.total)}` : `${conversion.done}/${conversion.total}`}</small>
          {/if}
          {#if conversion.error}<small class="model-setup-error">{conversion.error}</small>{/if}
          <label class="inline-check"><input type="checkbox" bind:checked={removeBF16Sources}> <span>검증 성공 후 BF16 원본 삭제</span></label>
          <button type="button" class="primary" disabled={convertingImageCheckpoints || conversion.preparing || imageCheckpointStatus?.preparing || (!civitaiToken.trim() && !imageCheckpointStatus?.token_configured)} onclick={convertImageCheckpointsNVFP4}>
            {conversion.preparing ? 'NVFP4 준비 중…' : '선택 모델 NVFP4 준비'}
          </button>
          <small>변환 프로필: <a href={conversion.profile_source} target="_blank" rel="noreferrer">출처</a> · 커밋 {conversion.profile_commit?.slice(0, 8)}</small>
        {/if}
      </section>

      <section class="settings-card">
        <h3>LTX 영상 모델 준비</h3>
        <p>저장된 Hugging Face 토큰을 사용해 일반 영상 모델과 공식 A2V dev·distilled LoRA, 공개 Motion LoRA를 준비합니다.</p>
        {#if videoModelStatus}
          <div class="storage-stats">
            <span><small>상태</small><strong>{videoModelStatus.ready ? '준비 완료' : videoModelStatus.preparing ? '다운로드 중' : '준비 필요'}</strong></span>
            <span><small>파일</small><strong>{videoModelStatus.ready_files}/{videoModelStatus.required_files}</strong></span>
            <span><small>Motion LoRA</small><strong>{videoModelStatus.motion_lora_ready ? '준비됨' : '대기'}</strong></span>
            <span><small>A2V</small><strong>{videoModelStatus.a2v_ready ? '준비됨' : videoModelStatus.preparing ? '다운로드 중' : '추가 준비'}</strong></span>
          </div>
          {#if videoModelStatus.error}<small class="model-setup-error">{videoModelStatus.error}</small>{/if}
        {/if}
        <small><a href="https://huggingface.co/Lightricks/LTX-2.5" target="_blank" rel="noreferrer">LTX-2.5 라이선스 동의 ↗</a> 후 다운로드 인증에 같은 계정의 read 토큰을 저장하세요.</small>
        <button type="button" class="primary" disabled={preparingVideoModels || videoModelStatus?.preparing || (!hfToken.trim() && !videoModelStatus?.token_configured && !videoModelStatus?.ready)} onclick={prepareVideoModels}>
          {videoModelStatus?.preparing ? '모델 준비 중…' : videoModelStatus?.ready && !videoModelStatus?.a2v_ready ? 'A2V 모델 준비' : videoModelStatus?.ready ? '파일 다시 확인' : '모델 준비 시작'}
        </button>
      </section>
      </div>
      {/if}

      {#if settingsSection === 'storage'}
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
      {/if}

      <div class="settings-grid">
        {#if settingsSection === 'defaults'}
        <section class="settings-card">
          <h3>이미지</h3>
          <label>기본 체크포인트<select bind:value={settings.image.default_checkpoint}><option value="official-int8">Krea 2 Turbo · INT8 · 기본</option><option value="official">Krea 2 Turbo · NVFP4 · 고속</option><option value="chriscole-edit-v1.1" disabled={!imageCheckpointStatus?.variants?.find((item) => item.id === 'chriscole-edit-v1.1')?.ready}>Krea 2 Turbo Edit v1.1 · FP8</option><option value="moody-v7" disabled={!imageCheckpointStatus?.variants?.find((item) => item.id === 'moody-v7')?.ready}>Moody Krea 2 Mix V7 · NVFP4</option><option value="moody-cutie-v4" disabled={!imageCheckpointStatus?.variants?.find((item) => item.id === 'moody-cutie-v4')?.ready}>Moody Cutie Mix V4 · NVFP4</option><option value="moody-amateur-v1" disabled={!imageCheckpointStatus?.variants?.find((item) => item.id === 'moody-amateur-v1')?.ready}>Moody Amateur Mix V1 · NVFP4</option><option value="ray-v1" disabled={!imageCheckpointStatus?.variants?.find((item) => item.id === 'ray-v1')?.ready}>Ray Artshoot V1 · FP8</option><option value="ray-v2" disabled={!imageCheckpointStatus?.variants?.find((item) => item.id === 'ray-v2')?.ready}>Ray Artshoot V2 · FP8</option><option value="ray-v2-nvfp4" disabled={!imageCheckpointStatus?.nvfp4_conversion?.variants?.find((item) => item.id === 'ray-v2')?.validated}>Ray Artshoot V2 · NVFP4</option><option value="ray-v3" disabled={!imageCheckpointStatus?.variants?.find((item) => item.id === 'ray-v3')?.ready}>Ray Artshoot V3 · INT8</option><option value="ray-v4" disabled={!imageCheckpointStatus?.variants?.find((item) => item.id === 'ray-v4')?.ready}>Ray Artshoot V4 · INT8</option><option value="ray-v4-nvfp4" disabled={!imageCheckpointStatus?.nvfp4_conversion?.variants?.find((item) => item.id === 'ray-v4')?.validated}>Ray Artshoot V4 · NVFP4</option></select><small>INT8은 품질·LoRA 충실도, NVFP4는 속도·메모리를 우선합니다.</small></label>
          {#each imageModeChoices as mode}
            <div class="backend-setting"><strong>{imageModeMeta[mode].label}</strong><label>Endpoint<input type="url" bind:value={settings.image.backends[mode].endpoint} required></label><label>모델<input bind:value={settings.image.backends[mode].model} required></label></div>
          {/each}
          <div class="fields three">
            <label>기본 너비<input type="number" min="256" max="2048" step="any" bind:value={settings.image.default_width} onchange={() => settings.image.default_width = snapDimension(settings.image.default_width, 8, 256, 2048)}></label>
            <label>기본 높이<input type="number" min="256" max="2048" step="any" bind:value={settings.image.default_height} onchange={() => settings.image.default_height = snapDimension(settings.image.default_height, 8, 256, 2048)}></label>
            <label>참조 이미지 수<input type="number" min="1" max="16" bind:value={settings.image.max_reference_images}></label>
          </div>
        </section>

        <section class="settings-card">
          <h3>영상</h3>
          <label>모델<input bind:value={settings.video.model} required></label>
          <div class="fields">
            <label>기본 너비<input type="number" min="256" step="any" bind:value={settings.video.default_width} onchange={() => settings.video.default_width = snapDimension(settings.video.default_width, 64, 256, 1920)}></label>
            <label>기본 높이<input type="number" min="256" step="any" bind:value={settings.video.default_height} onchange={() => settings.video.default_height = snapDimension(settings.video.default_height, 64, 256, 1920)}></label>
            <label class="duration-field"><span>기본 길이 (초) <small>{framesForDuration(settingsVideoDurationSeconds, settings.video.default_fps)} 프레임 · 8k+1</small></span><input aria-label="기본 영상 길이 초" type="number" min={8 / Number(settings.video.default_fps || 24)} step="any" bind:value={settingsVideoDurationSeconds} onchange={() => settingsVideoDurationSeconds = snapVideoDuration(settingsVideoDurationSeconds, settings.video.default_fps)}></label>
            <label>기본 FPS<input type="number" min="1" max="60" step="any" bind:value={settings.video.default_fps} onchange={() => settingsVideoDurationSeconds = snapVideoDuration(settingsVideoDurationSeconds, settings.video.default_fps)}></label>
            <label>Motion LoRA 기본값<select bind:value={settings.video.default_motion_lora_enabled}><option value={false}>꺼짐</option><option value={true}>켜짐</option></select></label>
            <label>Motion LoRA 강도<input type="number" min="0" max="1" step="any" disabled={!settings.video.default_motion_lora_enabled} bind:value={settings.video.default_motion_lora_strength}><small>권장 0.35~0.70 · 제안 0.50</small></label>
            <label>고해상도 가속<select bind:value={settings.video.acceleration}><option value="auto">자동</option><option value="dense">끄기</option></select><small>자동은 큰 영상에서만 DGX Spark용 SOL Attention 가속을 사용합니다.</small></label>
          </div>
          <small>설정을 저장하면 이후 영상 작업부터 즉시 적용됩니다. 모델 전환이 필요한 첫 작업에서만 파이프라인을 자동으로 다시 적재합니다.</small>
        </section>

        <section class="settings-card">
          <h3>프롬프트</h3>
          <label>향상 모델<input bind:value={settings.prompt_enhancement.model} required></label>
          <div class="fields">
            <label>프롬프트 향상 기본값<select bind:value={settings.prompt_enhancement.default_enabled}><option value={true}>켜짐</option><option value={false}>꺼짐</option></select></label>
            <label>프롬프트 준수 강화 기본값<select bind:value={settings.image.default_prompt_enhancer}><option value={true}>켜짐</option><option value={false}>꺼짐</option></select></label>
            <label>최대 토큰<input type="number" min="64" max="2048" bind:value={settings.prompt_enhancement.max_tokens}></label>
            <label>이미지 인식<select bind:value={settings.prompt_enhancement.vision_enabled}><option value={false}>꺼짐</option><option value={true}>켜짐</option></select></label>
          </div>
          <small>프롬프트 향상은 이미지와 영상에 함께 적용됩니다. 준수 강화는 Krea 2 이미지 생성의 Krea2T 기본값입니다.</small>
          <small>I2V 시작 이미지는 LTX에 직접 전달하므로 이미지 인식 기반 프롬프트 향상은 기본적으로 꺼짐을 유지하세요.</small>
        </section>
        {/if}

        {#if settingsSection === 'metadata'}
        <section class="settings-card">
          <h3>이미지 EXIF 제작자 정보</h3>
          <p>비워둔 항목은 새 이미지의 EXIF에서 생략됩니다.</p>
          <div class="fields">
            <label>제작자 이름<input maxlength="256" bind:value={settings.image_metadata.creator} placeholder="이름 또는 스튜디오명"></label>
            <label>저작권 문구<input maxlength="512" bind:value={settings.image_metadata.copyright} placeholder="© 2026 이름. All rights reserved."></label>
            <label>웹사이트·연락처<input maxlength="2048" bind:value={settings.image_metadata.website} placeholder="https://… 또는 이메일"></label>
          </div>
          <label>메모<textarea rows="3" maxlength="2000" bind:value={settings.image_metadata.note} placeholder="작품이나 제작자에 관한 짧은 안내"></textarea></label>
        </section>
        {/if}

        {#if settingsSection === 'defaults'}
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
            <label>기본 언어<select bind:value={settings.recognition.default_language}>{#each recognitionLanguages as option}<option value={option[0]}>{option[1]}</option>{/each}</select></label>
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
        {/if}
      </div>
      <div class="settings-save-bar"><small>변경 내용은 모든 설정 구역에 함께 저장됩니다.</small><button class="primary settings-save" disabled={busy}>{busy ? '저장 중…' : '설정 저장'}</button></div>
    </form>
