<script>
  import { onDestroy } from 'svelte'
  import PromptComposer from '../PromptComposer.svelte'
  import { lockModalScroll } from '../modalScroll.js'
  import {
    identityPreserveCatalog,
    imageAspectRatios,
    imageModeMeta,
    kreaPromptGuideSource,
    kreaStyleCatalog,
  } from '../lib/catalogs.js'
  import { snapDimension } from '../lib/videoTiming.js'

  export let activeJobs
  export let activeKreaModuleLabels
  export let addIdentityReferences
  export let addKreaRefs
  export let addRefs
  export let applyIdentityPreset
  export let applySmartResolution
  export let busy
  export let cannyEditorOpen
  export let checkpointVisible
  export let config
  export let disableAllKreaModules
  export let enhanceImagePrompt
  export let enhancingPrompt
  export let featureModulesOpen
  export let filterModeDefault
  export let filterModeMaximum
  export let filterPromptPreset
  export let generateImage
  export let hasKreaStyle
  export let hasUserLora
  export let identityPreserveCustom
  export let identityPreserveItems
  export let identityPreset
  export let identityUI
  export let imageAspectRatio
  export let imageCheckpointStatus
  export let imageCloneMessage
  export let clearImageCloneMessage = () => {}
  export let imageDisabledMessage
  export let imageEnhanceEnabled
  export let imageEnhancedPrompt
  export let imageEnhancementIsActive
  export let imageEnhancementIsCurrent
  export let imageForm
  export let imageMegapixels
  export let imageResolutionMode
  export let isPureOutpaint
  export let kreaAnyPaintImage
  export let kreaAnyPaintMask
  export let kreaAnyPaintMaskPreview
  export let kreaAnyPaintPreview
  export let kreaDepthImage
  export let kreaDepthPreview
  export let kreaIdentityImage
  export let kreaIdentityMask
  export let kreaIdentityMaskPreview
  export let kreaIdentityPreview
  export let kreaIdentityReferences
  export let kreaModuleMessage
  export let kreaModules
  export let kreaNK2EImage
  export let kreaNK2EPreprocessed
  export let kreaNK2EPreview
  export let kreaOptions
  export let kreaStrictMask
  export let kreaStrictMaskPreview
  export let kreaStyleLabel
  export let kreaStyleReferenceImages
  export let kreaStyleSelections
  export let kreaVisionImages
  export let looksLikeStructuredPrompt
  export let maskEditorMode
  export let openGarmentExtractor
  export let openImageSequence
  export let parentImageJobID
  export let presetImagePickerTarget
  export let openPromptExamples = () => {}
  export let rawImagePrompt
  export let recentImagePickerTarget
  export let refreshUserLoras
  export let refs
  export let remoteImageTarget
  export let removeIdentityReference
  export let removeKreaRef
  export let removeRef
  export let resetImageCreation
  export let resetImageEnhancement
  export let runtimeInfoOpen
  export let selectKreaCheckpoint
  export let selectedKreaCheckpoint
  export let selectedKreaCheckpointSource
  export let setKreaImage
  export let showImage
  export let showImageOnKey
  export let toggleIdentityPreserveItem
  export let toggleKreaModule
  export let toggleKreaStyle
  export let toggleUserLora
  export let updateKreaStyleStrength
  export let updateUserLoraStrength
  export let useCustomImageResolution
  export let userLoraCatalog
  export let userLoraLabel
  export let userLoraSelections

  let releaseFeatureModulesScroll = null
  $: {
    if (featureModulesOpen && !releaseFeatureModulesScroll) releaseFeatureModulesScroll = lockModalScroll()
    else if (!featureModulesOpen && releaseFeatureModulesScroll) {
      releaseFeatureModulesScroll()
      releaseFeatureModulesScroll = null
    }
  }
  onDestroy(() => releaseFeatureModulesScroll?.())
</script>

<form class="image-create-pane" onsubmit={(e) => { e.preventDefault(); generateImage() }}>
  <div class="section-title"><div><span>01</span><h2>이미지 생성</h2></div><div class="image-title-actions">{#if imageForm.mode === 'create'}<button type="button" class="quiet header-prompt-tool" onclick={() => openPromptExamples('image')}>예제{#if filterPromptPreset}<b>선택됨</b>{/if}</button><PromptComposer compact storageKey="spark-media-prompt-composer-image" activeStyles={kreaModules.style ? kreaStyleSelections.map((style) => style.name) : []} onApply={(prompt, mode) => { const currentPrompt = imageForm.prompt.trimEnd(); imageForm.prompt = mode === 'append' && currentPrompt ? `${currentPrompt}\n${prompt}` : prompt; filterPromptPreset = ''; resetImageEnhancement() }} />{/if}<a class="quiet portrait-lab-open" href="/tools/portrait-lab/" target="_blank" rel="noreferrer">P Lab ↗</a><button type="button" class="quiet image-create-reset" disabled={busy} title="프롬프트와 이미지 생성 설정을 모두 비웁니다." onclick={resetImageCreation}>초기화</button></div></div>
  {#if imageCloneMessage}<div class="clone-notice"><span>{imageCloneMessage}</span><button type="button" aria-label="불러오기 안내 닫기" onclick={clearImageCloneMessage}>×</button></div>{/if}
  {#if imageForm.mode === 'create'}
    <div class="prompt-tools-row">
      <button type="button" class="prompt-tool-open sequence-tool-open" disabled={busy} onclick={openImageSequence}><span>연속 생성</span></button>
      <button type="button" class="prompt-tool-open feature-tool-open" class:has-warning={Boolean(kreaModuleMessage)} aria-haspopup="dialog" onclick={() => featureModulesOpen = true}><span>기능 모듈</span>{#if activeKreaModuleLabels.length}<b>{activeKreaModuleLabels.length}개</b>{/if}</button>
      <button type="button" class="prompt-tool-open garment-tool-open" aria-haspopup="dialog" onclick={() => openGarmentExtractor()}><span>의상 추출</span></button>
    </div>
    {#if kreaModuleMessage}<small class="feature-module-toolbar-warning">{kreaModuleMessage}</small>{/if}
  {/if}
  <label>{kreaModules.identity ? '변경할 내용' : '프롬프트'}<textarea bind:value={imageForm.prompt} rows="7" placeholder="{kreaModules.identity ? '원본에서 바꿀 내용만 구체적으로 입력하세요.' : isPureOutpaint() ? '선택 사항 · 비워두면 원본을 자연스럽게 이어서 확장합니다.' : '만들고 싶은 장면을 입력하세요.'}"></textarea></label>
  {#if kreaModules.identity}
    <div class="identity-preserve-control">
      <div><strong>유지할 내용</strong><small>켜진 항목은 보존하고, 꺼진 항목은 변경을 허용합니다.{kreaModules.depth ? ' Depth 사용 중에는 자세·구도를 보존하지 않습니다.' : ''}</small></div>
      <div class="identity-preserve-chips">
        {#each identityPreserveCatalog as item}
          <button type="button" class:active={identityPreserveItems.includes(item.id)} disabled={kreaModules.depth && (item.id === 'pose' || item.id === 'composition')} onclick={() => toggleIdentityPreserveItem(item.id)}>{item.label}</button>
        {/each}
      </div>
      <label>추가 유지 조건<input bind:value={identityPreserveCustom} oninput={resetImageEnhancement} placeholder="예: 목걸이와 원본의 한글 문구"></label>
    </div>
  {/if}
  <div class="enhanced-prompt image-enhancer-panel" class:inactive={!imageEnhancementIsActive}>
    <div class="image-enhancer-panel-header">
      <div class="enhancer-panel-title"><strong title="연결된 Gemma 4 12B 모델이 Krea 2용 영어 프롬프트로 정리·확장합니다.">프롬프트 향상</strong><a href={kreaPromptGuideSource} target="_blank" rel="noreferrer">출처 ↗</a></div>
      <div class="enhancer-panel-actions">
        <button type="button" class="quiet enhancer-run" disabled={!imageEnhancementIsActive || enhancingPrompt || !rawImagePrompt().trim()} onclick={enhanceImagePrompt}>{enhancingPrompt ? '처리 중…' : imageEnhancementIsCurrent ? '다시 처리' : '프롬프트 향상'}</button>
      <div class="segmented compact">
        <button type="button" class:active={imageEnhanceEnabled} onclick={() => imageEnhanceEnabled = true}>켜짐</button>
        <button type="button" class:active={!imageEnhanceEnabled} onclick={() => imageEnhanceEnabled = false}>꺼짐</button>
      </div>
      </div>
    </div>
    {#if imageEnhancedPrompt.trim()}
      <textarea bind:value={imageEnhancedPrompt} rows="5" aria-label="Krea 향상 프롬프트"></textarea>
      <small>{looksLikeStructuredPrompt() ? 'JSON 형식은 원문을 유지합니다.' : imageEnhancementIsActive ? '실제 생성에 사용할 문장입니다. 확인하고 직접 수정할 수 있습니다.' : '꺼짐 · 기존 결과는 보존되며 실제 생성에는 원문을 사용합니다.'}</small>
    {:else}
      <small>{imageEnhancementIsActive ? '프롬프트 향상을 누르면 결과를 확인하고 직접 수정할 수 있습니다.' : '꺼짐 · 실제 생성에는 원문을 사용합니다.'}</small>
    {/if}
  </div>
  {#if imageForm.mode === 'create'}
    <section class="krea-runtime-controls" aria-label="Krea 모델 내부 조정">
      <div class="runtime-control-heading"><div><strong>모델 내부 조정</strong><small>필터 벡터와 텍스트 조건 강도를 간단히 조절합니다.</small></div><button type="button" class="runtime-info-button" aria-label="모델 내부 조정 설명" title="설명 보기" onclick={() => runtimeInfoOpen = true}>i</button></div>
      <div class="runtime-control-row">
        <label><span>필터 완화</span><select disabled={kreaOptions.checkpoint !== 'official'} value={kreaOptions.filter_mode} onchange={(event) => { const mode = event.currentTarget.value; kreaOptions = { ...kreaOptions, filter_mode: mode, filter_strength: filterModeDefault(mode) } }}><option value="off">{kreaOptions.checkpoint === 'official' ? '꺼짐 · 원본' : '체크포인트에 내장됨'}</option><option value="adherence">준수 강화 · skc3vo</option><option value="balanced">균형 · 2-vector</option><option value="strong">강함 · 3-vector</option></select></label>
        <label><span>완화 강도</span><input type="range" min="0" max={filterModeMaximum(kreaOptions.filter_mode)} step="0.01" disabled={kreaOptions.filter_mode === 'off'} bind:value={kreaOptions.filter_strength}><b>{Number(kreaOptions.filter_strength).toFixed(2)}</b></label>
      </div>
      <div class="runtime-control-row adherence">
        <div><strong>프롬프트 준수 강화</strong><small>Krea2T Enhancer · 객체 수와 배치 같은 복잡한 지시를 더 강하게 반영</small></div>
        <div class="segmented compact"><button type="button" class:active={kreaOptions.prompt_enhancer} onclick={() => kreaOptions = { ...kreaOptions, prompt_enhancer: true }}>켜짐</button><button type="button" class:active={!kreaOptions.prompt_enhancer} onclick={() => kreaOptions = { ...kreaOptions, prompt_enhancer: false }}>꺼짐</button></div>
      </div>
      {#if kreaOptions.prompt_enhancer}<div class="runtime-control-row"><label><span>강화 강도</span><input type="range" min="0" max="2" step="0.05" bind:value={kreaOptions.prompt_enhancer_strength}><b>{Number(kreaOptions.prompt_enhancer_strength).toFixed(2)}</b></label><label><span>텍스트 비중</span><input type="range" min="0.25" max="4" step="0.05" bind:value={kreaOptions.prompt_text_scale}><b>{Number(kreaOptions.prompt_text_scale).toFixed(2)}</b></label></div>{/if}
    </section>
  {/if}
  {#if imageForm.mode === 'create'}
    {#if featureModulesOpen}
      <div class="feature-module-backdrop" role="presentation" onclick={(event) => { if (event.target === event.currentTarget) featureModulesOpen = false }}>
        <div class="feature-module-modal" role="dialog" aria-modal="true" aria-label="기능 모듈">
          <header>
            <div><strong>기능 모듈</strong><small>필요한 기능만 켜면 내부 연결은 자동으로 구성됩니다. 변경 내용은 즉시 유지됩니다.</small></div>
            <button type="button" aria-label="닫기" onclick={() => featureModulesOpen = false}>×</button>
          </header>
          <div class="feature-module-content">
            {#if kreaModuleMessage}<div class="feature-module-warning">{kreaModuleMessage}</div>{/if}
            <section class="module-panel" aria-label="Krea 생성 모듈">
      <article class="module-card" class:enabled={kreaModules.identity}>
        <button type="button" class="module-toggle" aria-pressed={kreaModules.identity} onclick={() => toggleKreaModule('identity')}>
          <span class="module-icon">REF</span><span><strong>원본 수정</strong><small>Identity Edit · 원본의 인물이나 장면을 유지하면서 원하는 부분 변경</small></span><i></i>
        </button>
        {#if kreaModules.identity}
          <div class="module-body">
            {#if parentImageJobID}<div class="clone-notice"><span>결과 작업 {parentImageJobID.slice(0, 8)}에서 계속 편집 중</span><button type="button" onclick={() => parentImageJobID = ''}>×</button></div>{/if}
            <label>무엇을 할까요?<select value={identityPreset} onchange={(event) => applyIdentityPreset(event.currentTarget.value)}><option value="">직접 지시</option><option value="restage">같은 인물로 장면 변경</option><option value="sheet">2×2 캐릭터 시트</option><option value="faceSwap">얼굴 교체</option><option value="headSwap">머리 전체 교체</option><option value="personSwap">인물 교체</option><option value="tryon">의상 교체</option><option value="replace">선택 영역 교체</option></select></label>
            <div class="module-source-field"><label class="module-file">{identityUI.primary}<small>{identityUI.primaryHint}</small><input type="file" accept="image/*" onchange={(e) => setKreaImage('identity', e.currentTarget.files?.[0] || null)}><span class="module-file-display">{#if kreaIdentityPreview}<img role="button" tabindex="0" class="zoomable-source" src={kreaIdentityPreview} alt={`${identityUI.primary} 미리보기`} title="클릭하여 크게 보기" onclick={(event) => showImage(event, kreaIdentityPreview, identityUI.primary)} onkeydown={(event) => showImageOnKey(event, kreaIdentityPreview, identityUI.primary)}>{:else}<i>REF</i>{/if}<b title={kreaIdentityImage?.name || identityUI.primaryHint}>{kreaIdentityImage?.name || identityUI.primaryHint}</b></span></label><div class="module-source-actions"><button type="button" class="recent-result-open" onclick={() => recentImagePickerTarget = 'identity'}>최근 결과</button><button type="button" class="recent-result-open" onclick={() => presetImagePickerTarget = 'identity'}>프리셋</button><button type="button" class="recent-result-open" onclick={() => remoteImageTarget = 'identity'}>URL</button></div></div>
            {#if identityUI.showSecondary}
              <div class="module-source-field"><label class="module-file" class:optional={!identityUI.secondaryRequired}>{identityUI.secondary} · 최대 3장<small>{identityUI.secondaryHint}{identityUI.secondaryRequired ? ' · 1장 이상 필수' : ' · 선택 사항'} · 의상·포즈·소품을 함께 선택 가능</small><input type="file" accept="image/*" multiple onchange={(e) => addIdentityReferences(e.currentTarget.files)}><span class="module-file-display"><i>+REF</i><b>{kreaIdentityReferences.length ? `${kreaIdentityReferences.length}장 선택됨` : identityUI.secondaryHint}</b></span></label><div class="module-source-actions"><button type="button" class="recent-result-open" onclick={() => recentImagePickerTarget = 'identityReference'}>최근 결과</button><button type="button" class="recent-result-open" onclick={() => presetImagePickerTarget = 'identityReference'}>프리셋</button><button type="button" class="recent-result-open" onclick={() => remoteImageTarget = 'identityReference'}>URL</button></div></div>
              {#if kreaIdentityReferences.length}<div class="reference-previews identity-reference-previews">{#each kreaIdentityReferences as image, i}<div><button type="button" class="reference-preview-open" onclick={(event) => showImage(event, image.preview || image.url, `${identityUI.secondary} ${i + 1}`)}><img src={image.preview || image.url} alt={`${identityUI.secondary} ${i + 1}`}><span class="reference-preview-index">{i + 1}</span></button><button type="button" class="reference-preview-remove" aria-label={`${identityUI.secondary} ${i + 1} 제거`} onclick={() => removeIdentityReference(i)}>×</button></div>{/each}</div>{/if}
            {/if}
            <p class="identity-prompt-guide">{identityUI.guide}</p>
            <details class="module-advanced">
              <summary><span>고급 설정</span><small>닮음·참조 해석·마스크</small></summary>
              <div class="module-advanced-body">
                <div class="module-controls">
                  <label><span>편집 LoRA 강도 <b>{Number(kreaOptions.identity_strength).toFixed(2)}</b></span><input type="range" min="0" max="2" step="0.05" bind:value={kreaOptions.identity_strength}></label>
                  <label><span>보조 참조 강도 <b>{kreaOptions.ref_boost}</b></span><input type="range" min="0" max="10" step="0.5" bind:value={kreaOptions.ref_boost}></label>
                  <label><span>원본 유지 강도 <b>{kreaOptions.source_ref_boost}</b></span><input type="range" min="0" max="10" step="0.5" bind:value={kreaOptions.source_ref_boost}></label>
                  <label>참조 해석<select bind:value={kreaOptions.grounding_px}><option value={512}>변경 우선</option><option value={768}>균형</option><option value={1024}>얼굴 우선</option></select></label>
                </div>
                <div class="module-controls"><label>참조 맞춤<select bind:value={kreaOptions.identity_fit_mode}><option value="fit">전체 보존 · Fit</option><option value="crop">얼굴 확대 · Crop</option></select></label><label>VAE<select bind:value={kreaOptions.vae_mode}><option value="default">Qwen VAE</option><option value="wan">Wan 2.1 VAE · 권장</option><option value="real">Real VAE · 실험</option></select></label></div>
                <div class="module-controls">
                  <label class="module-file optional">닮음 집중 마스크 <small>흰 영역의 Identity 주의만 높임</small><input type="file" accept="image/*" onchange={(e) => setKreaImage('identityMask', e.currentTarget.files?.[0] || null)}><span class="module-file-display">{#if kreaIdentityMaskPreview}<img src={kreaIdentityMaskPreview} alt="닮음 집중 마스크">{:else}<i>FOCUS</i>{/if}<b>{kreaIdentityMask?.name || '선택 사항'}</b></span></label>
                  <button type="button" class="mask-editor-open" disabled={!kreaIdentityPreview} onclick={() => maskEditorMode = 'identity'}>얼굴·특징 집중 영역 칠하기</button>
                </div>
                <div class="module-controls">
                  <label class="module-file optional">변경 허용 마스크 <small>흰 영역 밖 픽셀을 원본 그대로 보존</small><input type="file" accept="image/*" onchange={(e) => setKreaImage('strictMask', e.currentTarget.files?.[0] || null)}><span class="module-file-display">{#if kreaStrictMaskPreview}<img src={kreaStrictMaskPreview} alt="변경 허용 마스크">{:else}<i>LOCK</i>{/if}<b>{kreaStrictMask?.name || '선택 사항'}</b></span></label>
                  <button type="button" class="mask-editor-open" disabled={!kreaIdentityPreview} onclick={() => maskEditorMode = 'strict'}>변경 허용 영역 칠하기</button>
                </div>
                {#if kreaStrictMask}<div class="module-controls"><label>마스크 확장<input type="number" min="0" max="128" bind:value={kreaOptions.strict_mask_grow}></label><label>경계 부드럽게<input type="number" min="0" max="128" step="any" bind:value={kreaOptions.strict_mask_feather}></label></div>{/if}
              </div>
            </details>
          </div>
        {/if}
      </article>
      <article class="module-card" class:enabled={kreaModules.depth}>
        <button type="button" class="module-toggle" aria-pressed={kreaModules.depth} onclick={() => toggleKreaModule('depth')}>
          <span class="module-icon">3D</span><span><strong>자세·구도</strong><small>Depth Control · 다른 이미지의 공간과 동작 반영</small></span><i></i>
        </button>
        {#if kreaModules.depth}
          <div class="module-body">
            <div class="module-source-field depth-source-field"><label class="module-file">구도 참조 이미지 <input type="file" accept="image/*" onchange={(e) => setKreaImage('depth', e.currentTarget.files?.[0] || null)}><span class="module-file-display">{#if kreaDepthPreview}<img role="button" tabindex="0" class="zoomable-source" src={kreaDepthPreview} alt="구도 참조 미리보기" title="클릭하여 크게 보기" onclick={(event) => showImage(event, kreaDepthPreview, 'Depth 구도 참조')} onkeydown={(event) => showImageOnKey(event, kreaDepthPreview, 'Depth 구도 참조')}>{:else}<i>3D</i>{/if}<b title={kreaDepthImage?.name || '원하는 자세와 구도의 이미지 선택'}>{kreaDepthImage?.name || '원하는 자세와 구도의 이미지 선택'}</b></span></label><div class="module-source-actions"><button type="button" class="recent-result-open" onclick={() => recentImagePickerTarget = 'depth'}>최근 결과</button><button type="button" class="recent-result-open" onclick={() => presetImagePickerTarget = 'depth'}>프리셋</button><button type="button" class="recent-result-open" onclick={() => remoteImageTarget = 'depth'}>URL</button></div></div>
            <label class="module-slider"><span>구도 고정 강도 <b>{Number(kreaOptions.depth_strength).toFixed(1)}</b></span><input type="range" min="0" max="2" step="0.1" bind:value={kreaOptions.depth_strength}></label>
          </div>
        {/if}
      </article>
      <article class="module-card" class:enabled={kreaModules.nk2e}>
        <button type="button" class="module-toggle" aria-pressed={kreaModules.nk2e} onclick={() => toggleKreaModule('nk2e')}>
          <span class="module-icon">N2</span><span><strong>실험 편집·윤곽</strong><small>NK2E v0.3 · 국소 변경 또는 Canny 자세 반영</small></span><i></i>
        </button>
        {#if kreaModules.nk2e}
          <div class="module-body">
            <div class="module-source-field"><label class="module-file">참조 이미지 <input type="file" accept="image/*" onchange={(e) => setKreaImage('nk2e', e.currentTarget.files?.[0] || null)}><span class="module-file-display">{#if kreaNK2EPreview}<img role="button" tabindex="0" class="zoomable-source" src={kreaNK2EPreview} alt="NK2E 참조 미리보기" title="클릭하여 크게 보기" onclick={(event) => showImage(event, kreaNK2EPreview, 'NK2E 참조')} onkeydown={(event) => showImageOnKey(event, kreaNK2EPreview, 'NK2E 참조')}>{:else}<i>N2</i>{/if}<b title={kreaNK2EImage?.name || '편집하거나 윤곽을 가져올 이미지 선택'}>{kreaNK2EImage?.name || '편집하거나 윤곽을 가져올 이미지 선택'}</b></span></label><div class="module-source-actions"><button type="button" class="recent-result-open" onclick={() => recentImagePickerTarget = 'nk2e'}>최근 결과</button><button type="button" class="recent-result-open" onclick={() => presetImagePickerTarget = 'nk2e'}>프리셋</button><button type="button" class="recent-result-open" onclick={() => remoteImageTarget = 'nk2e'}>URL</button></div></div>
            <div class="module-controls">
              <label>작업 방식<select bind:value={kreaOptions.nk2e_mode}><option value="edit">국소 편집</option><option value="canny">윤곽·자세 반영</option></select></label>
              <label class="module-slider"><span>반영 강도 <b>{Number(kreaOptions.nk2e_strength).toFixed(1)}</b></span><input type="range" min="0" max="2" step="0.1" bind:value={kreaOptions.nk2e_strength}></label>
            </div>
            {#if kreaOptions.nk2e_mode === 'canny'}<button type="button" class="mask-editor-open" disabled={!kreaNK2EPreview} onclick={() => cannyEditorOpen = true}>{kreaNK2EPreprocessed ? '완성된 윤곽맵 다시 편집' : 'Canny 미리보기·편집'}</button>{/if}
            <small class="module-caution">실험 기능입니다. 짧고 구체적인 변경 지시가 안정적이며, 현재 다른 Krea 모듈과는 함께 실행하지 않습니다.</small>
          </div>
        {/if}
      </article>
      <article class="module-card" class:enabled={kreaModules.anypaint}>
        <button type="button" class="module-toggle" aria-pressed={kreaModules.anypaint} onclick={() => toggleKreaModule('anypaint')}>
          <span class="module-icon">PAINT</span><span><strong>부분 수정·확장</strong><small>AnyPaint · 선택 영역 수정 또는 캔버스 바깥 생성</small></span><i></i>
        </button>
        {#if kreaModules.anypaint}
          <div class="module-body">
            <div class="module-controls">
              <div class="module-source-field"><label class="module-file">원본 이미지 <input type="file" accept="image/*" onchange={(e) => setKreaImage('anypaint', e.currentTarget.files?.[0] || null)}><span class="module-file-display">{#if kreaAnyPaintPreview}<img role="button" tabindex="0" class="zoomable-source" src={kreaAnyPaintPreview} alt="부분 수정 원본 미리보기" title="클릭하여 크게 보기" onclick={(event) => showImage(event, kreaAnyPaintPreview, '부분 수정·확장 원본')} onkeydown={(event) => showImageOnKey(event, kreaAnyPaintPreview, '부분 수정·확장 원본')}>{:else}<i>IMG</i>{/if}<b title={kreaAnyPaintImage?.name || '수정하거나 확장할 이미지 선택'}>{kreaAnyPaintImage?.name || '수정하거나 확장할 이미지 선택'}</b></span></label><div class="module-source-actions"><button type="button" class="recent-result-open" onclick={() => recentImagePickerTarget = 'anypaint'}>최근 결과</button><button type="button" class="recent-result-open" onclick={() => presetImagePickerTarget = 'anypaint'}>프리셋</button><button type="button" class="recent-result-open" onclick={() => remoteImageTarget = 'anypaint'}>URL</button></div></div>
              <label class="module-file optional">수정 마스크 <small>선택 사항 · 흰 영역을 새로 생성</small><input type="file" accept="image/*" onchange={(e) => setKreaImage('anypaintMask', e.currentTarget.files?.[0] || null)}><span class="module-file-display">{#if kreaAnyPaintMaskPreview}<img role="button" tabindex="0" class="zoomable-source" src={kreaAnyPaintMaskPreview} alt="부분 수정 마스크 미리보기" title="클릭하여 크게 보기" onclick={(event) => showImage(event, kreaAnyPaintMaskPreview, '수정 마스크')} onkeydown={(event) => showImageOnKey(event, kreaAnyPaintMaskPreview, '수정 마스크')}>{:else}<i>MASK</i>{/if}<b title={kreaAnyPaintMask?.name || '확장만 할 때는 비워두기'}>{kreaAnyPaintMask?.name || '확장만 할 때는 비워두기'}</b></span></label>
            </div>
            <button type="button" class="mask-editor-open" disabled={!kreaAnyPaintPreview} onclick={() => maskEditorMode = 'anypaint'}>원본 위에서 수정 영역 칠하기</button>
            <div class="outpaint-controls">
              <strong>이미지 확장</strong><small>원본 크기에 선택한 픽셀만큼 더합니다.</small>
              <div>
                <label>왼쪽<select bind:value={kreaOptions.outpaint_left}><option value={0}>없음</option><option value={64}>64px</option><option value={128}>128px</option><option value={256}>256px</option><option value={384}>384px</option><option value={512}>512px</option></select></label>
                <label>위쪽<select bind:value={kreaOptions.outpaint_top}><option value={0}>없음</option><option value={64}>64px</option><option value={128}>128px</option><option value={256}>256px</option><option value={384}>384px</option><option value={512}>512px</option></select></label>
                <label>오른쪽<select bind:value={kreaOptions.outpaint_right}><option value={0}>없음</option><option value={64}>64px</option><option value={128}>128px</option><option value={256}>256px</option><option value={384}>384px</option><option value={512}>512px</option></select></label>
                <label>아래쪽<select bind:value={kreaOptions.outpaint_bottom}><option value={0}>없음</option><option value={64}>64px</option><option value={128}>128px</option><option value={256}>256px</option><option value={384}>384px</option><option value={512}>512px</option></select></label>
              </div>
            </div>
            <div class="module-controls">
              <label class="module-slider"><span>생성 강도 <b>{Number(kreaOptions.anypaint_strength).toFixed(1)}</b></span><input type="range" min="0" max="2" step="0.1" bind:value={kreaOptions.anypaint_strength}></label>
              <label>경계 다시 그리기<select bind:value={kreaOptions.anypaint_boundary_redraw_px}><option value={0}>0px · 원본 우선</option><option value={16}>16px · 약하게</option><option value={32}>32px · 균형</option><option value={64}>64px · 자연스럽게</option></select></label>
            </div>
            <small class="module-caution">프롬프트에는 완성될 전체 장면을 적으세요. 원본 해상도 기준으로 작업하며 현재 다른 Krea 모듈과는 함께 실행하지 않습니다.</small>
          </div>
        {/if}
      </article>
      <article class="module-card" class:enabled={kreaModules.style}>
        <button type="button" class="module-toggle" aria-pressed={kreaModules.style} onclick={() => toggleKreaModule('style')}>
          <span class="module-icon">FX</span><span><strong>스타일 LoRA</strong><small>기본 모델 위에 시각 스타일 추가</small></span><i></i>
        </button>
        {#if kreaModules.style}
          <div class="module-body">
            <div class="lora-picker" aria-label="스타일 LoRA 선택">
              {#each kreaStyleCatalog as style}
                <button type="button" class:selected={hasKreaStyle(style.name)} aria-pressed={hasKreaStyle(style.name)} onclick={() => toggleKreaStyle(style.name)}><i>{hasKreaStyle(style.name) ? '✓' : '+'}</i><span><strong title={style.label}>{style.label}</strong><small title={style.detail}>{style.detail}</small></span></button>
              {/each}
            </div>
            {#if kreaStyleSelections.length}
              <div class="lora-stack">
                <header><strong>적용 순서</strong><span>{kreaStyleSelections.length}개 중첩</span></header>
                {#each kreaStyleSelections as style, index}
                  <div class="lora-stack-item">
                    <span><b>{index + 1}</b><strong title={kreaStyleLabel(style.name)}>{kreaStyleLabel(style.name)}</strong></span>
                    <label><input type="range" min="0" max="2" step="0.1" value={style.strength} oninput={(event) => updateKreaStyleStrength(style.name, event.currentTarget.value)}><b>{Number(style.strength).toFixed(1)}</b></label>
                    <button type="button" aria-label={`${kreaStyleLabel(style.name)} 제거`} onclick={() => toggleKreaStyle(style.name)}>×</button>
                  </div>
                {/each}
              </div>
            {:else}
              <small class="module-caution">위 목록에서 적용할 LoRA를 선택하세요.</small>
            {/if}
            <small class="module-caution">선택한 순서대로 중첩됩니다. 3개 이상은 스타일 충돌로 형태나 색이 과해질 수 있습니다.</small>
          </div>
        {/if}
      </article>
      <article class="module-card" class:enabled={kreaModules.userLora}>
        <button type="button" class="module-toggle" aria-pressed={kreaModules.userLora} onclick={() => toggleKreaModule('userLora')}>
          <span class="module-icon">MY</span><span><strong>사용자 LoRA</strong><small>LoRA 관리에서 등록한 인물·캐릭터·스타일</small></span><i></i>
        </button>
        {#if kreaModules.userLora}
          <div class="module-body">
            <div class="module-toolbar"><small>최대 5개까지 중첩할 수 있습니다.</small><button type="button" class="quiet" onclick={refreshUserLoras}>새로고침</button></div>
            {#if userLoraCatalog.length}
              <div class="lora-picker" aria-label="사용자 LoRA 선택">
                {#each userLoraCatalog as lora}
                  <button type="button" class:selected={hasUserLora(lora.filename)} aria-pressed={hasUserLora(lora.filename)} onclick={() => toggleUserLora(lora.filename)}><i>{hasUserLora(lora.filename) ? '✓' : '+'}</i><span><strong title={lora.name || lora.filename}>{lora.name || lora.filename}</strong><small title={lora.trigger_word || '트리거 없음'}>{lora.trigger_word || '트리거 없음'}</small></span></button>
                {/each}
              </div>
              {#if userLoraSelections.length}
                <div class="lora-stack">
                  <header><strong>적용 순서</strong><span>{userLoraSelections.length}개 중첩</span></header>
                  {#each userLoraSelections as selection, index}
                    <div class="lora-stack-item">
                      <span><b>{index + 1}</b><strong title={userLoraLabel(selection.filename)}>{userLoraLabel(selection.filename)}</strong></span>
                      <label><input type="range" min="-2" max="2" step="0.01" value={selection.strength} oninput={(event) => updateUserLoraStrength(selection.filename, event.currentTarget.value)}><b>{Number(selection.strength).toFixed(2)}</b></label>
                      <button type="button" aria-label={`${selection.filename} 제거`} onclick={() => toggleUserLora(selection.filename)}>×</button>
                    </div>
                  {/each}
                </div>
              {/if}
            {:else}
              <small class="module-caution">등록된 LoRA가 없습니다. 상단 LoRA 탭에서 먼저 추가하세요.</small>
            {/if}
          </div>
        {/if}
      </article>
      <article class="module-card" class:enabled={kreaModules.styleReference}>
        <button type="button" class="module-toggle" aria-pressed={kreaModules.styleReference} onclick={() => toggleKreaModule('styleReference')}>
          <span class="module-icon">REF</span><span><strong>스타일 이미지 참조</strong><small>Ostris Style Reference · 화풍·색감·질감 반영</small></span><i></i>
        </button>
        {#if kreaModules.styleReference}
          <div class="module-body">
            <div class="module-source-field"><label class="module-file">스타일 이미지 · 최대 2장<input type="file" accept="image/*" multiple onchange={(e) => addKreaRefs('styleReference', e.currentTarget.files)}><span class="module-file-display"><i>REF</i><b>{kreaStyleReferenceImages.length ? `${kreaStyleReferenceImages.length}장 선택됨` : '화풍을 가져올 이미지 선택'}</b></span></label><div class="module-source-actions"><button type="button" class="recent-result-open" onclick={() => recentImagePickerTarget = 'styleReference'}>최근 결과</button><button type="button" class="recent-result-open" onclick={() => presetImagePickerTarget = 'styleReference'}>프리셋</button><button type="button" class="recent-result-open" onclick={() => remoteImageTarget = 'styleReference'}>URL</button></div></div>
            {#if kreaStyleReferenceImages.length}<div class="reference-previews">{#each kreaStyleReferenceImages as image, i}<div><button type="button" class="reference-preview-open" onclick={(event) => showImage(event, image.preview || image.url, `스타일 참조 ${i + 1}`)}><img src={image.preview || image.url} alt="스타일 참조 {i + 1}"></button><button type="button" class="reference-preview-remove" aria-label="스타일 참조 제거" onclick={() => removeKreaRef('styleReference', i)}>×</button></div>{/each}</div>{/if}
            <label class="module-slider"><span>참조 강도 <b>{Number(kreaOptions.style_reference_strength).toFixed(1)}</b></span><input type="range" min="0" max="2" step="0.1" bind:value={kreaOptions.style_reference_strength}></label>
            <small class="module-caution">전용 INT8 모델을 사용하므로 다른 Krea 모듈과는 아직 함께 실행하지 않습니다.</small>
          </div>
        {/if}
      </article>
      <article class="module-card" class:enabled={kreaModules.vision}>
        <button type="button" class="module-toggle" aria-pressed={kreaModules.vision} onclick={() => toggleKreaModule('vision')}>
          <span class="module-icon">VL</span><span><strong>내용·구도 참조</strong><small>Qwen3-VL · 사물·배치·시각적 내용을 의미적으로 반영</small></span><i></i>
        </button>
        {#if kreaModules.vision}
          <div class="module-body">
            <div class="module-source-field"><label class="module-file">참조 이미지 · 최대 4장<input type="file" accept="image/*" multiple onchange={(e) => addKreaRefs('vision', e.currentTarget.files)}><span class="module-file-display"><i>VL</i><b>{kreaVisionImages.length ? `${kreaVisionImages.length}장 선택됨` : '내용을 참고할 이미지 선택'}</b></span></label><div class="module-source-actions"><button type="button" class="recent-result-open" onclick={() => recentImagePickerTarget = 'vision'}>최근 결과</button><button type="button" class="recent-result-open" onclick={() => presetImagePickerTarget = 'vision'}>프리셋</button><button type="button" class="recent-result-open" onclick={() => remoteImageTarget = 'vision'}>URL</button></div></div>
            {#if kreaVisionImages.length}<div class="reference-previews">{#each kreaVisionImages as image, i}<div><button type="button" class="reference-preview-open" onclick={(event) => showImage(event, image.preview || image.url, `내용·구도 참조 ${i + 1}`)}><img src={image.preview || image.url} alt="내용 참조 {i + 1}"></button><button type="button" class="reference-preview-remove" aria-label="내용 참조 제거" onclick={() => removeKreaRef('vision', i)}>×</button></div>{/each}</div>{/if}
            <div class="module-controls">
              <label>참조 방식<select bind:value={kreaOptions.vision_mode}><option value="descriptor">자연스럽게 반영</option><option value="instruct">변경 지시와 결합</option></select></label>
              <label>이미지 해석<select bind:value={kreaOptions.vision_megapixels}><option value={0.5}>빠르게</option><option value={1}>균형</option><option value={2}>세밀하게</option></select></label>
            </div>
            <small class="module-caution">정확한 얼굴 고정이나 인페인팅이 아닌 의미 기반 참조입니다.</small>
          </div>
        {/if}
      </article>
            </section>
          </div>
          <footer><button type="button" class="feature-modules-clear" disabled={!activeKreaModuleLabels.length} onclick={disableAllKreaModules}>모두 끄기</button><button type="button" class="feature-modules-done" onclick={() => featureModulesOpen = false}>완료</button></footer>
        </div>
      </div>
    {/if}
  {:else}
    <div class="drop" role="button" tabindex="0" ondragover={(e) => e.preventDefault()} ondrop={(e) => { e.preventDefault(); addRefs(e.dataTransfer.files) }}>
      <input type="file" accept="image/*" multiple={imageForm.mode === 'edit'} onchange={(e) => addRefs(e.currentTarget.files)}>
      <strong>{refs.length ? `${imageForm.mode === 'control' ? '제어' : '참조'} 이미지 ${refs.length}개` : `${imageForm.mode === 'control' ? '제어' : '참조'} 이미지 놓기`}</strong>
      <small>{imageForm.mode === 'control' ? '필수 · 윤곽을 추출할 이미지 1장' : `필수 · 최대 ${config?.image.max_reference_images || 4}개`} · 클릭하거나 드래그</small>
      {#if refs.length}<div class="drop-reference-previews">{#each refs as image, i}<div><button type="button" class="reference-preview-open" onclick={(event) => showImage(event, image.preview || image.url, `${imageModeMeta[imageForm.mode].label} 원본 ${i + 1}`)}><img src={image.preview || image.url} alt="참조 원본 {i + 1}"></button><button type="button" class="reference-preview-remove" aria-label="참조 원본 제거" onclick={(event) => { event.preventDefault(); event.stopPropagation(); removeRef(i) }}>×</button></div>{/each}</div>{/if}
    </div>
  {/if}
  <div class="resolution-control">
    <div class="resolution-heading"><div><strong>이미지 크기</strong><small>{imageForm.width}×{imageForm.height} · {(imageForm.width * imageForm.height / 1_000_000).toFixed(2)}MP</small></div><div class="segmented compact"><button type="button" class:active={imageResolutionMode === 'smart'} onclick={() => { imageResolutionMode = 'smart'; applySmartResolution() }}>간편</button><button type="button" class:active={imageResolutionMode === 'custom'} onclick={useCustomImageResolution}>직접</button></div></div>
    {#if imageResolutionMode === 'smart'}
      <div class="fields two smart-resolution-fields">
        <label>화면 비율<select bind:value={imageAspectRatio} onchange={applySmartResolution}>{#each imageAspectRatios as aspect}<option value={aspect[0]}>{aspect[0]} · {aspect[2]}</option>{/each}</select></label>
        <label>크기<select bind:value={imageMegapixels} onchange={applySmartResolution}><option value={0.75}>빠르게 · 0.75MP</option><option value={1}>기본 · 1MP</option><option value={2}>고해상도 · 2MP</option><option value={4} disabled={kreaModules.identity}>최대 품질 · 4MP</option></select></label>
      </div>
    {:else}
      <div class="fields two">
        <label>너비<input type="number" min="256" max="2048" step="any" bind:value={imageForm.width} onchange={() => imageForm.width = snapDimension(imageForm.width, 8, 256, 2048)}></label>
        <label>높이<input type="number" min="256" max="2048" step="any" bind:value={imageForm.height} onchange={() => imageForm.height = snapDimension(imageForm.height, 8, 256, 2048)}></label>
      </div>
    {/if}
  </div>
  {#if imageForm.mode === 'create'}
    <section class="image-generation-controls" aria-label="이미지 생성 설정">
      <div class="generation-control-heading"><strong>생성 설정</strong><small>{kreaOptions.sampling_preset === 'detail' ? 'ER-SDE / Simple' : kreaOptions.sampling_preset === 'moody' ? 'Euler A / Beta' : 'Euler / Simple'} · {kreaOptions.steps} steps</small></div>
      <div class="generation-control-grid">
        <label class="checkpoint-field"><span class="checkpoint-field-heading">체크포인트 <details class="checkpoint-help"><summary aria-label="체크포인트 설명">i</summary><span class="checkpoint-help-popover" role="tooltip"><b>첫 생성·모델 전환은 오래 걸릴 수 있습니다.</b><small>생성 시작 후 체크포인트·텍스트 인코더·VAE를 메모리에 적재합니다. 이미 적재된 모델을 다시 쓸 때보다 첫 작업의 대기 시간이 크게 늘 수 있습니다.</small>{#if selectedKreaCheckpointSource()}<a href={selectedKreaCheckpointSource()} target="_blank" rel="noreferrer">출처 ↗</a>{/if}</span></details></span><select value={selectedKreaCheckpoint()} onchange={(event) => selectKreaCheckpoint(event.currentTarget.value)}>{#if kreaModules.identity}<option value="identity-convrot">Identity 전용 · ConvRot INT8</option>{/if}<option value="official">Krea 2 Turbo · 공식 NVFP4</option>{#if checkpointVisible('chriscole-edit-v1.1')}<option value="chriscole-edit-v1.1" disabled={!imageCheckpointStatus?.variants?.find((item) => item.id === 'chriscole-edit-v1.1')?.ready}>Krea 2 Turbo Edit v1.1 · FP8</option>{/if}{#if checkpointVisible('moody-v7')}<option value="moody-v7" disabled={!imageCheckpointStatus?.variants?.find((item) => item.id === 'moody-v7')?.ready}>Moody Krea 2 Mix V7 · NVFP4</option>{/if}{#if checkpointVisible('moody-cutie-v4')}<option value="moody-cutie-v4" disabled={!imageCheckpointStatus?.variants?.find((item) => item.id === 'moody-cutie-v4')?.ready}>Moody Cutie Mix V4 · NVFP4</option>{/if}{#if checkpointVisible('moody-amateur-v1')}<option value="moody-amateur-v1" disabled={!imageCheckpointStatus?.variants?.find((item) => item.id === 'moody-amateur-v1')?.ready}>Moody Amateur Mix V1 · NVFP4</option>{/if}{#if checkpointVisible('ray-v1')}<option value="ray-v1" disabled={!imageCheckpointStatus?.variants?.find((item) => item.id === 'ray-v1')?.ready}>Ray Artshoot V1 · FP8</option>{/if}{#if checkpointVisible('ray-v2')}<option value="ray-v2" disabled={!imageCheckpointStatus?.variants?.find((item) => item.id === 'ray-v2')?.ready}>Ray Artshoot V2 · FP8</option>{/if}{#if checkpointVisible('ray-v2-nvfp4')}<option value="ray-v2-nvfp4" disabled={!imageCheckpointStatus?.nvfp4_conversion?.variants?.find((item) => item.id === 'ray-v2')?.validated}>Ray Artshoot V2 · NVFP4</option>{/if}{#if checkpointVisible('ray-v3')}<option value="ray-v3" disabled={!imageCheckpointStatus?.variants?.find((item) => item.id === 'ray-v3')?.ready}>Ray Artshoot V3 · INT8</option>{/if}{#if checkpointVisible('ray-v4')}<option value="ray-v4" disabled={!imageCheckpointStatus?.variants?.find((item) => item.id === 'ray-v4')?.ready}>Ray Artshoot V4 · INT8</option>{/if}{#if checkpointVisible('ray-v4-nvfp4')}<option value="ray-v4-nvfp4" disabled={!imageCheckpointStatus?.nvfp4_conversion?.variants?.find((item) => item.id === 'ray-v4')?.validated}>Ray Artshoot V4 · NVFP4</option>{/if}</select></label>
        <label class="sampling-field"><span>샘플링 프리셋</span><select bind:value={kreaOptions.sampling_preset}><option value="default">기본 · Euler / Simple</option><option value="detail">디테일 · ER-SDE / Simple</option><option value="moody">Moody 권장 · Euler A / Beta</option></select></label>
        <label><span>스텝</span><select bind:value={kreaOptions.steps}><option value={8}>8 · 기본</option><option value={10}>10 · 균형</option><option value={12}>12 · 디테일</option></select></label>
        {#if kreaModules.identity}<label><span>텍스트 인코더</span><select bind:value={kreaOptions.identity_encoder}><option value="heretic" disabled={imageCheckpointStatus?.identity_runtime && !imageCheckpointStatus.identity_runtime.heretic_ready}>Heretic · INT8 ConvRot</option><option value="default">기본 · Qwen3-VL FP8</option></select></label>{/if}
        <label><span>시드 <small>-1 무작위</small></span><input type="number" min="-1" bind:value={imageForm.seed}></label>
      </div>
    </section>
  {:else}
    <div class="fields"><label><span>시드 <small>-1은 무작위</small></span><input type="number" min="-1" bind:value={imageForm.seed}></label></div>
  {/if}
  <button class="primary" disabled={Boolean(imageDisabledMessage) || enhancingPrompt}>{enhancingPrompt ? '프롬프트 향상 중…' : busy ? '요청 중…' : imageEnhancementIsActive && !imageEnhancementIsCurrent ? '향상 및 생성 시작' : activeJobs().some((j) => j.kind === 'image' || j.kind === 'video' || j.kind === 'speech') ? '이미지 큐에 추가' : '생성 시작'}</button>
  {#if imageDisabledMessage}<small class="submit-hint">{imageDisabledMessage}</small>{/if}
</form>
