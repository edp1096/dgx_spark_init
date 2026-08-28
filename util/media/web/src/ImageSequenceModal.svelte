<script>
  import { onDestroy } from 'svelte'
  import { lockModalScroll } from './modalScroll.js'
  import ImageSequenceSceneCard from './ImageSequenceSceneCard.svelte'
  import ImageSequenceCharacterCard from './ImageSequenceCharacterCard.svelte'
  import RecentImagePicker from './RecentImagePicker.svelte'
  import RemoteImageModal from './RemoteImageModal.svelte'

  export let imageSequenceOpen = false
  export let busy = false
  export let imageForm
  export let kreaOptions
  export let kreaModules
  export let imageSequenceEntryMode = 'story'
  export let imageSequenceStoryIdea = ''
  export let imageSequenceSceneCount = 5
  export let imageSequencePrompts = []
  export let imageSequenceEnhancedPrompts = []
  export let imageSequenceSceneTitles = []
  export let imageSequenceSharedPrompt = ''
  export let imageSequenceSharedPromptEdited = false
  export let imageSequencePlanning = false
  export let imageSequencePlanError = ''
  export let imageSequenceCharacters = []
  export let imageJobs = []
  export let setImageSequenceEntryMode = () => {}
  export let setImageSequenceStoryIdea = () => {}
  export let setImageSequenceSceneCount = () => {}
  export let setImageSequenceSharedPrompt = () => {}
  export let applyStorySequenceExample = () => {}
  export let applySceneSequenceExample = () => {}
  export let planImageSequence = () => {}
  export let imageSequenceBlockedMessage = () => ''
  export let removeImageSequenceScene = () => {}
  export let moveImageSequenceScene = () => {}
  export let updateImageSequencePrompt = () => {}
  export let addImageSequenceScene = () => {}
  export let addImageSequenceCharacter = () => {}
  export let removeImageSequenceCharacter = () => {}
  export let setImageSequenceCharacterName = () => {}
  export let addImageSequenceCharacterFiles = () => {}
  export let addImageSequenceCharacterResult = () => {}
  export let removeImageSequenceCharacterReference = () => {}
  export let setImageSequenceCharacterReIDReference = () => {}
  export let toggleImageSequenceCharacterTrait = () => {}
  export let generateImageSequenceCharacterSheet = () => {}
  export let approveImageSequenceCharacterSheet = () => {}
  export let discardImageSequenceCharacterSheet = () => {}
  export let analyzeImageSequenceCharacter = () => {}
  export let setImageSequenceCharacterDescription = () => {}
  export let setImageSequenceCharacterPrompt = () => {}
  export let imageSequenceCharacterPreview = () => ''
  export let imageSequenceCharacterReadinessMessage = () => ''
  export let generateImage = () => {}
  let releaseScroll = null
  let characterPickerIndex = -1
  let characterURLIndex = -1

  $: canPlan = imageSequenceEntryMode === 'story'
    ? Boolean(imageSequenceStoryIdea.trim())
    : imageSequencePrompts.length >= 2 && imageSequencePrompts.every((prompt) => prompt.trim())
  $: characterReady = !imageSequenceCharacterReadinessMessage()
  $: hasReIDReference = Boolean(imageSequenceCharacters[0]?.references?.length)
  $: canGenerate = imageSequenceEntryMode === 'scenes' && imageSequencePrompts.length >= 2 && imageSequencePrompts.every((prompt) => prompt.trim()) && characterReady

  $: {
    if (imageSequenceOpen && !releaseScroll) releaseScroll = lockModalScroll()
    else if (!imageSequenceOpen && releaseScroll) {
      releaseScroll()
      releaseScroll = null
    }
  }

  onDestroy(() => releaseScroll?.())
</script>

{#if imageSequenceOpen}
  <div class="image-sequence-backdrop" role="presentation" onclick={(event) => { if (event.target === event.currentTarget && !busy) imageSequenceOpen = false }}>
    <div class="image-sequence-modal" role="dialog" aria-modal="true" aria-label="다중 장면 생성">
      <header>
        <div><strong>다중 장면</strong><small>이야기·삽화용 독립 장면을 한 번에 만듭니다.</small></div>
        <div class="image-sequence-header-actions">
          <button type="button" class="image-sequence-example" disabled={busy || imageSequencePlanning} onclick={applyStorySequenceExample}>이야기 예시</button>
          <button type="button" class="image-sequence-example" disabled={busy || imageSequencePlanning} onclick={applySceneSequenceExample}>장면 예시</button>
          <button type="button" aria-label="닫기" disabled={busy} onclick={() => imageSequenceOpen = false}>×</button>
        </div>
      </header>
      <div class="image-sequence-content">
        <div class="image-sequence-mode-tabs" role="tablist" aria-label="장면 입력 방식">
          <button type="button" class:active={imageSequenceEntryMode === 'story'} onclick={() => setImageSequenceEntryMode('story')}><b>이야기 나누기</b><small>줄거리에서 장면을 자동 구성</small></button>
          <button type="button" class:active={imageSequenceEntryMode === 'scenes'} onclick={() => setImageSequenceEntryMode('scenes')}><b>장면 직접 작성</b><small>각 장면을 원하는 대로 입력</small></button>
        </div>

        {#if imageSequenceEntryMode === 'story'}
          <section class="image-sequence-story">
            <label for="image-sequence-story"><strong>이야기 또는 주제</strong><small>등장인물, 사건, 분위기와 반드시 유지할 특징을 함께 적으면 좋습니다.</small></label>
            <textarea id="image-sequence-story" rows="5" value={imageSequenceStoryIdea} placeholder="예: 조선시대 한양에서 사라진 서책을 찾는 젊은 여검객의 하루. 같은 인물과 복장, 영화 같은 사극 분위기를 유지한다." oninput={(event) => setImageSequenceStoryIdea(event.currentTarget.value)}></textarea>
            <div class="image-sequence-story-actions">
              <label><span>장면 수</span><select value={imageSequenceSceneCount} onchange={(event) => setImageSequenceSceneCount(event.currentTarget.value)}>{#each [2, 3, 4, 5, 6, 8, 10, 12] as count}<option value={count}>{count}장</option>{/each}</select></label>
              <button type="button" class="primary" disabled={busy || imageSequencePlanning || !canPlan} onclick={planImageSequence}>{imageSequencePlanning ? '구성 중…' : '장면 구성'}</button>
            </div>
          </section>
        {/if}

        <section class="image-sequence-characters">
          <div class="image-sequence-characters-heading">
            <span><strong>등장인물 외형 고정</strong><small>Gemma 상세 묘사와 Krea ReID를 함께 사용합니다. 첫 캐릭터는 선택한 대표 이미지를 ReID 기준으로 쓰고 추가 캐릭터는 텍스트로 고정합니다.</small></span>
            {#if imageSequenceCharacters.length < 4}<button type="button" disabled={busy || imageSequencePlanning} onclick={addImageSequenceCharacter}>+ 등장인물</button>{/if}
          </div>
          {#if imageSequenceCharacters.length}
            <div class="image-sequence-character-list">
              {#each imageSequenceCharacters as character, index (character.id)}
                <ImageSequenceCharacterCard
                  {character} {index} {busy} preview={imageSequenceCharacterPreview}
                  onName={(value) => setImageSequenceCharacterName(index, value)}
                  onFiles={(files) => addImageSequenceCharacterFiles(index, files)}
                  onRecent={() => characterPickerIndex = index}
                  onURL={() => characterURLIndex = index}
                  onRemoveReference={(referenceIndex) => removeImageSequenceCharacterReference(index, referenceIndex)}
                  onSetAnchor={(referenceIndex) => setImageSequenceCharacterReIDReference(index, referenceIndex)}
                  onToggleTrait={(trait) => toggleImageSequenceCharacterTrait(index, trait)}
                  onGenerateSheet={() => generateImageSequenceCharacterSheet(index)}
                  onApproveSheet={() => approveImageSequenceCharacterSheet(index)}
                  onDiscardSheet={() => discardImageSequenceCharacterSheet(index)}
                  onAnalyze={() => analyzeImageSequenceCharacter(index)}
                  onDescription={(value) => setImageSequenceCharacterDescription(index, value)}
                  onCanonicalPrompt={(value) => setImageSequenceCharacterPrompt(index, value)}
                  onRemove={() => removeImageSequenceCharacter(index)}
                />
              {/each}
            </div>
          {:else}
            <p class="image-sequence-characters-empty">등장인물 고정이 필요할 때만 추가하세요. 일반적인 여러 장면 묶음은 지금처럼 바로 구성할 수 있습니다.</p>
          {/if}
        </section>

        <div class="image-sequence-notice">
          <strong>모든 장면에 공통 적용</strong>
          <span>{imageForm.width}×{imageForm.height} · {kreaOptions.checkpoint} · {kreaOptions.steps} steps{#if kreaModules.style || kreaModules.userLora} · 선택한 LoRA{/if}</span>
        </div>
        <div class="image-sequence-independence"><span>{hasReIDReference ? 'ReID 독립 생성' : '독립 생성'}</span><p>{hasReIDReference ? '각 장면은 첫 등장인물의 기준 이미지를 직접 참조합니다. 사람 얼굴·체형에는 강하지만 로봇 부품이나 복잡한 의상·소품은 장면마다 달라질 수 있습니다.' : '각 장면은 직전 결과를 다시 편집하지 않습니다. 공통 인물·화풍은 상세 고정 문구와 선택한 LoRA로 유지합니다.'}</p></div>
        {#if imageSequenceBlockedMessage()}<div class="image-sequence-warning">{imageSequenceBlockedMessage()}</div>{/if}
        {#if imageSequencePlanError}<div class="image-sequence-warning">장면 계획: {imageSequencePlanError}</div>{/if}
        {#if imageSequenceCharacterReadinessMessage()}<div class="image-sequence-warning">외형 고정: {imageSequenceCharacterReadinessMessage()}</div>{:else if hasReIDReference}<div class="image-sequence-ready-note">대표 ReID와 외형 문구가 준비됐습니다. 전체 생성 전 <b>3장 시험</b>으로 얼굴·복장·소품 유지 정도를 확인하세요.</div>{/if}
        {#if imageSequenceSharedPrompt}
          <section class="image-sequence-shared">
            <label for="image-sequence-shared"><b>공통 캐릭터·세계 설정</b><small>한국어로 고칠 수 있습니다. 내부에서는 영어 고정 블록으로 한 번 변환해 각 장면에 같은 문구를 넣습니다.</small></label>
            <textarea id="image-sequence-shared" rows="4" value={imageSequenceSharedPrompt} oninput={(event) => setImageSequenceSharedPrompt(event.currentTarget.value)}></textarea>
            <span>{imageSequenceSharedPromptEdited && !imageSequenceEnhancedPrompts.length ? '수정됨 · 다시 정리하거나 바로 생성하면 영어 고정 블록을 다시 만듭니다.' : '영어 고정 블록 준비됨 · 장면마다 외형 문구를 다시 번역하지 않습니다.'}</span>
          </section>
        {/if}

        {#if imageSequenceEntryMode === 'scenes'}
          <ol class="image-sequence-scenes">
            {#each imageSequencePrompts as prompt, index}
              <ImageSequenceSceneCard {index} total={imageSequencePrompts.length} {prompt} enhancedPrompt={imageSequenceEnhancedPrompts[index] || ''} sceneTitle={imageSequenceSceneTitles[index] || ''} onRemove={() => removeImageSequenceScene(index)} onPrompt={(value) => updateImageSequencePrompt(index, value)} onMove={(direction) => moveImageSequenceScene(index, direction)} />
            {/each}
          </ol>
          {#if imageSequencePrompts.length < 12}<button type="button" class="image-sequence-add" onclick={addImageSequenceScene}>+ 장면 추가</button>{/if}
        {/if}
      </div>
      <footer>
        <small>스토리보드·삽화 묶음용이며 영상 키프레임의 시간적 연속성은 보장하지 않습니다.</small>
        {#if imageSequenceEntryMode === 'scenes'}<button type="button" class="quiet" disabled={busy || imageSequencePlanning || !canPlan} onclick={planImageSequence}>{imageSequencePlanning ? '정리 중…' : imageSequenceEnhancedPrompts.length ? '다시 정리' : '프롬프트 정리'}</button>{/if}
        <button type="button" class="quiet" disabled={busy} onclick={() => imageSequenceOpen = false}>닫기</button>
        {#if imageSequenceEntryMode === 'story'}
          <button type="button" class="primary" disabled={busy || imageSequencePlanning || !canPlan} onclick={planImageSequence}>{imageSequencePlanning ? '구성 중…' : '장면 구성'}</button>
        {:else}
          {#if imageSequencePrompts.length > 3}<button type="button" class="quiet" title="첫 3장만 생성해 일관성을 확인합니다." disabled={busy || imageSequencePlanning || !canGenerate || Boolean(imageSequenceBlockedMessage())} onclick={() => generateImage(imageSequencePrompts.slice(0, 3))}>3장 시험</button>{/if}
          <button type="button" class="primary" disabled={busy || imageSequencePlanning || !canGenerate || Boolean(imageSequenceBlockedMessage())} onclick={() => generateImage(imageSequencePrompts)}>{busy || imageSequencePlanning ? '준비 중…' : `${imageSequencePrompts.length}장 생성`}</button>
        {/if}
      </footer>
    </div>
  </div>
{/if}

<RecentImagePicker
  open={characterPickerIndex >= 0}
  title="캐릭터 참조 이미지 선택"
  jobs={imageJobs}
  zIndex={85}
  onSelect={(job) => { addImageSequenceCharacterResult(characterPickerIndex, job); characterPickerIndex = -1 }}
  onClose={() => characterPickerIndex = -1}
/>

<RemoteImageModal
  open={characterURLIndex >= 0}
  title="캐릭터 참조 URL 가져오기"
  append={true}
  zIndex={86}
  onImport={(file) => addImageSequenceCharacterFiles(characterURLIndex, [file])}
  onClose={() => characterURLIndex = -1}
/>
