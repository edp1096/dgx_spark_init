<script>
  import { onDestroy } from 'svelte'
  import { lockModalScroll } from './modalScroll.js'

  export let imageSequenceOpen = false
  export let busy = false
  export let imageForm
  export let kreaOptions
  export let kreaModules
  export let imageSequenceBase = null
  export let imageSequenceStrength = 0.85
  export let setImageSequenceBase = () => {}
  export let setImageSequenceStrength = () => {}
  export let imageSequencePrompts = []
  export let imageSequenceMaskPreviews = []
  export let imageSequenceRegions = []
  export let imageSequenceMaskEditorIndex = -1
  export let imageSequenceRegionPicker = -1
  export let recentImagePickerTarget = ''
  export let applyRobotSequenceExample = () => {}
  export let clearImageSequenceMasks = () => {}
  export let imageSequenceBlockedMessage = () => ''
  export let removeImageSequenceScene = () => {}
  export let updateImageSequencePrompt = () => {}
  export let imageSequenceRegionOption = () => ({ label: '' })
  export let addImageSequenceScene = () => {}
  export let generateImage = () => {}
  let releaseScroll = null

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
    <div class="image-sequence-modal" role="dialog" aria-modal="true" aria-label="연속 이미지 생성">
      <header>
        <div><strong>연속 이미지 생성</strong><small>첫 장을 만든 뒤, 직전 결과를 원본으로 사용해 다음 장면을 순서대로 만듭니다.</small></div>
        <div class="image-sequence-header-actions"><button type="button" class="image-sequence-example" disabled={busy} onclick={applyRobotSequenceExample}>로봇 3장 예제</button><button type="button" aria-label="닫기" disabled={busy} onclick={() => imageSequenceOpen = false}>×</button></div>
      </header>
      <div class="image-sequence-content">
        <div class="image-sequence-notice"><strong>현재 생성 설정 유지</strong><span>{imageForm.width}×{imageForm.height} · {kreaOptions.checkpoint} · {kreaOptions.steps} steps{#if kreaModules.style || kreaModules.userLora} · 선택한 LoRA 적용{/if}</span></div>
        <div class="image-sequence-base" class:ready={Boolean(imageSequenceBase)}>
          {#if imageSequenceBase}
            <img src={imageSequenceBase.url} alt="연속 생성 첫 장면"><span><small>첫 장면 준비됨</small><b title={imageSequenceBase.prompt}>{imageSequenceBase.name}</b><em>이 이미지 위에 장면별 마스크를 칠할 수 있습니다.</em></span><button type="button" onclick={() => { clearImageSequenceMasks(); setImageSequenceBase(null) }}>새로 생성</button>
          {:else}
            <span><small>첫 장면</small><b>프롬프트로 새로 생성</b><em>실제 이미지 위에 영역을 칠하려면 기존 결과를 첫 장으로 선택하세요.</em></span><button type="button" onclick={() => recentImagePickerTarget = 'sequenceBase'}>생성 이미지 선택</button>
          {/if}
        </div>
        {#if imageSequenceBlockedMessage()}<div class="image-sequence-warning">{imageSequenceBlockedMessage()}</div>{/if}
        <label class="image-sequence-strength"><span><strong>장면 연속성</strong><small>낮추면 동작·구도 변화가 커지고, 높이면 직전 장면을 더 강하게 유지합니다.</small></span><input type="range" min="0.4" max="1.2" step="0.05" value={imageSequenceStrength} oninput={(event) => setImageSequenceStrength(event.currentTarget.value)}><b>{Number(imageSequenceStrength).toFixed(2)}</b></label>
        <ol class="image-sequence-scenes">
          {#each imageSequencePrompts as prompt, index}
            <li>
              <div class="image-sequence-scene-heading"><span>{index + 1}</span><label for={`sequence-scene-${index}`}>{index === 0 ? '첫 장면 · 전체 묘사' : `장면 ${index + 1} · 직전 장면에서 바꿀 내용`}</label>{#if index > 1}<button type="button" aria-label={`장면 ${index + 1} 제거`} onclick={() => removeImageSequenceScene(index)}>×</button>{/if}</div>
              <textarea id={`sequence-scene-${index}`} rows="3" value={prompt} placeholder={index === 0 ? '인물·장소·조명·구도를 포함한 첫 장면' : '예: 같은 인물이 창가로 걸어가며 카메라가 옆으로 이동한다'} oninput={(event) => updateImageSequencePrompt(index, event.currentTarget.value)}></textarea>
              {#if index > 0}
                <div class="image-sequence-scene-tools">
                  <span>{#if imageSequenceMaskPreviews[index]}<img src={imageSequenceMaskPreviews[index]} alt={`장면 ${index + 1} 마스크`}>{:else}<i class={`image-sequence-region-preview region-${imageSequenceRegions[index] || 'all'}`}><span></span></i>{/if}<b>{imageSequenceRegions[index] === 'custom' ? '직접 칠한 영역' : imageSequenceRegionOption(imageSequenceRegions[index]).label}</b></span>
                  <button type="button" class="paint" disabled={!imageSequenceBase} title={imageSequenceBase ? '첫 장면 위에 변경 영역을 직접 칠합니다.' : '먼저 생성 이미지를 첫 장으로 선택하세요.'} onclick={() => imageSequenceMaskEditorIndex = index}>영역 칠하기</button>
                  <button type="button" onclick={() => imageSequenceRegionPicker = index}>빠른 영역</button>
                </div>
                <small>{imageSequenceRegions[index] === 'all' ? '전체 이미지를 프롬프트로 편집합니다.' : '마스크 밖은 직전 장면을 그대로 보존합니다.'}</small>
              {/if}
            </li>
          {/each}
        </ol>
        {#if imageSequencePrompts.length < 6}<button type="button" class="image-sequence-add" onclick={addImageSequenceScene}>+ 장면 추가</button>{/if}
      </div>
      <footer>
        <small>2~6장 · 영상 탭에는 자동 배치하지 않습니다.</small>
        <button type="button" class="quiet" disabled={busy} onclick={() => imageSequenceOpen = false}>닫기</button>
        <button type="button" class="primary" disabled={busy || Boolean(imageSequenceBlockedMessage()) || imageSequencePrompts.some((prompt) => !prompt.trim())} onclick={() => generateImage(imageSequencePrompts)}>{busy ? '큐에 추가 중…' : imageSequenceBase ? `나머지 ${imageSequencePrompts.length - 1}장 생성` : `${imageSequencePrompts.length}장 생성`}</button>
      </footer>
    </div>
  </div>
{/if}
