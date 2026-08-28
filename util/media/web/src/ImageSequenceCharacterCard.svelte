<script>
  import { onDestroy } from 'svelte'
  import { lockModalScroll } from './modalScroll.js'
  import { sequenceCharacterTraitChoices } from './lib/imageSequenceController.js'

  export let character
  export let index = 0
  export let busy = false
  export let preview = () => ''
  export let onName = () => {}
  export let onFiles = () => {}
  export let onRecent = () => {}
  export let onURL = () => {}
  export let onRemoveReference = () => {}
  export let onSetAnchor = () => {}
  export let onToggleTrait = () => {}
  export let onGenerateSheet = () => {}
  export let onApproveSheet = () => {}
  export let onDiscardSheet = () => {}
  export let onAnalyze = () => {}
  export let onDescription = () => {}
  export let onCanonicalPrompt = () => {}
  export let onRemove = () => {}

  let prepOpen = false
  let releaseScroll = null
  let progressClock = Date.now()
  let progressTimer = null
  $: observationEntries = Object.entries(character.observations || {}).filter(([, value]) => String(value || '').trim())
  $: anchorIndex = Math.min(character.reidReferenceIndex || 0, Math.max(0, character.references.length - 1))
  $: anchor = character.references[anchorIndex] || null
  $: lockedLabels = sequenceCharacterTraitChoices.filter(([value]) => character.lockedTraits?.[value]).map(([, label]) => label)
  $: sheetProgress = character.quadViewProgress || { detail: '시트 생성 준비', progress: 0.03 }
  $: sheetPercent = Math.max(3, Math.min(96, Math.round(Number(sheetProgress.progress || 0.03) * 100)))
  $: sheetElapsed = Math.max(0, Math.round((progressClock - Number(character.quadViewStartedAt || progressClock)) / 1000))
  $: {
    if (character.quadViewGenerating && !progressTimer) {
      progressClock = Date.now()
      progressTimer = setInterval(() => progressClock = Date.now(), 1000)
    } else if (!character.quadViewGenerating && progressTimer) {
      clearInterval(progressTimer)
      progressTimer = null
    }
  }
  $: {
    if (prepOpen && !releaseScroll) releaseScroll = lockModalScroll()
    else if (!prepOpen && releaseScroll) { releaseScroll(); releaseScroll = null }
  }
  onDestroy(() => { releaseScroll?.(); clearInterval(progressTimer) })
</script>

<article class="image-sequence-character-card compact">
  <header>
    <div class="image-sequence-character-summary">
      {#if anchor}<img src={preview(anchor)} alt={`${character.name} ReID 대표 이미지`}>{:else}<span>REF</span>{/if}
      <div><b>{character.name || `등장인물 ${index + 1}`}</b><small>{anchor ? `대표 이미지 · ${character.references.length}장 분석 자료` : '대표 이미지가 없습니다.'}</small></div>
    </div>
    <div class="image-sequence-character-badges">
      {#if index === 0 && anchor}<em>ReID</em>{:else if anchor}<em>텍스트 고정</em>{/if}
      {#if character.canonicalPromptEN}<em class="ready">분석 완료</em>{/if}
    </div>
    <button type="button" class="prepare" disabled={busy || character.analyzing} onclick={() => prepOpen = true}>캐릭터 준비</button>
    <button type="button" class="remove" aria-label="등장인물 제거" title="등장인물 제거" disabled={busy || character.analyzing} onclick={onRemove}>×</button>
  </header>
  {#if lockedLabels.length}<small class="image-sequence-character-lock-summary">고정: {lockedLabels.join(' · ')}</small>{/if}
</article>

{#if prepOpen}
  <div class="image-sequence-prep-backdrop" role="presentation" onclick={(event) => { if (event.target === event.currentTarget && !busy && !character.analyzing) prepOpen = false }}>
    <div class="image-sequence-prep-modal" role="dialog" aria-modal="true" aria-label={`${character.name} 캐릭터 준비`}>
      <header><div><strong>캐릭터 준비</strong><small>대표 ReID 이미지와 외형 고정 문구를 따로 관리합니다.</small></div><button type="button" aria-label="닫기" disabled={busy || character.analyzing} onclick={() => prepOpen = false}>×</button></header>
      <div class="image-sequence-prep-content">
        <div class="image-sequence-prep-workflow" aria-label="권장 캐릭터 준비 순서">
          <span><b>1</b> 대표 ReID 선택</span><i>→</i><span><b>2</b> 고정 특징 선택</span><i>→</i><span><b>3</b> Gemma 분석·승인</span><i>→</i><span><b>4</b> 3장 시험</span>
        </div>
        <label class="image-sequence-character-name"><span>이름</span><input value={character.name} placeholder={`등장인물 ${index + 1}`} oninput={(event) => onName(event.currentTarget.value)}></label>

        <section class="image-sequence-prep-section">
          <div class="image-sequence-prep-heading"><span><b>1. 참조 이미지</b><small>대표 이미지 한 장은 ReID에, 전체 이미지는 Gemma 외형 분석에 사용합니다.</small></span><em>{character.references.length}/6</em></div>
          <div class="image-sequence-character-sources">
            <label class="image-sequence-character-upload"><input type="file" accept="image/*" multiple disabled={busy || character.analyzing} onchange={(event) => { onFiles(event.currentTarget.files); event.currentTarget.value = '' }}><b>파일 선택</b><small>정면·전신·측면·세부 자료</small></label>
            <button type="button" disabled={busy || character.analyzing} onclick={onRecent}><b>생성 이미지</b><small>목록에서 가져오기</small></button>
            <button type="button" disabled={busy || character.analyzing} onclick={onURL}><b>URL</b><small>웹 이미지 가져오기</small></button>
          </div>
          {#if character.references.length}
            <div class="image-sequence-character-previews selectable">
              {#each character.references as reference, referenceIndex}
                <figure class:anchor={referenceIndex === anchorIndex}>
                  <button type="button" class="anchor-select" title="ReID 대표 이미지로 선택" disabled={busy || character.analyzing} onclick={() => onSetAnchor(referenceIndex)}><img src={preview(reference)} alt={reference.name || `참조 ${referenceIndex + 1}`}><span>{referenceIndex === anchorIndex ? '대표 · ReID' : '보조 분석'}</span></button>
                  <button type="button" class="reference-remove" aria-label="참조 제거" disabled={busy || character.analyzing} onclick={() => onRemoveReference(referenceIndex)}>×</button>
                  <figcaption>{reference.name || `참조 ${referenceIndex + 1}`}</figcaption>
                </figure>
              {/each}
            </div>
          {:else}<p class="image-sequence-prep-empty">사람은 얼굴이 선명한 상반신 또는 전신 한 장을 대표 이미지로 권장합니다.</p>{/if}
          <div class="image-sequence-reid-note"><b>{index === 0 ? '첫 캐릭터 · ReID 적용' : '추가 캐릭터 · 텍스트 고정'}</b><span>{index === 0 ? '대표 이미지가 모든 독립 장면에 직접 전달됩니다. 의상이나 액세서리도 따라올 수 있습니다.' : '현재 ReID는 첫 캐릭터 한 명에만 적용되며 이 캐릭터는 승인한 외형 문구로 유지합니다.'}</span></div>
        </section>

        <section class="image-sequence-prep-section">
          <div class="image-sequence-prep-heading"><span><b>2. 고정할 특징</b><small>선택하지 않은 항목은 분석 결과에는 보이지만 장면 고정 문구에서는 제외합니다.</small></span></div>
          <div class="image-sequence-trait-grid">{#each sequenceCharacterTraitChoices as [value, label]}<label><input type="checkbox" checked={Boolean(character.lockedTraits?.[value])} onchange={() => onToggleTrait(value)}><span>{label}</span></label>{/each}</div>
          <small class="image-sequence-trait-warning">ReID 자체가 대표 이미지의 특징을 읽으므로 액세서리·복장의 완전한 분리는 보장하지 않습니다.</small>
        </section>

        <section class="image-sequence-prep-section">
          <div class="image-sequence-prep-heading"><span><b>3. 외형 분석과 승인</b><small>전체 참조를 읽되 선택한 특징만 영어 고정 문구로 만듭니다.</small></span></div>
          <button type="button" class="image-sequence-character-analyze" disabled={busy || character.analyzing || !character.references.length || !lockedLabels.length} onclick={onAnalyze}>{character.analyzing ? 'Gemma가 외형을 읽는 중…' : character.canonicalPromptEN ? '이미지 다시 분석' : '이미지에서 외형 고정 문구 만들기'}</button>
          {#if character.error}<p class="image-sequence-character-error">{character.error}</p>{/if}
          {#if character.descriptionKO || character.canonicalPromptEN}
            <div class="image-sequence-character-review">
              <label><b>검토용 외형 설명</b><small>표정·자세·배경처럼 일시적인 내용이 섞였으면 고치세요.</small><textarea rows="4" value={character.descriptionKO} oninput={(event) => onDescription(event.currentTarget.value)}></textarea></label>
              <label><b>장면마다 넣을 영어 고정 문구</b><small>등장하는 장면에만 원문 그대로 삽입합니다.</small><textarea rows="8" value={character.canonicalPromptEN} oninput={(event) => onCanonicalPrompt(event.currentTarget.value)}></textarea></label>
              {#if observationEntries.length}<details><summary>부위별 전체 분석 {observationEntries.length}개 보기</summary><dl>{#each observationEntries as [key, value]}<dt>{key.replaceAll('_', ' ')}</dt><dd>{value}</dd>{/each}</dl></details>{/if}
            </div>
          {/if}
        </section>

        <section class="image-sequence-prep-section experimental">
          <div class="image-sequence-prep-heading"><span><b>선택 사항 · 4면 시트 후보</b><small>QuadView는 원본 외형을 바꿀 수 있어 자동으로 ReID 대표 이미지를 교체하지 않습니다.</small></span><em>실험</em></div>
          {#if character.quadViewCandidate}
            <div class="image-sequence-quadview-review">
              <figure><figcaption>원본 대표 이미지</figcaption><img src={preview(anchor)} alt="원본 ReID 대표 이미지"></figure>
              <figure><figcaption>4면 시트 후보</figcaption><img src={preview(character.quadViewCandidate)} alt="QuadView 후보"></figure>
            </div>
            <div class="image-sequence-quadview-actions"><button type="button" disabled={busy || character.quadViewGenerating} onclick={onDiscardSheet}>폐기</button><button type="button" class="primary" disabled={busy || character.quadViewGenerating || character.references.length >= 6} onclick={onApproveSheet}>보조 분석 자료로 승인</button></div>
          {:else}
            <p>대표 이미지를 바탕으로 얼굴 확대·정면·측면·후면 후보를 만듭니다. 원본과 직접 비교해 승인한 경우에만 Gemma의 보조 분석 자료로 추가하며, ReID 대표 이미지는 그대로 유지합니다. 약 2~4분 걸릴 수 있습니다.</p>
            {#if character.quadViewGenerating}
              <div class="image-sequence-sheet-progress" aria-label={`4면 시트 생성 ${sheetPercent}%`}>
                <span><b>{sheetProgress.detail || '4면 시트 생성 중'}</b><em>{sheetPercent}%</em></span>
                <div><i style={`width:${sheetPercent}%`}></i></div>
                <small>{sheetElapsed}초 경과 · 실제 엔진 단계</small>
              </div>
            {/if}
            <button type="button" class="image-sequence-character-analyze" disabled={busy || character.quadViewGenerating || !anchor} onclick={onGenerateSheet}>{character.quadViewGenerating ? '4면 시트 생성 중…' : '4면 시트 후보 만들기'}</button>
          {/if}
          {#if character.quadViewError}<p class="image-sequence-character-error">{character.quadViewError}</p>{/if}
        </section>
      </div>
      <footer><small>{character.canonicalPromptEN ? '외형 문구 승인됨' : '외형 분석을 완료해야 장면 생성이 활성화됩니다.'}</small><button type="button" disabled={busy || character.analyzing} onclick={() => prepOpen = false}>닫기</button></footer>
    </div>
  </div>
{/if}
