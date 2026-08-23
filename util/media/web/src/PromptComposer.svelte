<script>
  import { onDestroy } from 'svelte'
  import PoseLibraryModal from './PoseLibraryModal.svelte'
  import MakeupLibraryModal from './MakeupLibraryModal.svelte'
  import { lockModalScroll } from './modalScroll.js'

  export let onApply = () => {}
  export let activeStyles = []

  let expanded = false
  let selections = {}
  let customValues = {}
  let composedPrompt = ''
  let poseLibraryOpen = false
  let selectedPose = null
  let makeupLibraryOpen = false
  let selectedMakeup = null
  let exactText = ''
  let composerWarnings = []
  let selectedCount = 0
  let releaseScroll = null

  const actionPostures = new Set([
    'standing naturally and facing the camera', 'walking or running', 'seated in a relaxed pose',
    'a dynamic action pose', 'dancing with expressive movement', 'an unposed candid moment'
  ])
  const cameraFramings = new Set(['close-up portrait', 'waist-up medium shot', 'full-body shot', 'wide establishing shot'])
  const cameraAngles = new Set(['eye-level camera', 'low-angle view', 'high-angle view', 'over-the-shoulder view'])
  const styleLabels = {
    darkbrush: 'Dark Brush', dotmatrix: 'Dot Matrix', kidsdrawing: 'Kids Drawing', neondrip: 'Neon Drip',
    rainywindow: 'Rainy Window', retroanime: 'Retro Anime', softwatercolor: 'Soft Watercolor',
    sunsetblur: 'Sunset Blur', vintagetarot: 'Vintage Tarot'
  }
  const styleConflicts = {
    darkbrush: ['photorealistic photography', 'a polished 3D render'],
    dotmatrix: ['photorealistic photography', 'a polished 3D render', 'a soft watercolor painting'],
    kidsdrawing: ['photorealistic photography', 'a cinematic film still', 'editorial fashion photography', 'commercial product photography'],
    neondrip: ['photorealistic photography', 'commercial product photography'],
    rainywindow: ['anime cel-shaded illustration', 'an expressive oil painting'],
    retroanime: ['photorealistic photography', 'editorial fashion photography', 'an expressive oil painting', 'a soft watercolor painting', 'a polished 3D render'],
    softwatercolor: ['photorealistic photography', 'anime cel-shaded illustration', 'an expressive oil painting', 'a polished 3D render'],
    sunsetblur: ['a clean user-interface screen layout'],
    vintagetarot: ['photorealistic photography', 'editorial fashion photography', 'anime cel-shaded illustration', 'a polished 3D render', 'commercial product photography']
  }

  const groups = [
    {
      id: 'subject', label: '주제·인물', output: 'Subject', placeholder: '인물, 사물, 장면을 직접 입력',
      options: [
        ['single adult woman', '성인 여성 1명'], ['single adult man', '성인 남성 1명'],
        ['two people', '두 사람'], ['a group of people', '여러 사람'], ['a fictional character', '캐릭터'],
        ['a product or object', '제품·사물'], ['an animal', '동물'], ['architecture and landscape', '건축·풍경']
      ]
    },
    {
      id: 'appearance', label: '외형·복장', output: 'Appearance and clothing', placeholder: '헤어, 체형, 의상, 소재 등을 입력',
      options: [
        ['natural photorealistic features', '자연스러운 실사 외형'], ['East Asian facial features', '동아시아계 외형'],
        ['Western facial features', '서구권 외형'], ['casual everyday clothing', '캐주얼'],
        ['formal tailored clothing', '정장·포멀'], ['traditional clothing', '전통 의상'],
        ['futuristic clothing', '미래적 의상'], ['fantasy armor with detailed materials', '판타지 갑옷']
      ]
    },
    {
      id: 'expression', label: '표정', output: 'Expression', placeholder: '눈빛, 감정, 표정의 강도를 입력',
      options: [
        ['a neutral relaxed expression', '차분한 무표정'], ['a gentle natural smile', '은은한 미소'],
        ['a joyful open laugh', '밝게 웃음'], ['a skeptical, slightly disgusted expression', '회의적·불쾌'],
        ['a surprised expression', '놀람'], ['an angry, determined expression', '분노·결의'],
        ['a sorrowful expression', '슬픔'], ['a fearful, tense expression', '두려움·긴장']
      ]
    },
    {
      id: 'action', label: '동작·포즈', output: 'Action and pose', placeholder: '손동작, 시선, 자세, 상호작용을 입력',
      options: [
        ['standing naturally and facing the camera', '정면으로 자연스럽게 서기'], ['walking or running', '걷기·달리기'],
        ['seated in a relaxed pose', '편안하게 앉기'], ['looking back over the shoulder', '뒤돌아보기'],
        ['a dynamic action pose', '역동적 동작'], ['dancing with expressive movement', '춤추기'],
        ['interacting naturally with an object', '사물과 상호작용'], ['an unposed candid moment', '자연스러운 순간 포착']
      ]
    },
    {
      id: 'camera', label: '구도·카메라', output: 'Composition and camera', placeholder: '렌즈, 시점, 피사계 심도, 배치를 입력',
      options: [
        ['close-up portrait', '클로즈업'], ['waist-up medium shot', '상반신'], ['full-body shot', '전신'],
        ['wide establishing shot', '와이드 숏'], ['eye-level camera', '눈높이'], ['low-angle view', '로우 앵글'],
        ['high-angle view', '하이 앵글'], ['over-the-shoulder view', '오버숄더'],
        ['centered symmetrical composition', '중앙 대칭'], ['asymmetrical rule-of-thirds composition', '비대칭·삼분할']
      ]
    },
    {
      id: 'environment', label: '배경·환경', output: 'Environment', placeholder: '장소, 계절, 시간대, 주변 사물을 입력',
      options: [
        ['a clean photography studio', '스튜디오'], ['a lived-in home interior', '생활감 있는 실내'],
        ['a busy city street', '도시 거리'], ['a natural outdoor landscape', '자연 풍경'], ['a sunny beach', '해변'],
        ['a dense forest', '숲'], ['a futuristic science-fiction setting', 'SF 공간'],
        ['a magical fantasy environment', '판타지 공간'], ['a minimal uncluttered background', '미니멀 배경']
      ]
    },
    {
      id: 'lighting', label: '조명·색감', output: 'Lighting and color', placeholder: '광원 방향, 색상 팔레트, 대비를 입력',
      options: [
        ['soft natural daylight', '부드러운 자연광'], ['warm golden-hour sunlight', '골든아워'],
        ['direct hard flash', '강한 직광 플래시'], ['cinematic high-contrast lighting', '시네마틱 고대비'],
        ['colorful neon lighting', '네온 조명'], ['strong rim lighting', '림 라이트'],
        ['moody low-key lighting', '어두운 로우키'], ['bright high-key lighting', '밝은 하이키'],
        ['soft light entering from a window on the left', '좌측 창문광'], ['directional light from the right', '우측 방향광'],
        ['backlighting from behind the subject', '피사체 뒤 역광'], ['overhead tungsten lighting', '상단 텅스텐광'],
        ['soft diffused overcast light', '흐린 날 확산광'], ['hard frontal on-camera light', '정면 하드광'],
        ['a warm color palette', '따뜻한 색감'], ['a cool color palette', '차가운 색감'], ['a monochromatic palette', '단색 팔레트']
      ]
    },
    {
      id: 'texture', label: '재질·질감', output: 'Materials and texture', placeholder: '중요한 표면과 재질만 1~2개 입력',
      options: [
        ['natural skin texture', '자연스러운 피부결'], ['fine fabric weave', '섬세한 직물'],
        ['wet reflective surfaces', '젖은 반사 표면'], ['rough weathered stone', '풍화된 거친 석재'],
        ['brushed metal with subtle reflections', '브러시드 금속'], ['smooth matte vinyl', '매트 비닐'],
        ['visible paper grain', '종이 입자'], ['distinct analog film grain', '필름 그레인']
      ]
    },
    {
      id: 'style', label: '스타일·매체', output: 'Style and medium', placeholder: '작가명 대신 시각적 특성과 매체를 입력',
      options: [
        ['photorealistic photography', '포토리얼'], ['a cinematic film still', '영화 스틸'],
        ['editorial fashion photography', '패션 화보'], ['documentary snapshot photography', '다큐 스냅'],
        ['anime cel-shaded illustration', '애니 셀화'], ['a detailed digital illustration', '디지털 일러스트'],
        ['an expressive oil painting', '유화'], ['a soft watercolor painting', '수채화'],
        ['a polished 3D render', '3D 렌더'], ['commercial product photography', '제품 사진']
      ]
    },
    {
      id: 'text', label: '글자·레이아웃', output: 'Text and layout', placeholder: '삽입할 정확한 문구는 따옴표로 입력',
      options: [
        ['no visible text or lettering', '글자 없음'], ['leave clean negative space at the top', '상단 여백'],
        ['a polished poster layout', '포스터 배치'], ['a centered headline layout', '중앙 제목'],
        ['a product label or package layout', '라벨·패키지'], ['a clean user-interface screen layout', 'UI 화면'],
        ['crisp, correctly spelled, highly readable typography', '정확하고 선명한 글자']
      ]
    },
    {
      id: 'constraints', label: '반드시 유지할 조건', output: 'Constraints', placeholder: '변경 금지 요소나 정확한 개수 등을 입력',
      options: [
        ['preserve the subject identity', '인물 정체성 유지'], ['preserve the face and hairstyle', '얼굴·헤어 유지'],
        ['preserve the original pose', '포즈 유지'], ['preserve the original composition', '구도 유지'],
        ['preserve the original background', '배경 유지'], ['preserve the original colors and lighting', '색감·조명 유지'],
        ['keep the specified object count exactly', '사물 개수 엄수'], ['anatomically natural hands and fingers', '자연스러운 손'],
        ['do not add extra people or objects', '추가 인물·사물 금지']
      ]
    }
  ]

  function toggleOption(groupID, value) {
    const current = selections[groupID] || []
    selections = {
      ...selections,
      [groupID]: current.includes(value) ? current.filter((item) => item !== value) : [...current, value]
    }
  }

  function updateCustom(groupID, value) {
    customValues = { ...customValues, [groupID]: value }
  }

  function normalizedExactText(value) {
    return value.trim().replace(/^["“”]+|["“”]+$/g, '').replaceAll('"', "'")
  }

  function composePrompt(currentSelections, currentCustomValues, requestedText) {
    return groups.map((group) => {
      const parts = [...(currentSelections[group.id] || [])]
      const custom = (currentCustomValues[group.id] || '').trim()
      if (custom) parts.push(custom)
      if (group.id === 'text' && normalizedExactText(requestedText)) {
        parts.push(`the exact visible text reads "${normalizedExactText(requestedText)}"`)
      }
      return parts.length ? `${group.output}: ${parts.join(', ')}.` : ''
    }).filter(Boolean).join('\n')
  }

  function buildWarnings(currentSelections, pose, styles) {
    const warnings = []
    const actions = currentSelections.action || []
    const selectedPostures = actions.filter((item) => actionPostures.has(item))
    if (selectedPostures.length > 1) warnings.push('서로 다른 기본 동작이 여러 개 선택되었습니다. 한 장면에서 함께 가능한지 확인하세요.')
    if (pose && selectedPostures.length) warnings.push('포즈 라이브러리 자세와 기본 동작이 함께 선택되었습니다. 포즈를 우선하려면 기본 동작을 해제하세요.')

    const camera = currentSelections.camera || []
    if (camera.filter((item) => cameraFramings.has(item)).length > 1) warnings.push('클로즈업·상반신·전신·와이드 중 여러 프레이밍이 선택되었습니다.')
    if (camera.filter((item) => cameraAngles.has(item)).length > 1) warnings.push('눈높이·로우·하이·오버숄더 중 여러 카메라 시점이 선택되었습니다.')
    if (pose?.view === 'overhead view' && camera.some((item) => ['eye-level camera', 'low-angle view', 'over-the-shoulder view'].includes(item))) {
      warnings.push('선택한 오버헤드 포즈와 카메라 시점이 충돌합니다.')
    }
    if (pose?.view === 'elevated view' && camera.includes('low-angle view')) warnings.push('높은 시점 포즈와 로우 앵글이 충돌합니다.')

    const selectedStylePrompts = currentSelections.style || []
    for (const style of styles) {
      const conflicts = selectedStylePrompts.filter((item) => (styleConflicts[style] || []).includes(item))
      if (conflicts.length) warnings.push(`${styleLabels[style] || style} LoRA와 선택한 스타일·매체가 충돌할 수 있습니다.`)
    }
    if ((currentSelections.text || []).includes('no visible text or lettering') && normalizedExactText(exactText)) {
      warnings.push('글자 없음과 정확한 삽입 문구가 함께 지정되었습니다.')
    }
    return [...new Set(warnings)]
  }

  $: composedPrompt = composePrompt(selections, customValues, exactText)
  $: composerWarnings = buildWarnings(selections, selectedPose, activeStyles)
  $: selectedCount = Object.values(selections).reduce((count, values) => count + values.length, 0)
    + Object.values(customValues).filter((value) => value.trim()).length
    + (normalizedExactText(exactText) ? 1 : 0)

  $: {
    if (expanded && !releaseScroll) releaseScroll = lockModalScroll()
    else if (!expanded && releaseScroll) {
      releaseScroll()
      releaseScroll = null
    }
  }

  onDestroy(() => releaseScroll?.())

  function clearComposer() {
    selections = {}
    customValues = {}
    selectedPose = null
    selectedMakeup = null
    exactText = ''
  }

  function selectLibraryPose(pose) {
    const current = selections.action || []
    const withoutPrevious = selectedPose ? current.filter((item) => item !== selectedPose.prompt) : current
    selections = { ...selections, action: [...withoutPrevious, pose.prompt] }
    selectedPose = pose
  }

  function clearLibraryPose() {
    if (!selectedPose) return
    selections = { ...selections, action: (selections.action || []).filter((item) => item !== selectedPose.prompt) }
    selectedPose = null
  }

  function selectMakeupPreset(preset) {
    const current = selections.appearance || []
    const withoutPrevious = selectedMakeup ? current.filter((item) => item !== selectedMakeup.prompt) : current
    selections = { ...selections, appearance: [...withoutPrevious, preset.prompt] }
    selectedMakeup = preset
  }

  function clearMakeupPreset() {
    if (!selectedMakeup) return
    selections = { ...selections, appearance: (selections.appearance || []).filter((item) => item !== selectedMakeup.prompt) }
    selectedMakeup = null
  }

  function applyComposer(mode) {
    if (!composedPrompt) return
    onApply(composedPrompt, mode)
    expanded = false
  }

  function handleKeydown(event) {
    if (expanded && !poseLibraryOpen && !makeupLibraryOpen && event.key === 'Escape') expanded = false
  }
</script>

<svelte:window onkeydown={handleKeydown} />

<button class="prompt-composer" type="button" aria-haspopup="dialog" onclick={() => expanded = true}>
  <span>프롬프트 조립기</span>{#if selectedCount}<b>{selectedCount}개 선택</b>{/if}
</button>

{#if expanded}
  <div class="composer-modal-backdrop" role="presentation" onclick={(event) => { if (event.target === event.currentTarget) expanded = false }}>
    <section class="composer-modal" role="dialog" aria-modal="true" aria-label="Krea 2 프롬프트 조립기">
      <header>
        <div>
          <strong>Krea 2 프롬프트 조립기</strong>
          <small>원하는 항목을 골라 상세 프롬프트를 만들고 새로 쓰거나 현재 프롬프트 뒤에 추가합니다.</small>
        </div>
        <button type="button" aria-label="닫기" onclick={() => expanded = false}>×</button>
      </header>

      <div class="composer-body">
        <div class="composer-scroll">
          <div class="composer-grid">
            {#each groups as group}
              <fieldset>
                <legend>{group.label}</legend>
                <div class="composer-chips">
                  {#each group.options as option}
                    <button
                      type="button"
                      class:active={(selections[group.id] || []).includes(option[0])}
                      aria-pressed={(selections[group.id] || []).includes(option[0])}
                      onclick={() => toggleOption(group.id, option[0])}
                    >{option[1]}</button>
                  {/each}
                </div>
                {#if group.id === 'appearance'}
                  <div class="pose-library-control">
                    <button type="button" onclick={() => makeupLibraryOpen = true}>메이크업·얼굴 연출 20개 보기</button>
                    {#if selectedMakeup}
                      <span title={selectedMakeup.prompt}><b>{selectedMakeup.name}</b><button type="button" aria-label="선택한 얼굴 연출 지우기" onclick={clearMakeupPreset}>×</button></span>
                    {/if}
                  </div>
                {/if}
                {#if group.id === 'action'}
                  <div class="pose-library-control">
                    <button type="button" onclick={() => poseLibraryOpen = true}>포즈 120개 보기</button>
                    {#if selectedPose}
                      <span title={selectedPose.prompt}><b>{selectedPose.name.replace(/^\d+\s*\|\s*/, '')}</b><button type="button" aria-label="선택 포즈 지우기" onclick={clearLibraryPose}>×</button></span>
                    {/if}
                  </div>
                {/if}
                {#if group.id === 'text'}
                  <label class="exact-text-field"><span>정확한 삽입 문구</span><input type="text" bind:value={exactText} placeholder="따옴표 없이 문구만 입력"></label>
                {/if}
                <input
                  type="text"
                  value={customValues[group.id] || ''}
                  placeholder={group.placeholder}
                  aria-label={`${group.label} 직접 입력`}
                  oninput={(event) => updateCustom(group.id, event.currentTarget.value)}
                />
              </fieldset>
            {/each}
          </div>

          {#if composerWarnings.length}
            <div class="composer-warnings" role="status">
              <strong>선택 충돌 확인</strong>
              {#each composerWarnings as warning}<p>{warning}</p>{/each}
            </div>
          {/if}
        </div>

        <div class="composer-preview">
          <span>조립 결과</span>
          {#if composedPrompt}<pre>{composedPrompt}</pre>{:else}<p>위 항목을 선택하거나 직접 입력하세요.</p>{/if}
        </div>
      </div>

      <footer class="composer-actions">
        <button type="button" class="quiet" onclick={clearComposer} disabled={!composedPrompt}>모두 지우기</button>
        <button type="button" onclick={() => expanded = false}>닫기</button>
        <button type="button" class="apply secondary" onclick={() => applyComposer('append')} disabled={!composedPrompt}>이어서 적용</button>
        <button type="button" class="apply" onclick={() => applyComposer('replace')} disabled={!composedPrompt}>새로 적용</button>
      </footer>
    </section>
  </div>
{/if}

<PoseLibraryModal
  open={poseLibraryOpen}
  selectedID={selectedPose?.id || ''}
  onSelect={selectLibraryPose}
  onClose={() => poseLibraryOpen = false}
/>
<MakeupLibraryModal
  open={makeupLibraryOpen}
  selectedID={selectedMakeup?.id || ''}
  onSelect={selectMakeupPreset}
  onClose={() => makeupLibraryOpen = false}
/>

<style>
  .prompt-composer {
    display: flex;
    min-height: 48px;
    min-width: 0;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    height: 100%;
    margin: 0;
    padding: 0 12px;
    border: 1px solid #30373d;
    border-radius: 10px;
    color: #c6ccd1;
    background: #12161a;
    font-size: 11px;
    font-weight: 750;
  }

  .prompt-composer:hover { border-color: #657a53; background: #172018; }
  .prompt-composer b { color: #a8d970; font-size: 8px; }

  .composer-modal-backdrop {
    position: fixed;
    z-index: 35;
    inset: 0;
    display: grid;
    place-items: center;
    padding: 20px;
    background: #050708d9;
    backdrop-filter: blur(8px);
    overscroll-behavior: contain;
  }

  .composer-modal {
    display: grid;
    grid-template-rows: auto minmax(0, 1fr) auto;
    width: min(1120px, 96vw);
    height: min(860px, 92vh);
    overflow: hidden;
    overscroll-behavior: contain;
    border: 1px solid #4a555d;
    border-radius: 14px;
    background: #11161a;
    box-shadow: 0 24px 80px #000b;
  }

  .composer-modal > header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
    padding: 14px 16px;
    border-bottom: 1px solid #2d343a;
  }

  .composer-modal > header > div { display: grid; gap: 3px; }
  .composer-modal > header strong { color: #e3e8eb; font-size: 14px; }
  .composer-modal > header small { color: #76818a; font-size: 10px; }
  .composer-modal > header button {
    flex: 0 0 auto;
    width: 34px;
    height: 34px;
    padding: 0;
    border: 1px solid #394148;
    border-radius: 8px;
    color: #aab2b8;
    background: #191e22;
    font-size: 19px;
  }

  .composer-body {
    display: grid;
    min-height: 0;
    grid-template-rows: minmax(0, 1fr) auto;
    overflow: hidden;
  }

  .composer-scroll {
    min-height: 0;
    overflow-y: auto;
    padding: 14px 16px 10px;
    overscroll-behavior: contain;
  }

  .composer-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 9px;
  }

  fieldset {
    min-width: 0;
    margin: 0;
    padding: 10px;
    border: 1px solid #2b3238;
    border-radius: 9px;
    background: #0e1215;
  }

  legend { padding: 0 5px; color: #bcc5cb; font-size: 10px; font-weight: 750; }

  .composer-chips { display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 8px; }

  .composer-chips button {
    padding: 6px 8px;
    border: 1px solid #343b41;
    border-radius: 999px;
    color: #8e979e;
    background: #171c20;
    font-size: 9px;
    line-height: 1.1;
  }

  .composer-chips button.active {
    border-color: #789757;
    color: #e7f7d6;
    background: #293523;
  }

  .pose-library-control { display: grid; gap: 6px; margin: -1px 0 8px; }
  .pose-library-control > button {
    min-height: 34px;
    border: 1px dashed #657a53;
    border-radius: 7px;
    color: #b8db96;
    background: #202a1d;
    font-size: 9px;
    font-weight: 750;
  }
  .pose-library-control > span {
    display: flex;
    min-width: 0;
    align-items: center;
    justify-content: space-between;
    gap: 7px;
    padding: 6px 7px 6px 9px;
    border: 1px solid #536941;
    border-radius: 7px;
    color: #dceccc;
    background: #26301f;
    font-size: 9px;
  }
  .pose-library-control > span b { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .pose-library-control > span button {
    flex: 0 0 auto;
    width: 22px;
    height: 22px;
    padding: 0;
    border: 0;
    color: #9eb588;
    background: transparent;
    font-size: 15px;
  }

  fieldset input {
    box-sizing: border-box;
    width: 100%;
    height: 36px;
    padding: 0 9px;
    border: 1px solid #30373d;
    border-radius: 7px;
    color: #d8dde0;
    background: #11161a;
    font-size: 10px;
  }

  fieldset input:focus { border-color: #789757; outline: none; }

  .exact-text-field { display: grid; gap: 5px; margin: 0 0 7px; color: #89939a; font-size: 9px; font-weight: 700; }

  .composer-warnings {
    display: grid;
    gap: 4px;
    margin-top: 10px;
    padding: 9px 10px;
    border: 1px solid #725d34;
    border-radius: 9px;
    color: #cfbc91;
    background: #211c13;
  }
  .composer-warnings strong { font-size: 9px; }
  .composer-warnings p { margin: 0; font-size: 9px; line-height: 1.4; }

  .composer-preview {
    max-height: min(180px, 24vh);
    padding: 10px 16px;
    overflow: hidden;
    border-top: 1px solid #2d343a;
    background: #0d1114;
  }

  .composer-preview > span { color: #87928a; font-size: 9px; font-weight: 750; }
  .composer-preview pre,
  .composer-preview p {
    margin: 7px 0 0;
    color: #aeb7bd;
    font: 9px/1.55 ui-monospace, SFMono-Regular, Menlo, monospace;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
  }
  .composer-preview pre {
    max-height: min(140px, calc(24vh - 34px));
    overflow-y: auto;
    overscroll-behavior: contain;
  }
  .composer-preview p { color: #626c74; }

  .composer-actions {
    display: flex;
    justify-content: flex-end;
    gap: 7px;
    padding: 12px 16px;
    border-top: 1px solid #2d343a;
    background: #11161a;
  }
  .composer-actions button {
    min-height: 36px;
    padding: 0 12px;
    border: 1px solid #3a4248;
    border-radius: 8px;
    color: #aab2b8;
    background: #181d21;
    font-size: 10px;
    font-weight: 700;
  }
  .composer-actions button.quiet { margin-right: auto; }
  .composer-actions button.apply { border-color: #789757; color: #efffdc; background: #34452b; }
  .composer-actions button.apply.secondary { color: #cde9b0; background: #26301f; }
  .composer-actions button:disabled { cursor: default; opacity: .42; }

  @media (max-width: 700px) {
    .prompt-composer { padding: 0 9px; }
    .composer-modal-backdrop { padding: 0; }
    .composer-modal { width: 100vw; height: 100dvh; border: 0; border-radius: 0; }
    .composer-modal > header { padding: 11px 12px; }
    .composer-modal > header small { display: none; }
    .composer-grid { grid-template-columns: 1fr; }
    .composer-scroll { padding: 10px 10px 8px; }
    .composer-preview { max-height: min(150px, 22dvh); padding: 9px 10px; }
    .composer-preview pre { max-height: min(112px, calc(22dvh - 32px)); }
    .composer-chips button { padding: 7px 9px; }
    .composer-actions { padding: 9px 10px; }
    .composer-actions button { min-width: 0; padding: 0 7px; font-size: 9px; }
  }
</style>
