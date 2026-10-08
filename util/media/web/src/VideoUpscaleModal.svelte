<script>
  import { onDestroy } from 'svelte'
  import { lockModalScroll } from './modalScroll.js'

  export let source = null
  export let busy = false
  export let onSubmit = () => {}
  export let onClose = () => {}

  let scale = 2
  let batchSize = 5
  let temporalOverlap = 1
  let seed = -1
  let rangeMode = 'full'
  let startTime = 0
  let endTime = 10
  let advancedHelp = ''
  let previousSourceID = ''
  let releaseScroll = null

  $: width = Number(source?.width) || 0
  $: height = Number(source?.height) || 0
  $: duration = Number(source?.duration) || 0
  $: targetWidth = Math.round(width * Number(scale))
  $: targetHeight = Math.round(height * Number(scale))
  $: invalidScale = Number(scale) <= 1 || Number(scale) > 4
  $: invalidSize = width > 0 && height > 0 && Math.max(targetWidth, targetHeight) > 4096
  $: invalidRange = rangeMode === 'clip' && (startTime < 0 || endTime <= startTime || endTime > duration + .1 || endTime - startTime > 60.1)
  $: if (Number(batchSize) === 1 && Number(temporalOverlap) !== 0) temporalOverlap = 0

  $: if (source?.jobID && source.jobID !== previousSourceID) {
    previousSourceID = source.jobID
    rangeMode = Number(source.duration) > 60 ? 'clip' : 'full'
    startTime = 0
    endTime = Math.min(10, Number(source.duration) || 10)
    advancedHelp = ''
  }

  $: {
    if (source && !releaseScroll) releaseScroll = lockModalScroll()
    else if (!source && releaseScroll) {
      releaseScroll()
      releaseScroll = null
    }
  }

  function submit() {
    if (!source || busy || invalidScale || invalidSize || invalidRange) return
    onSubmit({ scale: Number(scale), batch_size: Number(batchSize), temporal_overlap: Number(temporalOverlap), seed: Number(seed), start_time: rangeMode === 'clip' ? Number(startTime) : 0, end_time: rangeMode === 'clip' ? Number(endTime) : 0 })
  }

  function presetScale(boxWidth, boxHeight) {
    if (width <= 0 || height <= 0) return 0
    const portrait = height > width
    const targetWidth = portrait ? boxHeight : boxWidth
    const targetHeight = portrait ? boxWidth : boxHeight
    return Math.min(targetWidth / width, targetHeight / height)
  }

  function setResolutionPreset(boxWidth, boxHeight) {
    const requestedScale = presetScale(boxWidth, boxHeight)
    if (requestedScale > 1 && requestedScale <= 4) scale = requestedScale
  }

  function resolutionPresetDisabled(boxWidth, boxHeight) {
    const requestedScale = presetScale(boxWidth, boxHeight)
    return requestedScale <= 1 || requestedScale > 4 || Math.max(width, height) * requestedScale > 4096
  }

  function resolutionPresetActive(boxWidth, boxHeight) {
    const requestedScale = presetScale(boxWidth, boxHeight)
    return requestedScale > 0 && Math.abs(Number(scale) - requestedScale) < .000001
  }

  function close() {
    if (!busy) onClose()
  }

  onDestroy(() => releaseScroll?.())
</script>

<svelte:window onkeydown={(event) => { if (source && event.key === 'Escape') close() }} />

{#if source}
  <div class="upscale-backdrop" role="presentation" onclick={(event) => { if (event.target === event.currentTarget) close() }}>
    <div class="upscale-modal" role="dialog" aria-modal="true" aria-label="영상 업스케일">
      <header><div><strong>영상 업스케일</strong><small title={source.title}>{source.title}</small></div><button type="button" aria-label="닫기" disabled={busy} onclick={close}>×</button></header>
      <div class="upscale-content">
        <section class="source-summary">
          <div><span>원본</span><strong>{width || '?'}×{height || '?'}</strong><small>{duration ? `${duration.toFixed(duration >= 60 ? 0 : 1)}초` : '길이 확인 중'}</small></div>
          <i>→</i>
          <div><span>결과</span><strong>{targetWidth || '?'}×{targetHeight || '?'}</strong><small>SeedVR2 복원</small></div>
        </section>
        <fieldset class="range-fieldset">
          <legend>처리 구간</legend>
          <div class="range-options"><button type="button" class:active={rangeMode === 'full'} disabled={duration > 60} title={duration > 60 ? '60초를 넘는 원본은 구간을 선택하세요.' : ''} onclick={() => rangeMode = 'full'}>전체 영상</button><button type="button" class:active={rangeMode === 'clip'} onclick={() => rangeMode = 'clip'}>구간 선택</button></div>
          {#if rangeMode === 'clip'}<div class="range-inputs"><label>시작 (초)<input type="number" min="0" max={duration} step="any" bind:value={startTime}></label><span>—</span><label>끝 (초)<input type="number" min="0.001" max={duration} step="any" bind:value={endTime}></label><small>최대 60초</small></div>{/if}
          {#if duration > 60}<small>긴 원본은 메모리 보호를 위해 최대 60초 구간으로 처리합니다.</small>{/if}
        </fieldset>
        <fieldset>
          <legend>확대 배율</legend>
          <div class="scale-options">
            {#each [1.5, 2, 3, 4] as value}<button type="button" class:active={Math.abs(Number(scale) - value) < .001} disabled={width > 0 && height > 0 && Math.max(width, height) * value > 4096} onclick={() => scale = value}>{value}배</button>{/each}
          </div>
          <div class="target-options">
            <button type="button" class:active={resolutionPresetActive(1920, 1088)} disabled={resolutionPresetDisabled(1920, 1088)} onclick={() => setResolutionPreset(1920, 1088)}>FHD · 1920×1088</button>
            <button type="button" class:active={resolutionPresetActive(2560, 1408)} disabled={resolutionPresetDisabled(2560, 1408)} onclick={() => setResolutionPreset(2560, 1408)}>2K/QHD · 2560×1408</button>
            <button type="button" class:active={resolutionPresetActive(3840, 2176)} disabled={resolutionPresetDisabled(3840, 2176)} onclick={() => setResolutionPreset(3840, 2176)}>4K · 3840×2176</button>
            <label>직접 배율<input type="number" min="1.01" max="4" step="0.01" bind:value={scale}></label>
          </div>
          <small>LTX 정렬 해상도 상자 안에 원본 비율을 유지해 맞춥니다. 세로 영상은 상자를 자동으로 회전합니다.</small>
        </fieldset>
        <details>
          <summary>고급 설정</summary>
          <div class="advanced-fields">
            <div class="advanced-field">
              <div class="advanced-field-title"><span>시간 배치</span><button type="button" class:active={advancedHelp === 'batch'} aria-label="시간 배치 설명" aria-expanded={advancedHelp === 'batch'} onclick={() => advancedHelp = advancedHelp === 'batch' ? '' : 'batch'}>i</button></div>
              {#if advancedHelp === 'batch'}<p class="advanced-help">한 번에 함께 복원하는 프레임 수입니다. 높이면 장면의 시간적 일관성이 좋아질 수 있지만 메모리 사용량과 처리 시간이 늘어납니다.</p>{/if}
              <select aria-label="시간 배치" bind:value={batchSize}><option value={1}>1 · 메모리 절약</option><option value={5}>5 · 기본</option><option value={9}>9 · 연속성 우선</option><option value={13}>13 · 긴 시간 배치</option></select>
            </div>
            <div class="advanced-field">
              <div class="advanced-field-title"><span>겹침 프레임</span><button type="button" class:active={advancedHelp === 'overlap'} aria-label="겹침 프레임 설명" aria-expanded={advancedHelp === 'overlap'} onclick={() => advancedHelp = advancedHelp === 'overlap' ? '' : 'overlap'}>i</button></div>
              {#if advancedHelp === 'overlap'}<p class="advanced-help">배치 경계에서 앞뒤 프레임을 겹쳐 처리합니다. 값을 높이면 경계의 깜빡임과 단절을 줄일 수 있지만 중복 연산이 늘어납니다.</p>{/if}
              <select aria-label="겹침 프레임" bind:value={temporalOverlap}><option value={0}>0 · 빠르게</option><option value={1} disabled={Number(batchSize) <= 1}>1 · 기본</option><option value={2} disabled={Number(batchSize) <= 2}>2 · 경계 안정</option><option value={3} disabled={Number(batchSize) <= 3}>3</option><option value={4} disabled={Number(batchSize) <= 4}>4 · 연속성 우선</option></select>
            </div>
            <label>시드<input type="number" min="-1" bind:value={seed}><small>-1은 무작위</small></label>
          </div>
        </details>
        {#if (rangeMode === 'full' ? duration : endTime - startTime) > 30}<p class="warning">긴 구간입니다. 프레임 수에 비례해 오래 걸리며 다른 생성 작업은 이 작업이 끝난 뒤 실행됩니다.</p>{/if}
        {#if invalidScale}<p class="error">배율은 1보다 크고 4 이하여야 합니다.</p>{/if}
        {#if invalidSize}<p class="error">선택한 배율은 4096px 제한을 초과합니다.</p>{/if}
        {#if invalidRange}<p class="error">시작과 끝을 원본 안에서 최대 60초 구간으로 입력하세요.</p>{/if}
      </div>
      <footer><button type="button" onclick={close} disabled={busy}>취소</button><button type="button" class="primary" onclick={submit} disabled={busy || invalidScale || invalidSize || invalidRange}>{busy ? '큐에 추가 중…' : '업스케일 시작'}</button></footer>
    </div>
  </div>
{/if}

<style>
  .upscale-backdrop{position:fixed;z-index:120;inset:0;display:grid;place-items:center;padding:18px;background:#050705df;backdrop-filter:blur(8px);overscroll-behavior:contain}
  .upscale-modal{overflow:hidden;width:min(560px,96vw);border:1px solid #3b463c;border-radius:14px;background:#151a16;box-shadow:0 24px 80px #000b}
  header{position:static;display:flex;align-items:center;justify-content:space-between;min-height:58px;padding:12px 16px;border-bottom:1px solid #303731;background:#181e19}
  header div{display:grid;gap:3px;min-width:0} header strong{color:#edf2eb;font-size:14px} header small{overflow:hidden;max-width:440px;color:#7d877e;font-size:10px;text-overflow:ellipsis;white-space:nowrap}
  header button{border:0;color:#aeb7af;background:transparent;font-size:24px;cursor:pointer} header button:disabled{opacity:.4}
  .upscale-content{display:grid;gap:13px;padding:14px}
  .source-summary{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:10px;border:1px solid #2e392f;border-radius:10px;padding:12px;background:#101511}
  .source-summary div{display:grid;gap:3px}.source-summary span{color:#849086;font-size:9px}.source-summary strong{color:#e3eee0;font-size:15px}.source-summary small{color:#748078;font-size:9px}.source-summary i{color:#71816e;font-style:normal}
  fieldset{display:grid;gap:8px;margin:0;border:1px solid #303a31;border-radius:10px;padding:10px}legend{padding:0 5px;color:#b8c5b8;font-size:10px}.scale-options{display:grid;grid-template-columns:repeat(4,1fr);gap:7px}.scale-options button,.target-options button{min-height:34px;border:1px solid #3a463b;border-radius:7px;color:#b8c2b9;background:#1b221c;cursor:pointer}.scale-options button.active,.target-options button.active{border-color:#8aae70;color:#dffbc8;background:#293526}.scale-options button:disabled,.target-options button:disabled{opacity:.32;cursor:not-allowed}.target-options{display:grid;grid-template-columns:repeat(3,1fr);gap:7px}.target-options label{display:grid;grid-template-columns:auto 1fr;grid-column:1/-1;align-items:center;gap:5px;border:1px solid #3a463b;border-radius:7px;padding:0 7px;color:#8e998f;font-size:9px;background:#111612}.target-options input{min-width:0;width:100%;height:30px;border:0;color:#d4ddd5;background:transparent;text-align:right}fieldset>small{color:#78827a;font-size:9px}
  .range-options{display:grid;grid-template-columns:1fr 1fr;gap:7px}.range-options button{min-height:31px;border:1px solid #3a463b;border-radius:7px;color:#aeb9af;background:#1b221c;cursor:pointer}.range-options button.active{border-color:#789263;color:#dbefca;background:#283325}.range-options button:disabled{opacity:.35;cursor:not-allowed}.range-inputs{display:grid;grid-template-columns:1fr auto 1fr auto;align-items:end;gap:7px}.range-inputs label{display:grid;gap:4px;color:#8e998f;font-size:9px}.range-inputs input{height:31px;min-width:0;border:1px solid #364037;border-radius:6px;padding:0 7px;color:#d4ddd5;background:#111612}.range-inputs span,.range-inputs small{padding-bottom:8px;color:#748077;font-size:9px}
  details{border:1px solid #303a31;border-radius:10px;padding:9px 10px}summary{color:#aab7aa;font-size:10px;cursor:pointer}.advanced-fields{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin-top:10px}.advanced-fields label,.advanced-field{position:relative;display:grid;align-content:start;gap:5px;color:#8e9a8f;font-size:9px}.advanced-field-title{display:flex;align-items:center;justify-content:space-between;min-height:16px}.advanced-field-title button{display:grid;place-items:center;width:16px;height:16px;border:1px solid #4a574b;border-radius:50%;padding:0;color:#9da99e;background:#1a211b;font:700 9px/1 serif;cursor:pointer}.advanced-field-title button.active{border-color:#86a76d;color:#e1f5d2;background:#30402b}.advanced-help{position:absolute;z-index:3;right:0;bottom:calc(100% - 13px);width:min(250px,78vw);margin:0;border:1px solid #536050;border-radius:8px;padding:8px 9px;color:#d5ddd4;background:#202820;box-shadow:0 8px 24px #000a;font-size:9px;line-height:1.45}.advanced-field:first-child .advanced-help{right:auto;left:0}.advanced-fields select,.advanced-fields input{min-width:0;height:32px;border:1px solid #364037;border-radius:6px;padding:0 7px;color:#d4ddd5;background:#111612}.advanced-fields small{color:#707a72;font-size:8px}
  .warning,.error{margin:0;border-radius:7px;padding:8px 10px;font-size:9px;line-height:1.5}.warning{color:#d7bd82;background:#3b2c174f}.error{color:#f0a7a7;background:#4b20205c}
  footer{display:flex;gap:8px;padding:9px 12px;border-top:1px solid #303731;background:#181e19}footer button{flex:1 1 25%;min-width:0;min-height:31px;border:1px solid #3c463e;border-radius:7px;padding:5px 11px;color:#b9c4ba;background:#202621;cursor:pointer}footer .primary{flex:3 1 75%;border-color:#668253;color:#e2f7d1;background:#34482d}footer button:disabled{opacity:.45;cursor:not-allowed}
  @media(max-width:600px){.upscale-backdrop{padding:6px}.advanced-fields{grid-template-columns:1fr}.target-options{grid-template-columns:repeat(3,1fr)}header small{max-width:72vw}}
</style>
