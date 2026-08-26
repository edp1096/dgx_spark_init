<script>
  import { onDestroy } from 'svelte'
  import { lockModalScroll } from './modalScroll.js'

  export let job = null
  export let busy = false
  export let onSubmit = () => {}
  export let onClose = () => {}

  const formatOptions = [
    ['srt', 'SRT'], ['vtt', 'VTT'], ['timestamped_txt', '타임코드 TXT'], ['txt', '일반 TXT']
  ]
  let mode = 'none'
  let formats = ['srt', 'txt']
  let previousID = ''
  let releaseScroll = null

  $: if (job?.id && job.id !== previousID) {
    previousID = job.id
    mode = job.params?.translation_mode || 'none'
    formats = Array.isArray(job.params?.output_formats) && job.params.output_formats.length
      ? [...job.params.output_formats]
      : Object.keys(job.outputs || {}).filter((value) => formatOptions.some(([id]) => id === value))
    if (!formats.length) formats = ['srt', 'txt']
  }

  $: {
    if (job && !releaseScroll) releaseScroll = lockModalScroll()
    else if (!job && releaseScroll) {
      releaseScroll()
      releaseScroll = null
    }
  }

  function toggleFormat(value) {
    formats = formats.includes(value) ? formats.filter((item) => item !== value) : [...formats, value]
  }

  function close() {
    if (!busy) onClose()
  }

  function submit() {
    if (!job || busy || !formats.length) return
    onSubmit({ translation_mode: mode, output_formats: formats })
  }

  onDestroy(() => releaseScroll?.())
</script>

<svelte:window onkeydown={(event) => { if (job && event.key === 'Escape') close() }} />

{#if job}
  <div class="subtitle-regenerate-backdrop" role="presentation" onclick={(event) => { if (event.target === event.currentTarget) close() }}>
    <section class="subtitle-regenerate-modal" role="dialog" aria-modal="true" aria-label="자막 재생성">
      <header><div><strong>자막 재생성</strong><small title={job.prompt}>{job.prompt}</small></div><button type="button" aria-label="닫기" disabled={busy} onclick={close}>×</button></header>
      <div class="subtitle-regenerate-content">
        <p>저장된 원문과 번역문을 다시 조합합니다. 영상 다운로드와 음성 인식은 반복하지 않습니다.</p>
        <fieldset>
          <legend>자막 표시</legend>
          <div class="subtitle-mode-options">
            <button type="button" class:active={mode === 'none'} onclick={() => mode = 'none'}><strong>원문만</strong><small>받아쓴 언어 그대로</small></button>
            <button type="button" class:active={mode === 'translated'} onclick={() => mode = 'translated'}><strong>번역문만</strong><small>{job.params?.target_language || '번역 언어'}만 표시</small></button>
            <button type="button" class:active={mode === 'bilingual'} onclick={() => mode = 'bilingual'}><strong>원문 + 번역문</strong><small>한 큐에 두 줄로 표시</small></button>
          </div>
        </fieldset>
        <fieldset>
          <legend>결과 파일</legend>
          <div class="subtitle-format-options">
            {#each formatOptions as option}
              <button type="button" class:active={formats.includes(option[0])} aria-pressed={formats.includes(option[0])} onclick={() => toggleFormat(option[0])}>{option[1]}</button>
            {/each}
          </div>
          {#if !formats.length}<small class="format-warning">결과 형식을 하나 이상 선택하세요.</small>{/if}
        </fieldset>
      </div>
      <footer><button type="button" disabled={busy} onclick={close}>취소</button><button type="button" class="primary" disabled={busy || !formats.length} onclick={submit}>{busy ? '재생성 중…' : '자막 재생성'}</button></footer>
    </section>
  </div>
{/if}

<style>
  .subtitle-regenerate-backdrop{position:fixed;z-index:112;inset:0;display:grid;place-items:center;padding:16px;background:#050705df;backdrop-filter:blur(8px);overscroll-behavior:contain}
  .subtitle-regenerate-modal{overflow:hidden;width:min(620px,96vw);border:1px solid #3b463c;border-radius:14px;color:#e5eae5;background:#151a16;box-shadow:0 24px 80px #000b}
  header{position:static;display:flex;align-items:center;justify-content:space-between;height:50px;padding:8px 14px;border-bottom:1px solid #303731;background:#181e19}
  header>div{display:grid;min-width:0;gap:2px} header strong{font-size:13px} header small{overflow:hidden;max-width:480px;color:#7d877e;font-size:9px;text-overflow:ellipsis;white-space:nowrap}
  header button{border:0;color:#aeb7af;background:transparent;font-size:22px}
  .subtitle-regenerate-content{display:grid;gap:14px;padding:16px}.subtitle-regenerate-content>p{margin:0;color:#8f9990;font-size:10px;line-height:1.6}
  fieldset{margin:0;border:1px solid #303831;border-radius:10px;padding:10px;background:#111512}legend{padding:0 5px;color:#b9c2ba;font-size:10px}
  .subtitle-mode-options{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px}.subtitle-mode-options button{display:grid;gap:4px;min-height:62px;border:1px solid #384139;border-radius:8px;padding:9px;color:#cbd2cc;background:#1a201b;text-align:left}.subtitle-mode-options button small{color:#758077;font-size:8px;line-height:1.4}.subtitle-mode-options button strong{font-size:10px}.subtitle-mode-options button.active{border-color:#789757;color:#e8ffd8;background:#26301f;box-shadow:0 0 0 2px #a8e56d0c}.subtitle-mode-options button.active small{color:#a8bf95}
  .subtitle-format-options{display:flex;flex-wrap:wrap;gap:6px}.subtitle-format-options button{border:1px solid #384139;border-radius:7px;padding:6px 9px;color:#8c968e;background:#1a201b;font-size:9px}.subtitle-format-options button.active{border-color:#789757;color:#dfffc5;background:#26301f}.format-warning{display:block;margin-top:7px;color:#eaa0a5;font-size:9px}
  footer{display:grid;grid-template-columns:1fr 3fr;gap:8px;padding:10px 14px;border-top:1px solid #303731;background:#181e19}footer button{min-height:34px;border:1px solid #3b453c;border-radius:8px;color:#b7c0b8;background:#202621;font-size:10px}footer .primary{border-color:#91b36f;color:#17220f;background:#b7ed75;font-weight:750}button:disabled{cursor:not-allowed;opacity:.5}
  :global(html[data-theme="light"]) .subtitle-regenerate-backdrop{background:#1c241d66}
  :global(html[data-theme="light"]) .subtitle-regenerate-modal{border-color:#cbd4cc;color:#263028;background:#fff;box-shadow:0 24px 70px #3545382e}
  :global(html[data-theme="light"]) header,:global(html[data-theme="light"]) footer{border-color:#d5ddd6;background:#f7f9f7}
  :global(html[data-theme="light"]) header small,:global(html[data-theme="light"]) .subtitle-regenerate-content>p{color:#68746b}
  :global(html[data-theme="light"]) header button{color:#59665d}
  :global(html[data-theme="light"]) fieldset{border-color:#d2dad3;background:#f6f8f6}
  :global(html[data-theme="light"]) legend{color:#465249}
  :global(html[data-theme="light"]) .subtitle-mode-options button,:global(html[data-theme="light"]) .subtitle-format-options button{border-color:#cbd3cc;color:#59665d;background:#fff}
  :global(html[data-theme="light"]) .subtitle-mode-options button small{color:#758078}
  :global(html[data-theme="light"]) .subtitle-mode-options button.active,:global(html[data-theme="light"]) .subtitle-format-options button.active{border-color:#7d9b65;color:#304724;background:#dfead8}
  :global(html[data-theme="light"]) .subtitle-mode-options button.active small{color:#526a47}
  :global(html[data-theme="light"]) footer button{border-color:#c7d0c9;color:#465249;background:#fff}
  :global(html[data-theme="light"]) footer .primary{border-color:#789757;color:#203018;background:#b7ed75}
  @media(max-width:560px){.subtitle-regenerate-backdrop{padding:6px}.subtitle-mode-options{grid-template-columns:1fr}.subtitle-mode-options button{min-height:50px}.subtitle-regenerate-content{padding:12px}}
</style>
