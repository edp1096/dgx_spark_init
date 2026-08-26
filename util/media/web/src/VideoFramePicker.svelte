<script>
  import { onDestroy } from 'svelte'
  import { lockModalScroll } from './modalScroll.js'

  export let source = null
  export let onClose = () => {}
  export let onUse = () => {}

  let player
  let currentTime = 0
  let duration = 0
  let extracting = ''
  let message = ''
  let error = ''
  let releaseScroll = null

  function unlockScroll() {
    releaseScroll?.()
    releaseScroll = null
  }

  function closeModal() {
    try { player?.pause() } catch {}
    onClose()
  }

  function loaded() {
    duration = Number(player?.duration) || Number(source?.duration) || 0
    currentTime = Number(player?.currentTime) || 0
  }

  function seek(value) {
    const next = Math.min(Math.max(0, Number(value) || 0), duration || Number.MAX_SAFE_INTEGER)
    currentTime = next
    if (player) player.currentTime = next
  }

  async function useFrame(target) {
    if (!source?.jobID || extracting) return
    extracting = target
    message = ''
    error = ''
    try {
      const response = await fetch(`/api/jobs/${encodeURIComponent(source.jobID)}/frame?time=${encodeURIComponent(currentTime.toFixed(6))}`)
      if (!response.ok) throw new Error((await response.text()) || `HTTP ${response.status}`)
      const blob = await response.blob()
      const label = target === 'start' ? 'start' : target === 'end' ? 'end' : `keyframe-${currentTime.toFixed(3)}`
      const file = new File([blob], `${label}.jpg`, { type: blob.type || 'image/jpeg' })
      await onUse(file, target, currentTime, duration)
      message = target === 'start' ? '시작 이미지로 보냈습니다.' : target === 'end' ? '마지막 이미지로 보냈습니다.' : `${currentTime.toFixed(2)}초를 키프레임으로 추가했습니다.`
    } catch (cause) {
      error = cause.message || '프레임을 추출하지 못했습니다.'
    } finally {
      extracting = ''
    }
  }

  $: {
    if (source && !releaseScroll) releaseScroll = lockModalScroll()
    else if (!source) unlockScroll()
  }

  onDestroy(unlockScroll)
</script>

<svelte:window onkeydown={(event) => { if (source && event.key === 'Escape') closeModal() }} />

{#if source}
  <div class="frame-picker-backdrop" role="presentation" onclick={(event) => { if (event.target === event.currentTarget) closeModal() }}>
    <section class="frame-picker" role="dialog" aria-modal="true" aria-label="영상 장면 선택">
      <header><div><strong>영상 장면 선택</strong><small title={source.title}>{source.title}</small></div><button type="button" aria-label="닫기" onclick={closeModal}>×</button></header>
      <div class="frame-stage">
        <!-- svelte-ignore a11y_media_has_caption -->
        <video bind:this={player} src={source.src} controls preload="metadata" playsinline onloadedmetadata={loaded} ontimeupdate={() => currentTime = Number(player?.currentTime) || 0}></video>
      </div>
      <div class="frame-controls">
        <input aria-label="장면 위치" type="range" min="0" max={Math.max(0.001, duration)} step="any" value={currentTime} oninput={(event) => seek(event.currentTarget.value)}>
        <label>위치 (초)<input type="number" min="0" max={duration || undefined} step="any" value={currentTime.toFixed(3)} onchange={(event) => seek(event.currentTarget.value)}></label>
        <span>{currentTime.toFixed(2)} / {duration ? duration.toFixed(2) : '—'}초</span>
      </div>
      {#if message}<p class="frame-message">{message}</p>{/if}
      {#if error}<p class="frame-error">{error}</p>{/if}
      <footer>
        <button type="button" disabled={Boolean(extracting)} onclick={() => useFrame('start')}>{extracting === 'start' ? '추출 중…' : '시작 이미지로'}</button>
        <button type="button" disabled={Boolean(extracting)} onclick={() => useFrame('keyframe')}>{extracting === 'keyframe' ? '추출 중…' : '키프레임 추가'}</button>
        <button type="button" disabled={Boolean(extracting)} onclick={() => useFrame('end')}>{extracting === 'end' ? '추출 중…' : '마지막 이미지로'}</button>
        <button type="button" class="close" onclick={closeModal}>닫기</button>
      </footer>
    </section>
  </div>
{/if}

<style>
  .frame-picker-backdrop{position:fixed;z-index:110;inset:0;display:grid;place-items:center;padding:18px;background:#050705df;backdrop-filter:blur(8px)}
  .frame-picker{display:grid;overflow:hidden;width:min(980px,96vw);max-height:94dvh;border:1px solid #3b463c;border-radius:14px;background:#151a16;box-shadow:0 24px 80px #000b}
  header{display:flex;align-items:center;min-height:50px;padding:8px 14px;border-bottom:1px solid #303731;background:#181e19}
  header div{display:grid;gap:2px;min-width:0} header strong{color:#edf2eb;font-size:14px} header small{overflow:hidden;color:#828d83;font-size:10px;text-overflow:ellipsis;white-space:nowrap}
  header button{margin-left:auto;border:0;color:#aeb7af;background:transparent;font-size:24px;cursor:pointer}
  .frame-stage{display:grid;place-items:center;min-height:220px;max-height:65dvh;background:#050605}.frame-stage video{display:block;width:100%;max-height:65dvh;object-fit:contain}
  .frame-controls{display:grid;grid-template-columns:minmax(0,1fr) 130px auto;align-items:end;gap:10px;padding:10px 12px;border-top:1px solid #252c26}
  .frame-controls>input{width:100%;margin-bottom:7px}.frame-controls label{display:grid;gap:4px;color:#8e998f;font-size:10px}.frame-controls label input{min-height:34px;border:1px solid #354036;border-radius:7px;padding:6px 8px;color:#e4ebe5;background:#101411}.frame-controls span{padding-bottom:8px;color:#bac4bb;font-size:11px;white-space:nowrap}
  .frame-message,.frame-error{margin:0;padding:0 12px 8px;font-size:11px}.frame-message{color:#bada9e}.frame-error{color:#ff9c93}
  footer{display:flex;gap:7px;padding:9px 12px;border-top:1px solid #303731;background:#181e19}footer button{min-height:32px;border:1px solid #465247;border-radius:7px;padding:5px 11px;color:#d8e8d2;background:#253026;font-size:11px;cursor:pointer}footer button:disabled{opacity:.45;cursor:not-allowed}footer .close{margin-left:auto;color:#b9c4ba;background:#202621}
  @media(max-width:700px){.frame-picker-backdrop{padding:5px}.frame-stage,.frame-stage video{max-height:52dvh}.frame-controls{grid-template-columns:1fr 105px}.frame-controls>span{display:none}footer{display:grid;grid-template-columns:repeat(3,1fr)}footer button{padding:4px 5px;font-size:10px}footer .close{grid-column:1/-1;margin-left:0}}
</style>
