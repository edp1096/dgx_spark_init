<script>
  import { onDestroy } from 'svelte'
  import { api } from './api.js'
  import { lockModalScroll } from './modalScroll.js'

  export let open = false
  export let title = 'URL 이미지 가져오기'
  export let append = false
  export let onImport = () => {}
  export let onClose = () => {}
  export let zIndex = 82

  let url = ''
  let loading = false
  let error = ''
  let releaseScroll = null
  let previousOpen = false

  $: {
    if (open && !previousOpen) {
      url = ''
      error = ''
    }
    previousOpen = open
  }

  $: {
    if (open && !releaseScroll) releaseScroll = lockModalScroll()
    else if (!open && releaseScroll) {
      releaseScroll()
      releaseScroll = null
    }
  }

  onDestroy(() => releaseScroll?.())

  async function importImage() {
    const source = url.trim()
    if (!source || loading) return
    loading = true
    error = ''
    try {
      const result = await api.remoteImage(source)
      const file = new File([result.blob], result.filename, { type: result.blob.type || 'image/png' })
      await onImport(file)
      onClose()
    } catch (cause) {
      error = cause.message || '이미지를 가져오지 못했습니다.'
    } finally {
      loading = false
    }
  }
</script>

<svelte:window onkeydown={(event) => { if (open && event.key === 'Escape' && !loading) onClose() }} />

{#if open}
  <div class="remote-image-backdrop" style:z-index={zIndex} role="presentation" onclick={(event) => { if (event.target === event.currentTarget && !loading) onClose() }}>
    <div class="remote-image-modal" role="dialog" aria-modal="true" aria-label={title}>
      <header><div><strong>{title}</strong><small>웹 이미지 주소를 서버에서 내려받아 업로드 파일처럼 사용합니다.</small></div><button type="button" aria-label="닫기" disabled={loading} onclick={onClose}>×</button></header>
      <form onsubmit={(event) => { event.preventDefault(); importImage() }}>
        <label>이미지 URL<input type="url" bind:value={url} placeholder="https://example.com/image.jpg" autocomplete="off" required autofocus></label>
        <small>PNG · JPEG · WebP · GIF, 최대 32MB{append ? ' · 가져온 이미지는 기존 선택에 추가됩니다.' : ''}</small>
        {#if error}<p>{error}</p>{/if}
        <div><button type="button" disabled={loading} onclick={onClose}>취소</button><button type="submit" class="primary" disabled={loading || !url.trim()}>{loading ? '가져오는 중…' : '가져오기'}</button></div>
      </form>
    </div>
  </div>
{/if}

<style>
  .remote-image-backdrop { position:fixed; z-index:82; inset:0; display:grid; place-items:center; padding:16px; background:#050708df; backdrop-filter:blur(8px); }
  .remote-image-modal { width:min(560px,96vw); overflow:hidden; border:1px solid #465058; border-radius:14px; background:#12171b; box-shadow:0 24px 80px #000c; }
  header { position:static; display:flex; width:100%; height:auto; min-height:54px; align-items:center; justify-content:space-between; gap:12px; padding:9px 13px; border-bottom:1px solid #2b3339; background:#12171b; }
  header > div { display:grid; gap:2px; }
  header strong { color:#e5e9eb; font-size:13px; }
  header small { color:#7c878e; font-size:9px; }
  header button { width:30px; height:30px; padding:0; border:0; color:#aeb7bc; background:transparent; font-size:20px; }
  form { display:grid; gap:10px; padding:16px; }
  label { display:grid; gap:7px; margin:0; color:#aeb7bc; font-size:11px; }
  input { width:100%; padding:11px 12px; border:1px solid #384249; border-radius:9px; color:#eef1f2; background:#0d1114; outline:none; }
  input:focus { border-color:#718d57; box-shadow:0 0 0 3px #9dcc7020; }
  form > small { color:#768188; font-size:9px; }
  p { margin:0; padding:9px 10px; border-radius:8px; color:#f2b4b7; background:#392126; font-size:10px; overflow-wrap:anywhere; }
  form > div { display:flex; justify-content:flex-end; gap:7px; margin-top:4px; }
  form button { min-width:88px; padding:9px 12px; border:1px solid #3b444a; border-radius:8px; color:#c6cdd1; background:#1a2024; }
  form button.primary { border:0; color:#17200f; background:#b7ed75; font-weight:750; }
  button:disabled { opacity:.45; cursor:default; }
  @media(max-width:600px) {
    .remote-image-backdrop { align-items:end; padding:8px; }
    .remote-image-modal { width:100%; margin-bottom:max(0px,env(safe-area-inset-bottom)); }
    header small { display:none; }
  }
</style>
