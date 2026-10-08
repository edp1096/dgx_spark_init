<script>
  import { onDestroy } from 'svelte'
  import { lockModalScroll } from './modalScroll.js'

  export let audio = null
  export let onClose = () => {}
  export let onA2V = () => {}

  let releaseScroll = null
  let player

  function closeModal() {
    try { player?.pause() } catch {}
    onClose()
  }

  function unlockScroll() {
    releaseScroll?.()
    releaseScroll = null
  }

  $: {
    if (audio && !releaseScroll) releaseScroll = lockModalScroll()
    else if (!audio) unlockScroll()
  }

  onDestroy(unlockScroll)
</script>

<svelte:window onkeydown={(event) => { if (audio && event.key === 'Escape') closeModal() }} />

{#if audio}
  <div class="audio-modal-backdrop" role="presentation" onclick={(event) => { if (event.target === event.currentTarget) closeModal() }}>
    <div class="audio-modal" role="dialog" aria-modal="true" aria-label="생성 음성 크게 보기">
      <header><div><strong>생성 음성</strong>{#if audio.detail}<small title={audio.detail}>{audio.detail}</small>{/if}</div><button type="button" aria-label="닫기" onclick={closeModal}>×</button></header>
      <div class="audio-modal-content">
        <audio bind:this={player} controls preload="metadata" src={audio.src}></audio>
        <section><strong>읽은 문장</strong><p>{audio.prompt}</p></section>
        {#if audio.instructions}<section><strong>연기 지시</strong><p>{audio.instructions}</p></section>{/if}
      </div>
      <footer><a href={audio.src} target="_blank" rel="noreferrer">원본 파일 열기</a>{#if audio.jobID}<button type="button" onclick={() => onA2V(audio.jobID)}>영상 생성</button>{/if}<button type="button" onclick={closeModal}>닫기</button></footer>
    </div>
  </div>
{/if}

<style>
  .audio-modal-backdrop { position:fixed; z-index:100; inset:0; display:grid; place-items:center; padding:20px; background:#050705df; backdrop-filter:blur(8px); overscroll-behavior:contain; }
  .audio-modal { display:grid; overflow:hidden; width:min(760px,96vw); max-height:94vh; border:1px solid #3b463c; border-radius:14px; background:#151a16; box-shadow:0 24px 80px #000b; }
  header { position:static; display:flex; align-items:center; justify-content:space-between; width:100%; height:auto; min-height:58px; padding:12px 16px; border-bottom:1px solid #303731; background:#181e19; }
  header div { display:grid; gap:3px; min-width:0; }
  header strong { color:#edf2eb; font-size:14px; }
  header small { overflow:hidden; max-width:min(760px,70vw); color:#7d877e; font-size:10px; text-overflow:ellipsis; white-space:nowrap; }
  header button { margin-left:auto; border:0; color:#aeb7af; background:transparent; font-size:24px; cursor:pointer; }
  .audio-modal-content { display:grid; gap:10px; padding:14px; background:#090b0a; }
  audio { width:100%; }
  .audio-modal section { padding:10px 12px; border:1px solid #2b332c; border-radius:8px; background:#111612; }
  .audio-modal section strong { color:#aeb8af; font-size:10px; }
  .audio-modal section p { margin:7px 0 0; color:#dce3dd; font-size:11px; line-height:1.6; white-space:pre-wrap; }
  footer { display:flex; align-items:center; justify-content:space-between; gap:8px; padding:11px 14px; border-top:1px solid #303731; background:#181e19; }
  footer a, footer button { border:1px solid #3a443b; border-radius:7px; padding:7px 10px; color:#cdecaa; background:#202721; font-size:10px; text-decoration:none; }
  @media(max-width:700px) { .audio-modal-backdrop{padding:6px} }
</style>
