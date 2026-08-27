<script>
  import { onDestroy } from 'svelte'
  import { lockModalScroll } from './modalScroll.js'
  import LocalMediaPlayer from './LocalMediaPlayer.svelte'

  export let video = null
  export let onClose = () => {}
  export let onSelectFrames = () => {}
  export let onUpscale = () => {}
  export let onTranscribe = () => {}
  export let onLoadSettings = () => {}

  let releaseScroll = null
  let stage

  function stopStageMedia() {
    if (!stage) return
    for (const media of stage.querySelectorAll('video, audio')) {
      try { media.pause() } catch {}
    }
  }

  function closeModal() {
    stopStageMedia()
    onClose()
  }

  function unlockScroll() {
    releaseScroll?.()
    releaseScroll = null
  }

  $: {
    if (video && !releaseScroll) releaseScroll = lockModalScroll()
    else if (!video) unlockScroll()
  }

  onDestroy(unlockScroll)
</script>

<svelte:window onkeydown={(event) => { if (video && event.key === 'Escape') closeModal() }} />

{#if video}
  <div class="video-modal-backdrop" role="presentation" onclick={(event) => { if (event.target === event.currentTarget) closeModal() }}>
    <div class="video-modal" role="dialog" aria-modal="true" aria-label="영상 크게 보기">
      <header>
        <div><strong>{video.title || '생성 영상'}</strong>{#if video.detail}<small title={video.detail}>{video.detail}</small>{/if}</div>
        <button type="button" aria-label="닫기" onclick={closeModal}>×</button>
      </header>
      <div class="video-modal-stage" bind:this={stage}>
        <LocalMediaPlayer src={video.src} autoplay={false} thumbnails={video.thumbnails} captionSrc={video.captionSrc || ''} captionLabel={video.captionLabel || '자막'} />
      </div>
      {#if video.prompt}<p class="video-modal-prompt">{video.prompt}</p>{/if}
      <footer><a href={video.src} target="_blank" rel="noreferrer">원본 파일 열기</a>{#if video.jobID}{#if video.canLoadSettings}<button type="button" onclick={() => onLoadSettings(video.jobID)}>설정 불러오기</button>{/if}<button type="button" onclick={() => onSelectFrames(video.jobID)}>장면 선택</button><button type="button" onclick={() => onUpscale(video.jobID)}>업스케일</button><button type="button" onclick={() => onTranscribe(video.jobID)}>자막 생성</button>{/if}<button type="button" onclick={closeModal}>닫기</button></footer>
    </div>
  </div>
{/if}

<style>
  .video-modal-backdrop {
    position: fixed;
    z-index: 100;
    inset: 0;
    display: grid;
    place-items: center;
    padding: 20px;
    background: #050705df;
    backdrop-filter: blur(8px);
    overscroll-behavior: contain;
  }

  .video-modal {
    display: grid;
    overflow: hidden;
    width: min(1240px, 96vw);
    max-height: 94vh;
    border: 1px solid #3b463c;
    border-radius: 14px;
    background: #151a16;
    box-shadow: 0 24px 80px #000b;
  }

  header {
    position: static;
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: auto;
    min-height: 58px;
    padding: 12px 16px;
    border-bottom: 1px solid #303731;
    background: #181e19;
  }

  header div { display: grid; gap: 3px; min-width: 0; }
  header strong { color: #edf2eb; font-size: 14px; }
  header small { overflow: hidden; max-width: min(760px, 70vw); color: #7d877e; font-size: 10px; text-overflow: ellipsis; white-space: nowrap; }
  header button { margin-left: auto; border: 0; color: #aeb7af; background: transparent; font-size: 24px; cursor: pointer; }

  .video-modal-stage {
    display: grid;
    place-items: center;
    overflow: hidden;
    min-height: 220px;
    height: min(calc(94vh - 190px), 69.75vw);
    background: #090b0a;
  }

  .video-modal-prompt {
    overflow: auto;
    max-height: 84px;
    margin: 0;
    padding: 10px 14px;
    border-top: 1px solid #252b26;
    color: #b9c0ba;
    font-size: 11px;
    line-height: 1.5;
  }

  footer {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 8px;
    padding: 8px 12px;
    border-top: 1px solid #303731;
    background: #181e19;
  }

  footer a,
  footer button {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    box-sizing: border-box;
    min-height: 28px;
    border: 1px solid #3c463e;
    border-radius: 6px;
    padding: 4px 9px;
    color: #b9c4ba;
    background: #202621;
    font-size: 10px;
    line-height: 1;
    text-decoration: none;
    cursor: pointer;
  }

  footer a { margin-right: auto; color: #cdecaa; }
  footer a:hover,
  footer button:hover { border-color: #60705f; background: #293129; }

  @media (max-width: 700px) {
    .video-modal-backdrop { padding: 6px; }
    .video-modal-stage { height: min(calc(94dvh - 210px), 56.25vw); min-height: 190px; }
    footer { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); }
    footer a { grid-column: 1 / -1; margin-right: 0; }
    footer button { min-width: 0; padding-inline: 4px; font-size: 9px; }
  }
</style>
