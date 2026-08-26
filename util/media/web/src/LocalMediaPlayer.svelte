<script>
  import { onMount } from 'svelte'
  import Artplayer from 'artplayer'

  export let src = ''
  export let poster = ''
  export let autoplay = false
  export let captionSrc = ''
  export let captionLabel = '자막'
  export let thumbnails = null

  let container

  const subtitleColors = [
    { value: '#ffffff', label: '흰색' },
    { value: '#ffe36e', label: '노란색' },
    { value: '#7fe7ff', label: '하늘색' },
    { value: '#9ff28a', label: '연두색' },
    { value: '#111111', label: '검정색' }
  ]
  const subtitleBackgrounds = [
    { value: 'transparent', label: '없음' },
    { value: 'rgba(0, 0, 0, 0.35)', label: '옅게' },
    { value: 'rgba(0, 0, 0, 0.62)', label: '보통' },
    { value: 'rgba(0, 0, 0, 0.85)', label: '진하게' }
  ]

  function savedChoice(key, options, fallback) {
    const saved = localStorage.getItem(key)
    return options.some((option) => option.value === saved) ? saved : fallback
  }

  function optionLabel(options, value) {
    return options.find((option) => option.value === value)?.label || options[0].label
  }

  function subtitleShadow(color) {
    return color === '#111111'
      ? '0 1px 3px #fff, 0 1px 7px #fff'
      : '0 1px 4px #000, 0 1px 8px #000'
  }

  function setupFullscreenGuard(art) {
    const player = art.template?.$player
    const video = art.video
    if (!player || !video) return () => {}

    let locked = false
    let repeated = false
    let unlockTimer = 0
    let destroyed = false

    function recoverFrame(force = false) {
      if (destroyed) return
      requestAnimationFrame(() => requestAnimationFrame(() => {
        if (destroyed) return
        // Recalculate Artplayer and the browser compositor after the native
        // fullscreen surface has moved between DOM/layout contexts.
        player.getBoundingClientRect()
        art.emit('resize')
        window.dispatchEvent(new Event('resize'))
        video.style.transform = 'translateZ(0)'
        if (force && Number.isFinite(video.currentTime)) {
          try { video.currentTime = video.currentTime } catch {}
        }
      }))
    }

    function lockTransition() {
      locked = true
      player.classList.add('fullscreen-transition')
      clearTimeout(unlockTimer)
      unlockTimer = window.setTimeout(() => {
        locked = false
        player.classList.remove('fullscreen-transition')
        recoverFrame(repeated)
        repeated = false
      }, 900)
    }

    function waitForNativeExit() {
      if (!art.fullscreen) return Promise.resolve()
      return new Promise((resolve) => {
        let settled = false
        const finish = () => {
          if (settled) return
          settled = true
          art.off('fullscreen', changed)
          clearTimeout(timeout)
          resolve()
        }
        const changed = (state) => { if (!state) finish() }
        const timeout = window.setTimeout(finish, 800)
        art.on('fullscreen', changed)
        art.fullscreen = false
      })
    }

    async function switchMode(mode) {
      if (mode === 'fullscreen') {
        const entering = !art.fullscreen
        if (entering && art.fullscreenWeb) art.fullscreenWeb = false
        art.fullscreen = entering
        return
      }

      const entering = !art.fullscreenWeb
      if (!entering) {
        art.fullscreenWeb = false
        return
      }
      await waitForNativeExit()
      if (!destroyed) art.fullscreenWeb = true
    }

    function interceptFullscreenClick(event) {
      const control = event.target.closest?.('.art-control-fullscreen, .art-control-fullscreenWeb')
      if (!control || !player.contains(control)) return
      event.preventDefault()
      event.stopImmediatePropagation()
      if (locked) {
        repeated = true
        return
      }
      lockTransition()
      void switchMode(control.classList.contains('art-control-fullscreenWeb') ? 'fullscreenWeb' : 'fullscreen')
    }

    function blockRapidDoubleClick(event) {
      if (!locked) return
      event.preventDefault()
      event.stopImmediatePropagation()
      repeated = true
    }

    function fullscreenChanged() {
      recoverFrame(false)
    }

    player.addEventListener('click', interceptFullscreenClick, true)
    player.addEventListener('dblclick', blockRapidDoubleClick, true)
    art.on('fullscreen', fullscreenChanged)
    art.on('fullscreenWeb', fullscreenChanged)

    return () => {
      destroyed = true
      clearTimeout(unlockTimer)
      player.removeEventListener('click', interceptFullscreenClick, true)
      player.removeEventListener('dblclick', blockRapidDoubleClick, true)
      art.off('fullscreen', fullscreenChanged)
      art.off('fullscreenWeb', fullscreenChanged)
      player.classList.remove('fullscreen-transition')
    }
  }

  const korean = {
    'Video Info': '영상 정보', Close: '닫기', 'Video Load Failed': '영상을 불러오지 못했습니다.',
    Volume: '음량', Play: '재생', Pause: '일시정지', Rate: '속도', Mute: '음소거',
    'Video Flip': '화면 뒤집기', Horizontal: '좌우', Vertical: '상하', Reconnect: '다시 연결',
    'Show Setting': '설정 열기', 'Hide Setting': '설정 닫기', Screenshot: '화면 저장',
    'Play Speed': '재생 속도', 'Aspect Ratio': '화면 비율', Default: '기본', Normal: '보통',
    Open: '열기', 'Switch Subtitle': '자막 전환', Fullscreen: '전체 화면',
    'Exit Fullscreen': '전체 화면 종료', 'Web Fullscreen': '창 전체 화면',
    'Exit Web Fullscreen': '창 전체 화면 종료', 'PIP Mode': '화면 속 화면',
    'Exit PIP Mode': '화면 속 화면 종료', 'Subtitle Offset': '자막 싱크',
    'Last Seen': '이어서 보기', 'Jump Play': '이어서 재생'
  }

  onMount(() => {
    if (!container || !src) return undefined

    const savedSubtitleValue = localStorage.getItem('spark-media-subtitle-size')
    const savedSubtitleSize = savedSubtitleValue === null ? Number.NaN : Number(savedSubtitleValue)
    const subtitleSize = Number.isFinite(savedSubtitleSize)
      ? Math.min(48, Math.max(12, savedSubtitleSize))
      : 20
    const subtitleColor = savedChoice('spark-media-subtitle-color', subtitleColors, '#ffffff')
    const subtitleBackground = savedChoice('spark-media-subtitle-background', subtitleBackgrounds, 'rgba(0, 0, 0, 0.62)')

    // A media modal must own the only live Artplayer instance. This also
    // clears an instance left behind by a rapid modal switch before Svelte's
    // normal component cleanup has completed.
    for (const previous of [...Artplayer.instances]) {
      try { previous.pause() } catch {}
      try { previous.destroy(true) } catch {}
    }

    const art = new Artplayer({
      container,
      url: src,
      ...(poster ? { poster } : {}),
      theme: '#b7ed75',
      lang: 'ko',
      i18n: { ko: korean },
      autoplay,
      mutex: true,
      backdrop: true,
      playsInline: true,
      lock: true,
      gesture: true,
      fastForward: true,
      autoOrientation: true,
      playbackRate: true,
      aspectRatio: true,
      screenshot: true,
      setting: true,
      settings: captionSrc ? [
        {
          name: 'subtitle-size',
          html: '자막 크기',
          tooltip: `${subtitleSize}px`,
          range: [subtitleSize, 12, 48, 1],
          onChange(item) {
            const size = item.range[0]
            this.subtitle.style('fontSize', `${size}px`)
            localStorage.setItem('spark-media-subtitle-size', String(size))
            return `${size}px`
          },
          onRange(item) {
            const size = item.range[0]
            this.subtitle.style('fontSize', `${size}px`)
            localStorage.setItem('spark-media-subtitle-size', String(size))
            return `${size}px`
          }
        },
        {
          name: 'subtitle-color',
          html: '자막 글자색',
          tooltip: optionLabel(subtitleColors, subtitleColor),
          selector: subtitleColors.map((option) => ({ ...option, html: option.label, default: option.value === subtitleColor })),
          onSelect(item) {
            this.subtitle.style({ color: item.value, textShadow: subtitleShadow(item.value) })
            localStorage.setItem('spark-media-subtitle-color', item.value)
            return item.html
          }
        },
        {
          name: 'subtitle-background',
          html: '자막 배경',
          tooltip: optionLabel(subtitleBackgrounds, subtitleBackground),
          selector: subtitleBackgrounds.map((option) => ({ ...option, html: option.label, default: option.value === subtitleBackground })),
          onSelect(item) {
            this.template.$player.style.setProperty('--spark-subtitle-background', item.value)
            localStorage.setItem('spark-media-subtitle-background', item.value)
            return item.html
          }
        }
      ] : [],
      pip: true,
      fullscreen: true,
      fullscreenWeb: true,
      subtitleOffset: Boolean(captionSrc),
      miniProgressBar: true,
      ...(thumbnails ? { thumbnails } : {}),
      ...(captionSrc ? { subtitle: {
        url: captionSrc,
        name: captionLabel,
        type: 'vtt',
        encoding: 'utf-8',
        style: { color: subtitleColor, fontSize: `${subtitleSize}px`, textShadow: subtitleShadow(subtitleColor) }
      } } : {})
    })
    if (captionSrc) art.template.$player.style.setProperty('--spark-subtitle-background', subtitleBackground)
    const destroyFullscreenGuard = setupFullscreenGuard(art)
    return () => {
      destroyFullscreenGuard()
      try { art.pause() } catch {}
      try {
        art.video.pause()
        art.video.removeAttribute('src')
        art.video.load()
      } catch {}
      art.destroy(true)
    }
  })
</script>

<div class="local-media-player" bind:this={container}></div>

<style>
  .local-media-player { width: 100%; height: 100%; min-height: 180px; background: #000; }
  .local-media-player :global(.fullscreen-transition .art-control-fullscreen),
  .local-media-player :global(.fullscreen-transition .art-control-fullscreenWeb) {
    opacity: .35 !important;
  }
  .local-media-player :global(.art-subtitle-line) {
    padding: .12em .38em;
    border-radius: .18em;
    background: var(--spark-subtitle-background, rgba(0, 0, 0, .62));
    box-decoration-break: clone;
    -webkit-box-decoration-break: clone;
  }
</style>
