<script>
  import { onDestroy } from 'svelte'
  import { lockModalScroll } from './modalScroll.js'

  export let open = false
  export let source = ''
  export let existingMask = ''
  export let onApply = () => {}
  export let onClose = () => {}
  export let title = '수정 영역 칠하기'
  export let description = '빨간 영역을 Krea가 새로 생성합니다.'
  export let outputName = 'krea-mask.png'

  let canvas
  let loadedSource = ''
  let loadedMask = ''
  let width = 0
  let height = 0
  let tool = 'brush'
  let brushSize = 6
  let drawing = false
  let lastPoint = null
  let pointerTransform = null
  let activePointerId = null
  let undoStack = []
  let preset = ''
  let shape = { x: 30, y: 20, w: 40, h: 60 }
  let shapeAction = ''
  let shapeStart = null
  let releaseScroll = null

  $: {
    if (open && !releaseScroll) releaseScroll = lockModalScroll()
    else if (!open && releaseScroll) {
      releaseScroll()
      releaseScroll = null
    }
  }

  onDestroy(() => releaseScroll?.())

  $: if (open && source && canvas && (source !== loadedSource || existingMask !== loadedMask)) loadCanvas(source, existingMask, canvas)

  // The modal's {#if} destroys its canvas when it closes. Reinitialize every
  // newly-created canvas even when the source and mask URLs did not change.
  function initializeCanvas(node) {
    canvas = node
    loadCanvas(source, existingMask, node)
    return {
      destroy() {
        if (canvas === node) canvas = null
        drawing = false
        lastPoint = null
        pointerTransform = null
        activePointerId = null
      }
    }
  }

  function loadImage(url) {
    return new Promise((resolve, reject) => {
      const image = new Image()
      image.onload = () => resolve(image)
      image.onerror = reject
      image.src = url
    })
  }

  async function loadCanvas(sourceURL, maskURL, targetCanvas = canvas) {
    loadedSource = sourceURL
    loadedMask = maskURL
    undoStack = []
    try {
      const image = await loadImage(sourceURL)
      width = image.naturalWidth
      height = image.naturalHeight
      await Promise.resolve()
      if (!targetCanvas?.isConnected) return
      targetCanvas.width = width
      targetCanvas.height = height
      const context = targetCanvas.getContext('2d')
      context.clearRect(0, 0, width, height)
      if (maskURL) {
        const mask = await loadImage(maskURL)
        const scratch = document.createElement('canvas')
        scratch.width = width
        scratch.height = height
        const scratchContext = scratch.getContext('2d')
        scratchContext.drawImage(mask, 0, 0, width, height)
        const pixels = scratchContext.getImageData(0, 0, width, height)
        for (let index = 0; index < pixels.data.length; index += 4) {
          const intensity = Math.max(pixels.data[index], pixels.data[index + 1], pixels.data[index + 2])
          pixels.data[index] = 255
          pixels.data[index + 1] = 70
          pixels.data[index + 2] = 60
          pixels.data[index + 3] = intensity > 127 ? 125 : 0
        }
        context.putImageData(pixels, 0, 0)
      }
    } catch {
      loadedSource = ''
    }
  }

  function canvasTransform() {
    const bounds = canvas.getBoundingClientRect()
    return {
      left: bounds.left,
      top: bounds.top,
      scaleX: width / Math.max(1, bounds.width),
      scaleY: height / Math.max(1, bounds.height)
    }
  }

  function pointFromEvent(event, transform = pointerTransform || canvasTransform()) {
    return {
      x: Math.max(0, Math.min(canvas.width, (event.clientX - transform.left) * transform.scaleX)),
      y: Math.max(0, Math.min(canvas.height, (event.clientY - transform.top) * transform.scaleY))
    }
  }

  function remember() {
    if (!canvas || !width || !height) return
    undoStack = [...undoStack.slice(-11), canvas.getContext('2d').getImageData(0, 0, width, height)]
  }

  function drawLine(from, to) {
    const context = canvas.getContext('2d')
    context.save()
    context.globalCompositeOperation = tool === 'eraser' ? 'destination-out' : 'source-over'
    context.strokeStyle = tool === 'eraser' ? 'rgba(0,0,0,1)' : 'rgba(255,70,60,.72)'
    context.lineWidth = Math.max(2, Math.min(canvas.width, canvas.height) * brushSize / 100)
    context.lineCap = 'round'
    context.lineJoin = 'round'
    context.beginPath()
    context.moveTo(from.x, from.y)
    context.lineTo(to.x, to.y)
    context.stroke()
    context.restore()
  }

  function startDrawing(event) {
    if ((event.pointerType === 'mouse' && event.button !== 0) || activePointerId !== null) return
    event.preventDefault()
    remember()
    drawing = true
    activePointerId = event.pointerId
    pointerTransform = canvasTransform()
    lastPoint = pointFromEvent(event, pointerTransform)
    drawLine(lastPoint, lastPoint)
    canvas.setPointerCapture(event.pointerId)
  }

  function continueDrawing(event) {
    if (!drawing || event.pointerId !== activePointerId) return
    event.preventDefault()
    const next = pointFromEvent(event)
    drawLine(lastPoint, next)
    lastPoint = next
  }

  function stopDrawing(event) {
    if (event.pointerId !== activePointerId) return
    event.preventDefault()
    drawing = false
    lastPoint = null
    pointerTransform = null
    activePointerId = null
    if (canvas?.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId)
  }

  function undo() {
    const previous = undoStack.at(-1)
    if (!previous) return
    canvas.getContext('2d').putImageData(previous, 0, 0)
    undoStack = undoStack.slice(0, -1)
  }

  function clearMask() {
    remember()
    canvas.getContext('2d').clearRect(0, 0, width, height)
  }

  function choosePreset(value) {
    preset = value
    if (value === 'portrait') shape = { x: 32, y: 12, w: 36, h: 76 }
    else if (value === 'face') shape = { x: 39, y: 10, w: 22, h: 34 }
    else if (value === 'wide') shape = { x: 15, y: 30, w: 70, h: 40 }
    else if (value) shape = { x: 30, y: 25, w: 40, h: 50 }
  }

  function startShape(event, action) {
    event.preventDefault()
    event.stopPropagation()
    shapeAction = action
    shapeStart = { clientX: event.clientX, clientY: event.clientY, ...shape }
    event.currentTarget.setPointerCapture(event.pointerId)
  }

  function moveShape(event) {
    if (!shapeAction || !shapeStart) return
    const bounds = canvas.getBoundingClientRect()
    const dx = (event.clientX - shapeStart.clientX) * 100 / bounds.width
    const dy = (event.clientY - shapeStart.clientY) * 100 / bounds.height
    if (shapeAction === 'move') {
      shape = { ...shape, x: Math.max(0, Math.min(100 - shape.w, shapeStart.x + dx)), y: Math.max(0, Math.min(100 - shape.h, shapeStart.y + dy)) }
    } else {
      shape = { ...shape, w: Math.max(5, Math.min(100 - shape.x, shapeStart.w + dx)), h: Math.max(5, Math.min(100 - shape.y, shapeStart.h + dy)) }
    }
  }

  function stopShape(event) {
    shapeAction = ''
    shapeStart = null
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId)
  }

  function applyPreset() {
    if (!preset) return
    remember()
    const context = canvas.getContext('2d')
    const x = width * shape.x / 100
    const y = height * shape.y / 100
    const w = width * shape.w / 100
    const h = height * shape.h / 100
    context.save()
    context.fillStyle = 'rgba(255,70,60,.72)'
    context.beginPath()
    if (preset === 'face') context.ellipse(x + w / 2, y + h / 2, w / 2, h / 2, 0, 0, Math.PI * 2)
    else context.rect(x, y, w, h)
    context.fill()
    context.restore()
    preset = ''
  }

  function saveMask() {
    if (!canvas || !width || !height) return
    if (preset) applyPreset()
    const overlay = canvas.getContext('2d').getImageData(0, 0, width, height)
    const output = document.createElement('canvas')
    output.width = width
    output.height = height
    const context = output.getContext('2d')
    const mask = context.createImageData(width, height)
    for (let index = 0; index < mask.data.length; index += 4) {
      const selected = overlay.data[index + 3] > 20 ? 255 : 0
      mask.data[index] = selected
      mask.data[index + 1] = selected
      mask.data[index + 2] = selected
      mask.data[index + 3] = 255
    }
    context.putImageData(mask, 0, 0)
    output.toBlob((blob) => {
      if (!blob) return
      onApply(new File([blob], outputName, { type: 'image/png' }))
    }, 'image/png')
  }
</script>

<svelte:window onkeydown={(event) => { if (open && event.key === 'Escape') onClose() }} />

{#if open}
  <div class="mask-editor-backdrop" role="presentation" onclick={(event) => { if (event.target === event.currentTarget) onClose() }}>
    <section class="mask-editor" role="dialog" aria-modal="true" aria-label="수정 마스크 편집기">
      <header><div><strong>{title}</strong><small>{description}</small></div><button type="button" aria-label="닫기" onclick={onClose}>×</button></header>
      <div class="mask-toolbar">
        <div class="segmented"><button type="button" class:active={tool === 'brush'} onclick={() => tool = 'brush'}>브러시</button><button type="button" class:active={tool === 'eraser'} onclick={() => tool = 'eraser'}>지우개</button></div>
        <label>크기 <input type="range" min="1" max="20" step="1" bind:value={brushSize}><b>{brushSize}%</b></label>
        <button type="button" onclick={undo} disabled={!undoStack.length}>실행 취소</button>
        <button type="button" onclick={clearMask}>전체 지우기</button>
      </div>
      <div class="mask-stage" style={`--mask-aspect:${(width || 1) / (height || 1)};aspect-ratio:${width || 1}/${height || 1}`}>
        {#if source}<img src={source} alt="마스크 원본">{/if}
        <canvas use:initializeCanvas onpointerdown={startDrawing} onpointermove={continueDrawing} onpointerup={stopDrawing} onpointercancel={stopDrawing} onlostpointercapture={stopDrawing}></canvas>
        {#if preset}
          <div class:ellipse={preset === 'face'} class="mask-preset-shape" style={`left:${shape.x}%;top:${shape.y}%;width:${shape.w}%;height:${shape.h}%`} onpointerdown={(event) => startShape(event, 'move')} onpointermove={moveShape} onpointerup={stopShape} onpointercancel={stopShape}>
            <span>이동</span><i onpointerdown={(event) => startShape(event, 'resize')} onpointermove={moveShape} onpointerup={stopShape} onpointercancel={stopShape}></i>
          </div>
        {/if}
      </div>
      <div class="mask-presets">
        <label>선택 프리셋<select value={preset} onchange={(event) => choosePreset(event.currentTarget.value)}><option value="">사용 안 함</option><option value="portrait">인물 전체</option><option value="face">얼굴·머리</option><option value="wide">가로 영역</option><option value="rectangle">사각 영역</option></select></label>
        <small>{preset ? '현재 선택은 저장할 때 자동 반영됩니다. 여러 영역을 만들 때만 오른쪽 버튼으로 누적하세요.' : '프리셋을 고르면 원본 위에 이동 가능한 선택 영역이 표시됩니다.'}</small>
        <button type="button" disabled={!preset} onclick={applyPreset}>현재 위치 추가·계속</button>
      </div>
      <footer><button type="button" onclick={onClose}>취소</button><button type="button" class="primary" onclick={saveMask}>이 마스크 사용</button></footer>
    </section>
  </div>
{/if}
