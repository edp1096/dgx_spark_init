<script>
  import { onDestroy } from 'svelte'
  import { lockModalScroll } from './modalScroll.js'

  export let open = false
  export let source = ''
  export let onApply = () => {}
  export let onClose = () => {}
  export let preprocessed = false

  let canvas
  let original
  let width = 0
  let height = 0
  let threshold = 90
  let tool = 'white'
  let brushSize = 3
  let drawing = false
  let last = null
  let snapshots = []
  let label = ''
  let labelSize = 8
  let loadedKey = ''
  let releaseScroll = null

  $: {
    if (open && !releaseScroll) releaseScroll = lockModalScroll()
    else if (!open && releaseScroll) {
      releaseScroll()
      releaseScroll = null
    }
  }

  onDestroy(() => releaseScroll?.())

  $: if (open && source && canvas && loadedKey !== `${source}:${preprocessed}`) load(source)

  function loadImage(url) {
    return new Promise((resolve, reject) => { const image = new Image(); image.onload = () => resolve(image); image.onerror = reject; image.src = url })
  }

  async function load(url) {
    original = await loadImage(url)
    loadedKey = `${url}:${preprocessed}`
    width = original.naturalWidth; height = original.naturalHeight
    canvas.width = width; canvas.height = height
    if (preprocessed) canvas.getContext('2d').drawImage(original, 0, 0)
    else renderEdges()
  }

  function renderEdges() {
    if (!original || !canvas) return
    const scratch = document.createElement('canvas'); scratch.width = width; scratch.height = height
    const context = scratch.getContext('2d', { willReadFrequently: true }); context.drawImage(original, 0, 0)
    const pixels = context.getImageData(0, 0, width, height).data
    const gray = new Uint8Array(width * height)
    for (let i = 0, p = 0; i < pixels.length; i += 4, p++) gray[p] = pixels[i] * .299 + pixels[i + 1] * .587 + pixels[i + 2] * .114
    const output = canvas.getContext('2d').createImageData(width, height)
    for (let y = 1; y < height - 1; y++) for (let x = 1; x < width - 1; x++) {
      const p = y * width + x
      const gx = -gray[p-width-1] + gray[p-width+1] - 2*gray[p-1] + 2*gray[p+1] - gray[p+width-1] + gray[p+width+1]
      const gy = -gray[p-width-1] - 2*gray[p-width] - gray[p-width+1] + gray[p+width-1] + 2*gray[p+width] + gray[p+width+1]
      const value = Math.hypot(gx, gy) >= threshold ? 255 : 0
      const i = p * 4; output.data[i] = value; output.data[i+1] = value; output.data[i+2] = value; output.data[i+3] = 255
    }
    canvas.getContext('2d').putImageData(output, 0, 0); snapshots = []
  }

  function point(event) { const box = canvas.getBoundingClientRect(); return { x:(event.clientX-box.left)*width/box.width, y:(event.clientY-box.top)*height/box.height } }
  function remember() { snapshots = [...snapshots.slice(-9), canvas.getContext('2d').getImageData(0,0,width,height)] }
  function line(a,b) { const c=canvas.getContext('2d'); c.strokeStyle=tool==='white'?'white':'black'; c.lineWidth=Math.max(2,Math.min(width,height)*brushSize/100); c.lineCap='round'; c.beginPath(); c.moveTo(a.x,a.y); c.lineTo(b.x,b.y); c.stroke() }
  function down(event) { if (event.button) return; event.preventDefault(); remember(); drawing=true; last=point(event); line(last,last); canvas.setPointerCapture(event.pointerId) }
  function move(event) { if(!drawing)return; const next=point(event); line(last,next); last=next }
  function up(){ drawing=false; last=null }
  function undo(){ const state=snapshots.at(-1); if(!state)return; canvas.getContext('2d').putImageData(state,0,0); snapshots=snapshots.slice(0,-1) }
  function addText(){ if(!label.trim())return; remember(); const c=canvas.getContext('2d'); c.fillStyle='white'; c.textAlign='center'; c.textBaseline='middle'; c.font=`700 ${Math.max(12,height*labelSize/100)}px sans-serif`; c.fillText(label.trim(),width/2,height/2); label='' }
  function save(){ canvas.toBlob((blob)=>blob&&onApply(new File([blob],'krea-canny-map.png',{type:'image/png'})),'image/png') }
</script>

<svelte:window onkeydown={(event)=>{if(open&&event.key==='Escape')onClose()}} />
{#if open}
  <div class="mask-editor-backdrop" role="presentation" onclick={(event)=>{if(event.target===event.currentTarget)onClose()}}>
    <section class="mask-editor" role="dialog" aria-modal="true" aria-label="Canny 윤곽 편집기">
      <header><div><strong>Canny 윤곽 미리보기·편집</strong><small>흰 선은 따를 윤곽입니다. 브러시와 글자로 직접 보정할 수 있습니다.</small></div><button type="button" onclick={onClose}>×</button></header>
      <div class="mask-toolbar">
        <label>감도 <input type="range" min="20" max="240" step="5" bind:value={threshold} onchange={renderEdges} disabled={preprocessed}><b>{threshold}</b></label>
        <div class="segmented"><button type="button" class:active={tool==='white'} onclick={()=>tool='white'}>흰 선</button><button type="button" class:active={tool==='black'} onclick={()=>tool='black'}>지우기</button></div>
        <label>크기 <input type="range" min="1" max="12" bind:value={brushSize}></label><button type="button" onclick={undo} disabled={!snapshots.length}>실행 취소</button>
      </div>
      <div class="mask-stage" style={`aspect-ratio:${width||1}/${height||1}`}><canvas bind:this={canvas} onpointerdown={down} onpointermove={move} onpointerup={up} onpointercancel={up}></canvas></div>
      <div class="mask-presets"><label>글자 윤곽 <input bind:value={label} placeholder="윤곽맵에 넣을 글자"></label><label>크기 <input type="range" min="2" max="25" bind:value={labelSize}></label><button type="button" onclick={addText} disabled={!label.trim()}>가운데 추가</button></div>
      <footer><button type="button" onclick={onClose}>취소</button><button type="button" class="primary" onclick={save}>이 윤곽맵 사용</button></footer>
    </section>
  </div>
{/if}
