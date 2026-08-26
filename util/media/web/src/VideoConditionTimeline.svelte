<script>
  export let duration = 5
  export let fps = 24
  export let startImage = null
  export let endImage = null
  export let keyframes = []
  export let imageURL = (image) => image?.preview || image?.url || ''
  export let onMove = () => {}
  export let large = false

  let track
  let dragging = null

  const clamp = (value, min, max) => Math.min(max, Math.max(min, value))
  const finalTime = () => Math.max(1 / Math.max(1, Number(fps)), Number(duration) || 0)
  const position = (time) => `${clamp(Number(time) / finalTime(), 0, 1) * 100}%`

  function move(event) {
    if (dragging === null || !track) return
    const box = track.getBoundingClientRect()
    const ratio = clamp((event.clientX - box.left) / Math.max(1, box.width), 0, 1)
    onMove(dragging, ratio * finalTime())
  }

  function startDrag(event, id) {
    dragging = id
    try { event.currentTarget.setPointerCapture?.(event.pointerId) } catch {}
    move(event)
  }

  function stopDrag(event) {
    try { event.currentTarget.releasePointerCapture?.(event.pointerId) } catch {}
    dragging = null
  }

  function nudge(event, keyframe) {
    if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return
    event.preventDefault()
    const frames = event.shiftKey ? 5 : 1
    const direction = event.key === 'ArrowRight' ? 1 : -1
    onMove(keyframe.id, Number(keyframe.time) + direction * frames / Math.max(1, Number(fps)))
  }
</script>

<div class="timeline-shell" class:large aria-label="영상 장면 타임라인">
  <div class="timeline-labels"><span>0초</span><strong>장면을 좌우로 드래그</strong><span>{finalTime().toFixed(2)}초</span></div>
  <div class="timeline-track" bind:this={track}>
    <div class="timeline-line"></div>
    <div class="marker boundary start" style="left:0%">
      <div class="thumb">{#if imageURL(startImage)}<img src={imageURL(startImage)} alt="시작 장면">{:else}<span>S</span>{/if}</div>
      <small>시작</small>
    </div>
    {#each keyframes as keyframe, index (keyframe.id)}
      <button
        type="button"
        class="marker keyframe"
        class:dragging={dragging === keyframe.id}
        style={`left:${position(keyframe.time)}`}
        title={`키프레임 ${index + 1} · ${Number(keyframe.time).toFixed(2)}초 · 드래그하여 이동`}
        onpointerdown={(event) => startDrag(event, keyframe.id)}
        onpointermove={move}
        onpointerup={stopDrag}
        onpointercancel={stopDrag}
        onkeydown={(event) => nudge(event, keyframe)}
      >
        <span class="thumb">{#if imageURL(keyframe.image)}<img src={imageURL(keyframe.image)} alt="키프레임 {index + 1}">{:else}<b>K{index + 1}</b>{/if}</span>
        <small>{Number(keyframe.time).toFixed(2)}초</small>
      </button>
    {/each}
    <div class="marker boundary end" style="left:100%">
      <div class="thumb">{#if imageURL(endImage)}<img src={imageURL(endImage)} alt="마지막 장면">{:else}<span>E</span>{/if}</div>
      <small>마지막</small>
    </div>
  </div>
</div>

<style>
  .timeline-shell { overflow:hidden; padding:10px 12px 12px; border:1px solid #313b32; border-radius:10px; background:linear-gradient(135deg,#111712,#0b0f0c); }
  .timeline-labels { display:flex; align-items:center; justify-content:space-between; gap:8px; color:#7f8b80; font-size:9px; }
  .timeline-labels strong { color:#aeb9af; font-weight:600; letter-spacing:.02em; }
  .timeline-track { position:relative; height:82px; margin:4px 32px 0; touch-action:none; user-select:none; }
  .timeline-line { position:absolute; top:31px; right:0; left:0; height:8px; border:1px solid #59665a; border-radius:99px; background:linear-gradient(90deg,#879e70 0%,#c9b85f 48%,#a56868 100%); box-shadow:inset 0 1px 3px #0008; }
  .marker { position:absolute; top:4px; z-index:2; display:grid; justify-items:center; gap:3px; width:54px; padding:0; transform:translateX(-50%); border:0; color:#dce5dc; background:transparent; font:inherit; }
  .marker .thumb { display:grid; place-items:center; overflow:hidden; width:48px; height:48px; border:2px solid #768276; border-radius:8px; background:#171d18; box-shadow:0 3px 10px #0009; }
  .marker img { width:100%; height:100%; object-fit:cover; pointer-events:none; }
  .marker small { padding:2px 5px; border-radius:99px; color:#b8c1b9; background:#111712e8; font-size:8px; white-space:nowrap; }
  .keyframe { cursor:ew-resize; }
  .keyframe:hover .thumb, .keyframe.dragging .thumb { border-color:#d5ee98; box-shadow:0 0 0 3px #b8d4772e,0 5px 16px #000c; }
  .keyframe.dragging { z-index:5; }
  .boundary { pointer-events:none; }
  .boundary .thumb { border-color:#566057; color:#879188; }
  .start { transform:translateX(-50%); }
  .end { transform:translateX(-50%); }
  .timeline-shell.large { padding:16px 20px 20px; }
  .timeline-shell.large .timeline-labels { font-size:11px; }
  .timeline-shell.large .timeline-track { height:142px; margin:10px 54px 0; }
  .timeline-shell.large .timeline-line { top:49px; height:12px; }
  .timeline-shell.large .marker { width:94px; }
  .timeline-shell.large .marker .thumb { width:82px; height:82px; border-radius:11px; }
  .timeline-shell.large .marker small { padding:3px 7px; font-size:10px; }
  @media(max-width:700px) {
    .timeline-shell { padding-right:6px; padding-left:6px; }
    .timeline-track { margin-right:28px; margin-left:28px; }
    .marker { width:46px; }
    .marker .thumb { width:42px; height:42px; }
    .timeline-shell.large { padding:10px 8px 13px; }
    .timeline-shell.large .timeline-track { height:112px; margin:7px 36px 0; }
    .timeline-shell.large .timeline-line { top:39px; height:10px; }
    .timeline-shell.large .marker { width:66px; }
    .timeline-shell.large .marker .thumb { width:62px; height:62px; }
  }
</style>
