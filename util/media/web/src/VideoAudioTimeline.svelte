<script>
  export let duration = 5
  export let clips = []
  export let onMove = () => {}
  export let large = false

  let track
  let dragging = null
  const clamp = (value, min, max) => Math.min(max, Math.max(min, value))
  const total = () => Math.max(0.01, Number(duration) || 0.01)
  const left = (clip) => `${clamp(Number(clip.start) / total(), 0, 1) * 100}%`
  const width = (clip) => `${clamp(Number(clip.duration || 0.25) / total(), 0.015, 1) * 100}%`

  function move(event) {
    if (dragging === null || !track) return
    const box = track.getBoundingClientRect()
    const ratio = clamp((event.clientX - box.left) / Math.max(1, box.width), 0, 1)
    onMove(dragging, ratio * total())
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
</script>

<div class="audio-timeline" class:large aria-label="음성 배치 타임라인">
  <div class="labels"><span>0초</span><strong>음성 블록을 좌우로 드래그</strong><span>{total().toFixed(2)}초</span></div>
  <div class="track" bind:this={track}>
    <div class="line"></div>
    {#each clips as clip, index (clip.id)}
      <button
        type="button"
        class:dragging={dragging === clip.id}
        style={`left:${left(clip)};width:${width(clip)}`}
        title={`음성 ${index + 1} · ${Number(clip.start).toFixed(2)}초부터`}
        onpointerdown={(event) => startDrag(event, clip.id)}
        onpointermove={move}
        onpointerup={stopDrag}
        onpointercancel={stopDrag}
      ><b>A{index + 1}</b><span>{Number(clip.start).toFixed(2)}초</span></button>
    {/each}
  </div>
</div>

<style>
  .audio-timeline { padding:8px 10px 9px; border:1px solid #344039; border-radius:9px; background:#0d1210; }
  .labels { display:flex; align-items:center; justify-content:space-between; color:#748078; font-size:8px; }
  .labels strong { color:#9daaa0; font-weight:600; }
  .track { position:relative; height:42px; margin-top:4px; touch-action:none; user-select:none; }
  .line { position:absolute; top:15px; right:0; left:0; height:10px; border:1px solid #455349; border-radius:99px; background:#18211b; }
  button { position:absolute; top:7px; z-index:2; display:flex; min-width:30px; height:27px; align-items:center; justify-content:space-between; gap:4px; overflow:hidden; border:1px solid #91ad72; border-radius:6px; padding:3px 6px; color:#dcebcf; background:linear-gradient(90deg,#3d582d,#25391f); box-shadow:0 2px 8px #0008; cursor:ew-resize; touch-action:none; }
  button.dragging { z-index:4; border-color:#d7f59e; box-shadow:0 0 0 3px #b8d47730; }
  button b { font-size:8px; }
  button span { overflow:hidden; font-size:7px; text-overflow:ellipsis; white-space:nowrap; }
  .large { padding:13px 16px; }
  .large .labels { font-size:10px; }
  .large .track { height:66px; }
  .large .line { top:23px; height:14px; }
  .large button { top:11px; height:40px; padding:5px 9px; }
  .large button b { font-size:10px; }
  .large button span { font-size:9px; }
</style>
