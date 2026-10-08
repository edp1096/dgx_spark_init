<script>
  import { onDestroy } from 'svelte';
  export let catalog;
  export let onclose = () => {};
  let list;
  let drag = null;
  let targetID = '';
  let frame = 0;
  let announcement = '';
  function move(id, to) {
    const items = [...catalog.bundles];
    const from = items.findIndex(item => item.id === id);
    if (from < 0 || to < 0 || to >= items.length || from === to) return;
    const [item] = items.splice(from, 1);
    items.splice(to, 0, item);
    catalog = { ...catalog, bundles: items };
    announcement = `${item.name}, ${to + 1}번째로 이동`;
  }
  function stop() {
    cancelAnimationFrame(frame);
    drag = null; targetID = '';
  }
  onDestroy(stop);
  function locate() {
    const row = document.elementFromPoint(drag.x, drag.y)?.closest('[data-order-id]');
    targetID = row && list.contains(row) ? row.dataset.orderId : '';
  }
  function scroll() {
    if (!drag) return;
    let host = list.parentElement;
    while (host && !(host.scrollHeight > host.clientHeight && /auto|scroll/.test(getComputedStyle(host).overflowY))) host = host.parentElement;
    if (host) {
      const rect = host.getBoundingClientRect();
      const delta = drag.y < rect.top + 40 ? -8 : drag.y > rect.bottom - 40 ? 8 : 0;
      if (delta) { host.scrollTop += delta; locate(); }
    }
    frame = requestAnimationFrame(scroll);
  }
  function start(event, id) {
    if (event.button !== 0 || drag) return;
    event.preventDefault();
    event.currentTarget.focus({ preventScroll: true });
    event.currentTarget.setPointerCapture(event.pointerId);
    drag = { id, pointer: event.pointerId, x: event.clientX, y: event.clientY };
    targetID = id;
    frame = requestAnimationFrame(scroll);
  }
  function pointerMove(event) {
    if (!drag || drag.pointer !== event.pointerId) return;
    drag = { ...drag, x:event.clientX, y:event.clientY };
    locate();
  }
  function drop(event) {
    if (!drag || drag.pointer !== event.pointerId) return;
    pointerMove(event);
    if (targetID) move(drag.id, catalog.bundles.findIndex(item => item.id === targetID));
    stop();
  }
  function cancel(event) { if (drag?.pointer === event.pointerId) stop(); }
  function key(event, id, index) {
    if (event.key === 'Escape') { stop(); return; }
    if (!['ArrowUp','ArrowDown','Home','End'].includes(event.key)) return;
    event.preventDefault();
    move(id, event.key === 'Home' ? 0 : event.key === 'End' ? catalog.bundles.length - 1 : index + (event.key === 'ArrowUp' ? -1 : 1));
  }
</script>

<svelte:window onpointermove={pointerMove} onpointerup={drop} onpointercancel={cancel} onkeydown={event => { if (event.key === 'Escape') stop(); }} />
<section aria-label="세트 표시 순서">
  <div class="heading"><strong>표시 순서</strong><button type="button" onclick={onclose}>편집으로 돌아가기</button></div>
  <p>손잡이를 드래그하거나 화살표로 이동한 뒤 저장하세요.</p>
  <ol bind:this={list}>
    {#each catalog.bundles as item, index (item.id)}
      <li data-order-id={item.id} class:dragging={drag?.id === item.id} class:target={drag && targetID === item.id && drag.id !== item.id}>
        <button type="button" class="handle" aria-label={item.name + ' 순서 이동'} onpointerdown={event => start(event,item.id)} onkeydown={event => key(event,item.id,index)}>⠿</button>
        <span class="number">{index + 1}</span><span class="name">{item.name}</span>
        <button type="button" aria-label={item.name + ' 위로 이동'} disabled={index === 0 || !!drag} onclick={() => move(item.id,index-1)}>↑</button>
        <button type="button" aria-label={item.name + ' 아래로 이동'} disabled={index === catalog.bundles.length-1 || !!drag} onclick={() => move(item.id,index+1)}>↓</button>
      </li>
    {/each}
  </ol>
  <span class="sr-only" role="status">{announcement}</span>
</section>
<style>
  .heading { display:flex; align-items:center; justify-content:space-between; gap:8px; }
  p { font-size:12px; color:inherit; opacity:.7; }
  ol { list-style:none; margin:12px 0; padding:0; }
  li { display:grid; grid-template-columns:36px 22px minmax(0,1fr) 36px 36px; align-items:center; gap:5px; margin:6px 0; padding:7px; border:1px solid #80808040; border-radius:8px; }
  .name { min-width:0; overflow-wrap:anywhere; font-size:12px; }
  .number { font-size:11px; opacity:.6; text-align:center; }
  button { min-height:36px; padding:0 8px; border:1px solid #80808050; border-radius:6px; color:inherit; background:transparent; font:inherit; font-size:12px; cursor:pointer; }
  button:disabled { opacity:.35; cursor:default; }
  button:focus-visible { outline:2px solid #6584ed; outline-offset:2px; }
  .handle { touch-action:none; cursor:grab; user-select:none; font-size:22px; }
  .dragging { opacity:.5; }
  .dragging .handle { cursor:grabbing; }
  .target { border-color:#6584ed; background:#6584ed20; box-shadow:inset 0 0 0 1px #6584ed; }
  .sr-only { position:absolute; width:1px; height:1px; overflow:hidden; clip-path:inset(50%); }
</style>
