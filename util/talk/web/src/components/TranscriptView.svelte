<script>
  export let attachment;
  let opened = false;
  let busy = false;
  let transcript = null;
  let error = '';
  const label = (ids) => !ids?.length ? '화자 미확인' : ids.map((id) => `화자 ${id}`).join('·') + (ids.length > 1 ? ' (겹침·구분 불확실)' : '');
  const time = (seconds) => `${Math.floor(seconds / 60)}:${Math.floor(seconds % 60).toString().padStart(2, '0')}`;
  async function toggle() {
    opened = !opened;
    if (!opened) return;
    busy = true; error = '';
    try {
      const response = await fetch(`/api/media/transcript/${encodeURIComponent(attachment.id)}`);
      if (!response.ok) throw new Error(await response.text());
      transcript = await response.json();
    } catch (e) { error = e.message; }
    finally { busy = false; }
  }
</script>
<div class="transcript">
  <button type="button" onclick={toggle} aria-expanded={opened}>{opened ? '전사 접기' : '전사 보기'}</button>
  {#if opened}
    <div class="content" aria-live="polite">
      {#if busy}<p>불러오는 중…</p>
      {:else if error}<p>{error}</p>
      {:else if transcript}
        {#if transcript.warning}<p class="warning">{transcript.warning}</p>{/if}
        {#if transcript.turns?.length}
          {#each transcript.turns as turn}
            <p><small>{time(turn.start)}–{time(turn.end)}</small> <strong>{label(turn.speakers)}</strong><br />{turn.text}</p>
          {/each}
        {:else}<p>{transcript.text}</p>{/if}
      {/if}
    </div>
  {/if}
</div>
<style>
.transcript{width:100%;min-width:0}.transcript button{font:inherit;font-size:.85rem;color:inherit;background:transparent;border:1px solid #7a879966;border-radius:6px;padding:5px 10px;cursor:pointer}.content{max-height:50vh;overflow:auto;overflow-wrap:anywhere;white-space:pre-wrap;padding:8px 12px;margin-top:6px;border:1px solid #7a879944;border-radius:8px}.content p{margin:0 0 12px}.content small{opacity:.7}.warning{color:#b87916}
</style>
