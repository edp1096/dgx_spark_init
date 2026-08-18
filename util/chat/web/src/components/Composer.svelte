<script>
  export let pendingImages = [];
  export let uploadingImages = false;
  export let running = false;
  export let activeId = '';
  export let input = '';
  export let imageInput;
  export let dragActive = false;
  export let reasoningEffort = '';
  export let webToolsEnabled = false;
  export let onRemoveImage = () => {};
  export let onImageInputChange = () => {};
  export let onKeydown = () => {};
  export let onPaste = () => {};
  export let onStop = () => {};
  export let onSend = () => {};
</script>

<footer>
  {#if pendingImages.length || uploadingImages}
    <div class="pending-images">
      {#each pendingImages as attachment}
        <div><img src={attachment.url} alt={attachment.name} /><button onclick={() => onRemoveImage(attachment.id)} disabled={running} aria-label={`${attachment.name} 첨부 제거`}>×</button></div>
      {/each}
      {#if uploadingImages}<span class="uploading">이미지 업로드 중…</span>{/if}
    </div>
  {/if}
  <div class="composer" role="group" aria-label="메시지와 이미지 입력">
    <input class="image-input" class:drop-active={dragActive} bind:this={imageInput} type="file" accept="image/png,image/jpeg,image/webp" multiple onchange={onImageInputChange} />
    <button class="attach" onclick={() => imageInput?.click()} disabled={!activeId || running || uploadingImages || pendingImages.length >= 6} aria-label="이미지 첨부" title="이미지 첨부">＋</button>
    <textarea bind:value={input} onkeydown={onKeydown} onpaste={onPaste} placeholder={activeId ? '메시지를 입력하세요' : '새 대화를 만든 뒤 메시지를 입력하세요'} rows="1" disabled={!activeId || running}></textarea>
    {#if running}<button class="send stop" onclick={onStop}>■</button>{:else}<button class="send" onclick={onSend} disabled={!activeId || !input.trim() || uploadingImages}>↑</button>{/if}
  </div>
  <small>이미지 붙여넣기·드래그 가능 · Enter 전송 · Shift+Enter 줄바꿈 · reasoning: {reasoningEffort || '서버 기본값'} · 웹: {webToolsEnabled ? '자동' : '꺼짐'}</small>
</footer>
