<script>
  export let toast = null;
  export let onclose = () => {};
</script>

{#if toast}
  <div class="settings-toast-stack" aria-live="polite" aria-atomic="true">
    <div class:error={toast.kind === 'error'} class="settings-toast" role={toast.kind === 'error' ? 'alert' : 'status'}>
      <span>{toast.message}</span>
      <button type="button" onclick={onclose} aria-label="알림 닫기">×</button>
    </div>
  </div>
{/if}
