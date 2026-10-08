<script>
  import { createClientID } from '../../lib/client-id.js';
  import { onMount } from 'svelte';
  export let title;
  export let buttonText = '';
  let dialog, trigger;
  const headingID = `settings-help-${createClientID()}`;
  onMount(() => {
    document.body.appendChild(dialog);
    return () => dialog.remove();
  });
  function close() { dialog.close(); }
</script>

<button bind:this={trigger} type="button" class="settings-info" class:with-label={Boolean(buttonText)} aria-label={`${title} 도움말`} aria-haspopup="dialog" onclick={() => dialog.showModal()}>{#if buttonText}{buttonText}{:else}<span aria-hidden="true">i</span>{/if}</button>
<dialog bind:this={dialog} class="settings-help-dialog" aria-labelledby={headingID} onclose={() => trigger?.isConnected && trigger.focus()} onkeydown={event => event.key === 'Escape' && event.stopPropagation()} onclick={event => event.target === dialog && close()}>
  <div class="help-heading"><h3 id={headingID}>{title}</h3><button type="button" aria-label="도움말 닫기" onclick={close}>×</button></div>
  <div class="help-content"><slot /></div>
</dialog>

<style>
  .settings-info { display: inline-flex; align-items: center; justify-content: center; width: 28px; height: 28px; padding: 0; margin-left: 5px; border: 0; border-radius: 50%; background: transparent; color: inherit; vertical-align: middle; cursor: pointer; }
  .settings-info.with-label { justify-self: start; width: auto; height: 30px; margin-left: 0; padding: 0 9px; border: 1px solid #80808040; border-radius: 8px; font-size: 11px; }
  .settings-info span { display: grid; place-items: center; width: 16px; height: 16px; border: 1px solid currentColor; border-radius: 50%; font: 600 12px/1 sans-serif; }
  .settings-info:hover { background: #80808020; }
  .settings-info:focus-visible { outline: 2px solid #6584ed; outline-offset: 2px; }
  .settings-help-dialog { width: min(460px, calc(100vw - 32px)); max-height: calc(100dvh - 40px); box-sizing: border-box; margin: auto; padding: 20px; overflow-y: auto; border: 1px solid #80808040; border-radius: 14px; background: #191e28; color: #d7dfef; box-shadow: 0 16px 60px #0005; }
  .settings-help-dialog::backdrop { background: #0006; }
  .help-heading { display: flex; align-items: center; justify-content: space-between; gap: 14px; }
  h3 { margin: 0; font-size: 16px; }
  .help-heading button { flex-shrink: 0; width: 32px; height: 32px; padding: 0; border: 0; border-radius: 7px; background: #80808020; color: inherit; font-size: 22px; cursor: pointer; }
  .help-content { font-size: 13px; line-height: 1.7; overflow-wrap: anywhere; }
  .help-content :global(p) { margin: 12px 0 0; }
  :global(html[data-theme="light"]) .settings-help-dialog { background: #fff; color: #243047; }
</style>
