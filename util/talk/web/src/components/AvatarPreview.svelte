<script>
  import { onMount } from 'svelte';
  import { avatarURL } from '../lib/avatars.js';
  export let value = '';
  export let fallback = 'spark';
  export let name = '';
  export let onclose = () => {};
  export let onsettings = () => {};
  let dialog;
  onMount(() => dialog.showModal());
  function settings() { dialog.close(); onsettings(); }
</script>

<!-- svelte-ignore a11y_no_noninteractive_element_interactions: native dialog backdrop dismissal -->
<dialog bind:this={dialog} class="avatar-preview" aria-label={`${name} 아바타 크게 보기`} onclose={onclose} onclick={(event) => { if (event.target === dialog) dialog.close(); }}>
  <div class="avatar-preview-content">
    <header><strong>{name}</strong><button type="button" aria-label="아바타 크게 보기 닫기" onclick={() => dialog.close()}><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18" /></svg></button></header>
    <img src={avatarURL(value, fallback)} alt={`${name} 아바타`} />
    <footer><button type="button" onclick={settings}>프로필 설정</button></footer>
  </div>
</dialog>

<style>
.avatar-preview{box-sizing:border-box;width:min(560px,calc(100vw - 24px));max-height:calc(100dvh - 24px);padding:0;border:1px solid #414958;border-radius:16px;background:#181d27;color:#e9edf5;overflow:auto}.avatar-preview::backdrop{background:#000b;backdrop-filter:blur(5px)}.avatar-preview-content{padding:14px}.avatar-preview header{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:12px}.avatar-preview strong{overflow-wrap:anywhere}.avatar-preview button{font:inherit;color:inherit;background:transparent;border:1px solid #80808060;border-radius:8px;padding:7px 12px;cursor:pointer}.avatar-preview header button{display:grid;place-items:center;flex:0 0 36px;width:36px;height:36px;padding:0;border:0}.avatar-preview img{display:block;width:100%;height:min(480px,65dvh);object-fit:contain;border-radius:10px}.avatar-preview footer{display:flex;justify-content:flex-end;margin-top:12px}:global(html[data-theme="light"]) .avatar-preview{background:#fff;color:#27354a;border-color:#ccd3df}
</style>
