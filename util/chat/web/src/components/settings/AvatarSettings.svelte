<script>
  import { uploadImage } from '../../api.js';
  import { avatarPresets, avatarURL } from '../../lib/avatars.js';

  export let appearance;
  export let onnotify = () => {};
  export let onuploaded = () => {};

  let assistantInput;
  let userInput;
  let uploadingRole = '';
  let expandedRole = '';

  function selectedName(value, fallback) {
    if (value?.startsWith('/api/images/')) return '커스텀 이미지';
    const id = value?.startsWith('preset:') ? value.slice(7) : fallback;
    return avatarPresets.find((preset) => preset.id === id)?.name ?? '기본 프리셋';
  }

  function select(role, id) {
    appearance = { ...appearance, [`${role}_avatar`]: `preset:${id}` };
    expandedRole = '';
  }

  async function upload(role, event) {
    const file = event.currentTarget.files?.[0];
    event.currentTarget.value = '';
    if (!file || uploadingRole) return;
    uploadingRole = role;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 30000);
    try {
      const attachment = await uploadImage(file, controller.signal);
      appearance = { ...appearance, [`${role}_avatar`]: attachment.url };
      expandedRole = '';
      onuploaded(attachment.id);
      onnotify(`${role === 'assistant' ? 'AI' : '내'} 커스텀 아바타를 올렸습니다. 설정 저장을 누르면 반영됩니다.`);
    } catch (error) {
      onnotify(error.name === 'AbortError' ? '아바타 업로드 시간이 초과되었습니다.' : error.message, 'error');
    } finally {
      clearTimeout(timeout);
      uploadingRole = '';
    }
  }
</script>

<fieldset class="avatar-settings">
  <legend>아바타</legend>
  {#each [{ role: 'assistant', title: 'AI 아바타', fallback: 'spark' }, { role: 'user', title: '내 아바타', fallback: 'person-blue' }] as section}
    {@const selected = appearance[`${section.role}_avatar`]}
    <div class="avatar-setting-section">
      <div class="avatar-setting-summary">
        <img src={avatarURL(selected, section.fallback)} alt={`${section.title} 미리보기`} />
        <span class="avatar-setting-label">
          <strong>{section.title}</strong>
          <small>{selectedName(selected, section.fallback)}</small>
        </span>
        <button
          type="button"
          class:active={expandedRole === section.role}
          aria-expanded={expandedRole === section.role}
          onclick={() => expandedRole = expandedRole === section.role ? '' : section.role}
        >{expandedRole === section.role ? '닫기' : '변경'}</button>
      </div>
      {#if expandedRole === section.role}
        <div class="avatar-picker">
          <div class="avatar-preset-grid">
            {#each avatarPresets as preset}
              <button class:selected={selected === `preset:${preset.id}`} onclick={() => select(section.role, preset.id)} title={preset.name} aria-label={`${section.title}: ${preset.name}`}>
                <img src={preset.url} alt="" /><span>{preset.name}</span>
              </button>
            {/each}
          </div>
          <button class="avatar-upload" onclick={() => section.role === 'assistant' ? assistantInput?.click() : userInput?.click()} disabled={Boolean(uploadingRole)}>
            {uploadingRole === section.role ? '업로드 중…' : '커스텀 이미지 올리기'}
          </button>
        </div>
      {/if}
      {#if section.role === 'assistant'}
        <input class="avatar-file-input" bind:this={assistantInput} type="file" accept="image/png,image/jpeg,image/webp" onchange={(event) => upload('assistant', event)} />
      {:else}
        <input class="avatar-file-input" bind:this={userInput} type="file" accept="image/png,image/jpeg,image/webp" onchange={(event) => upload('user', event)} />
      {/if}
    </div>
  {/each}
  <small>변경을 눌러 기본 프리셋을 고르거나 커스텀 이미지를 올릴 수 있습니다.</small>
</fieldset>
