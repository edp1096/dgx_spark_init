<script>
  import { onMount } from 'svelte';
  import { cleanupMedia, getMediaUsage, saveConfig } from '../api.js';
  import PromptPresetManager from './settings/PromptPresetManager.svelte';
  import AvatarSettings from './settings/AvatarSettings.svelte';

  export let settings;
  export let models = [];
  export let keepImageIds = [];
  export let onclose = () => {};
  export let onsaved = async () => {};

  let settingsAPIKey = '';
  let clearAPIKey = false;
  let settingsNotice = '';
  let mediaUsage = null;
  let cleaningMedia = false;
  let avatarKeepIds = [];

  onMount(async () => {
    normalizePromptPresetSettings();
    try { mediaUsage = await getMediaUsage(); }
    catch (error) { settingsNotice = error.message; }
  });

  function normalizePromptPresetSettings() {
    if (!settings?.model) return;
    if (!Array.isArray(settings.model.system_prompt_presets)) settings.model.system_prompt_presets = [];
    if (!settings.model.system_prompt_preset) settings.model.system_prompt_preset = '';
  }

  function formatBytes(value) {
    if (!value) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB'];
    const index = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1);
    return `${(value / (1024 ** index)).toFixed(index ? 1 : 0)} ${units[index]}`;
  }

  async function removeUnusedMedia() {
    if (cleaningMedia || !mediaUsage?.unused_files) return;
    if (!confirm(`대화에서 사용하지 않는 이미지 ${mediaUsage.unused_files}개를 삭제할까요?`)) return;
    cleaningMedia = true;
    settingsNotice = '';
    try {
      const result = await cleanupMedia([...keepImageIds, ...avatarKeepIds]);
      mediaUsage = result.usage;
      settingsNotice = `미사용 이미지 ${result.removed.files}개(${formatBytes(result.removed.bytes)})를 정리했습니다.`;
    } catch (error) { settingsNotice = error.message; }
    finally { cleaningMedia = false; }
  }

  async function persistSettings() {
    try {
      const result = await saveConfig({
        server: settings.server,
        model: settings.model,
        tools: settings.tools,
        appearance: settings.appearance,
        api_key: settingsAPIKey,
        clear_api_key: clearAPIKey,
      });
      settings = result.config;
      await onsaved(settings);
      settingsNotice = result.restart_required
        ? '저장했습니다. 주소 또는 DB 변경은 앱을 재시작하면 반영됩니다.'
        : '저장했으며 즉시 반영했습니다.';
      settingsAPIKey = '';
      clearAPIKey = false;
    } catch (error) { settingsNotice = error.message; }
  }
</script>

<div class="modal-backdrop" role="presentation" onclick={(event) => event.target === event.currentTarget && onclose()}>
  <div class="settings-modal" role="dialog" aria-modal="true" aria-labelledby="settings-title">
    <div class="modal-title"><h2 id="settings-title">설정</h2><button onclick={onclose} aria-label="닫기">×</button></div>
    <label>API endpoint<input bind:value={settings.model.endpoint} placeholder="http://192.168.100.61:8000" /></label>
    <label>Endpoint API key (선택)<input type="password" bind:value={settingsAPIKey} placeholder={settings.api_key_set ? '설정됨 — 변경할 때만 입력' : '인증이 필요할 때만 입력'} /></label>
    {#if settings.api_key_set}<label class="check"><input type="checkbox" bind:checked={clearAPIKey} /> 저장된 Endpoint API key 제거</label>{/if}
    <label>기본 모델<input bind:value={settings.model.default_model} list="model-list" placeholder="비우면 첫 모델 자동 선택" /></label>
    <datalist id="model-list">{#each models as model}<option value={model}></option>{/each}</datalist>
    <label>기본 reasoning effort<input bind:value={settings.model.reasoning_effort} list="reasoning-levels" placeholder="medium 또는 0.0~0.99" /></label>
    <PromptPresetManager model={settings.model} bind:notice={settingsNotice} />
    <AvatarSettings bind:appearance={settings.appearance} bind:notice={settingsNotice} onuploaded={(id) => avatarKeepIds = [...avatarKeepIds, id]} />
    <fieldset>
      <legend>웹 도구</legend>
      <label class="check"><input type="checkbox" bind:checked={settings.tools.enabled} /> web_search / web_fetch 활성화</label>
      <label>최대 호출 라운드<input type="number" min="1" max="8" bind:value={settings.tools.max_rounds} /></label>
      <label>검색 결과 수<input type="number" min="1" max="10" bind:value={settings.tools.search_results} /></label>
      <label>도구 타임아웃<input bind:value={settings.tools.timeout} placeholder="15s" /></label>
    </fieldset>
    <label>Listen address<input bind:value={settings.server.listen_addr} placeholder="0.0.0.0:8585" /></label>
    <label>SQLite 파일<input bind:value={settings.server.database} placeholder="sparktalk.db" /></label>
    <fieldset>
      <legend>이미지 보관</legend>
      {#if mediaUsage}
        <div class="media-usage"><span>전체 {mediaUsage.files}개 · {formatBytes(mediaUsage.bytes)}</span><span>미사용 {mediaUsage.unused_files}개 · {formatBytes(mediaUsage.unused_bytes)}</span></div>
        <button class="media-cleanup" onclick={removeUnusedMedia} disabled={cleaningMedia || !mediaUsage.unused_files}>{cleaningMedia ? '정리 중…' : '미사용 이미지 정리'}</button>
      {:else}<span class="media-loading">보관 현황을 불러오는 중…</span>{/if}
      <small>현재 대화에 첨부됐거나 전송 대기 중인 이미지는 유지합니다.</small>
    </fieldset>
    <p class="settings-help">Endpoint·모델·reasoning·시스템 프롬프트는 즉시 반영됩니다. Listen address와 DB 파일 변경은 재시작 후 반영됩니다.</p>
    {#if settingsNotice}<p class="settings-notice">{settingsNotice}</p>{/if}
    <div class="modal-actions"><button class="secondary" onclick={onclose}>닫기</button><button class="primary" onclick={persistSettings}>저장</button></div>
  </div>
</div>
