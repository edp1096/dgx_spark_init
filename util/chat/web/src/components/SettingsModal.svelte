<script>
  import { onMount } from 'svelte';
  import { cleanupMedia, getHealth, getMediaUsage, saveConfig } from '../api.js';
  import PromptPresetManager from './settings/PromptPresetManager.svelte';
  import AvatarSettings from './settings/AvatarSettings.svelte';

  export let settings;
  export let models = [];
  export let keepMediaIds = [];
  export let onclose = () => {};
  export let onsaved = async () => {};

  let settingsAPIKey = '';
  let clearAPIKey = false;
  let settingsNotice = '';
  let mediaUsage = null;
  let cleaningMedia = false;
  let avatarKeepIds = [];
  let serviceHealth = null;

  onMount(async () => {
    normalizePromptPresetSettings();
    const [usageResult, healthResult] = await Promise.allSettled([getMediaUsage(), getHealth()]);
    if (usageResult.status === 'fulfilled') mediaUsage = usageResult.value;
    else settingsNotice = usageResult.reason.message;
    if (healthResult.status === 'fulfilled') serviceHealth = healthResult.value;
  });

  function normalizePromptPresetSettings() {
    if (!settings?.model) return;
    if (!Array.isArray(settings.model.system_prompt_presets)) settings.model.system_prompt_presets = [];
    if (!settings.model.system_prompt_preset) settings.model.system_prompt_preset = '';
    if (!settings.context) settings.context = { enabled: true, window_tokens: 0, compact_at_percent: 80, output_reserve: 8192, safety_margin: 4096, recent_tokens: 32768, image_tokens: 2048 };
    if (!settings.asr) settings.asr = { enabled: true, ffmpeg_endpoint: 'http://127.0.0.1:8698', endpoint: 'http://127.0.0.1:8694', model: 'qwen3-asr', language: 'auto', prompt: '', timeout: '30m' };
  }

  function formatBytes(value) {
    if (!value) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB'];
    const index = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1);
    return `${(value / (1024 ** index)).toFixed(index ? 1 : 0)} ${units[index]}`;
  }

  async function removeUnusedMedia() {
    if (cleaningMedia || !mediaUsage?.unused_files) return;
    if (!confirm(`대화에서 사용하지 않는 미디어 ${mediaUsage.unused_files}개를 삭제할까요?`)) return;
    cleaningMedia = true;
    settingsNotice = '';
    try {
      const result = await cleanupMedia([...keepMediaIds, ...avatarKeepIds]);
      mediaUsage = result.usage;
      settingsNotice = `미사용 미디어 ${result.removed.files}개(${formatBytes(result.removed.bytes)})를 정리했습니다.`;
    } catch (error) { settingsNotice = error.message; }
    finally { cleaningMedia = false; }
  }

  async function persistSettings() {
    try {
      const result = await saveConfig({
        server: settings.server,
        model: settings.model,
        asr: settings.asr,
        context: settings.context,
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
    <fieldset>
      <legend>음성 인식</legend>
      <label class="check"><input type="checkbox" bind:checked={settings.asr.enabled} /> 음성·영상의 음성을 Qwen3-ASR로 전사</label>
      <label>SparkTalk Media API endpoint<input bind:value={settings.asr.ffmpeg_endpoint} placeholder="http://127.0.0.1:8698" /></label>
      <label>ASR API endpoint<input bind:value={settings.asr.endpoint} placeholder="http://127.0.0.1:8694" /></label>
      <label>ASR 모델<input bind:value={settings.asr.model} placeholder="qwen3-asr" /></label>
      <label>인식 언어<input bind:value={settings.asr.language} list="asr-languages" placeholder="auto 또는 Korean" /></label>
      <datalist id="asr-languages">
        {#each ['auto', 'Korean', 'Japanese', 'English', 'Chinese', 'Cantonese', 'French', 'German', 'Spanish', 'Portuguese', 'Italian', 'Russian', 'Thai', 'Vietnamese', 'Arabic', 'Turkish', 'Hindi', 'Indonesian', 'Malay', 'Dutch', 'Swedish', 'Danish', 'Finnish', 'Polish', 'Czech', 'Filipino', 'Persian', 'Greek', 'Romanian', 'Hungarian'] as language}<option value={language}></option>{/each}
      </datalist>
      <label>문맥·전문용어 힌트<textarea rows="2" bind:value={settings.asr.prompt} placeholder="비우면 사용하지 않음"></textarea></label>
      <label>처리 타임아웃<input bind:value={settings.asr.timeout} placeholder="30m" /></label>
      {#if serviceHealth?.asr}
        <div class="media-usage">
          <span>Media API · {serviceHealth.asr.ffmpeg?.status === 'ok' ? 'online' : serviceHealth.asr.ffmpeg?.status}</span>
          <span>ASR API · {serviceHealth.asr.asr?.status === 'ok' ? 'online' : serviceHealth.asr.asr?.status}{serviceHealth.asr.asr?.model ? ` · ${serviceHealth.asr.asr.model}` : ''}</span>
        </div>
      {/if}
      <small>음성 원본은 모델에 보내지 않고 전사문으로 대체합니다. 영상은 화면 정보와 전사문을 함께 보냅니다.</small>
    </fieldset>
    <fieldset>
      <legend>지능형 문맥 관리</legend>
      <label class="check"><input type="checkbox" bind:checked={settings.context.enabled} /> 오래된 원문을 구조화 요약으로 전환</label>
      <label>모델 context window (0은 백엔드 자동 감지)<input type="number" min="0" step="1024" bind:value={settings.context.window_tokens} /></label>
      <label>자동 정리 시작 비율<input type="number" min="50" max="95" bind:value={settings.context.compact_at_percent} /></label>
      <label>출력 예약 토큰<input type="number" min="256" step="256" bind:value={settings.context.output_reserve} /></label>
      <label>안전 여유 토큰<input type="number" min="256" step="256" bind:value={settings.context.safety_margin} /></label>
      <label>최근 원문 유지 토큰<input type="number" min="256" step="256" bind:value={settings.context.recent_tokens} /></label>
      <label>이미지 장당 보수적 추정 토큰<input type="number" min="1" step="128" bind:value={settings.context.image_tokens} /></label>
      <small>대화와 첨부 원본은 SQLite와 화면에 그대로 남고 모델로 보내는 활성 문맥만 정리합니다.</small>
    </fieldset>
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
      <legend>미디어 보관</legend>
      {#if mediaUsage}
        <div class="media-usage"><span>전체 {mediaUsage.files}개 · {formatBytes(mediaUsage.bytes)}</span><span>미사용 {mediaUsage.unused_files}개 · {formatBytes(mediaUsage.unused_bytes)}</span></div>
        <button class="media-cleanup" onclick={removeUnusedMedia} disabled={cleaningMedia || !mediaUsage.unused_files}>{cleaningMedia ? '정리 중…' : '미사용 미디어 정리'}</button>
      {:else}<span class="media-loading">보관 현황을 불러오는 중…</span>{/if}
      <small>현재 대화에 첨부됐거나 전송 대기 중인 이미지·음성·비디오는 유지합니다.</small>
    </fieldset>
    <p class="settings-help">Endpoint·모델·reasoning·시스템 프롬프트는 즉시 반영됩니다. Listen address와 DB 파일 변경은 재시작 후 반영됩니다.</p>
    {#if settingsNotice}<p class="settings-notice">{settingsNotice}</p>{/if}
    <div class="modal-actions"><button class="secondary" onclick={onclose}>닫기</button><button class="primary" onclick={persistSettings}>저장</button></div>
  </div>
</div>
