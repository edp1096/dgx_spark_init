<script>
  import { onDestroy, onMount } from 'svelte';
  import { cleanupMedia, getHealth, getMediaUsage, saveConfig } from '../api.js';
  import PromptPresetManager from './settings/PromptPresetManager.svelte';
  import AvatarSettings from './settings/AvatarSettings.svelte';
  import SSHSettings from './settings/SSHSettings.svelte';
  import SettingsToast from './settings/SettingsToast.svelte';

  export let settings;
  export let models = [];
  export let keepMediaIds = [];
  export let onclose = () => {};
  export let onsaved = async () => {};

  let settingsAPIKey = '';
  let clearAPIKey = false;
  let toast = null;
  let toastTimer;
  let mediaUsage = null;
  let cleaningMedia = false;
  let avatarKeepIds = [];
  let serviceHealth = null;
  let activeTab = 'model';
  const settingsTabs = [
    { id: 'model', label: '모델' },
    { id: 'voice', label: '음성' },
    { id: 'appearance', label: '외형' },
    { id: 'tools', label: '도구' },
    { id: 'app', label: '앱·저장' },
  ];

  onMount(async () => {
    normalizePromptPresetSettings();
    const [usageResult, healthResult] = await Promise.allSettled([getMediaUsage(), getHealth()]);
    if (usageResult.status === 'fulfilled') mediaUsage = usageResult.value;
    else notify(usageResult.reason.message, 'error');
    if (healthResult.status === 'fulfilled') serviceHealth = healthResult.value;
  });

  onDestroy(() => clearTimeout(toastTimer));

  function notify(message, kind = 'success') {
    if (!message) return;
    clearTimeout(toastTimer);
    toast = { message, kind };
    toastTimer = setTimeout(() => { toast = null; }, kind === 'error' ? 7000 : 4200);
  }

  function closeToast() {
    clearTimeout(toastTimer);
    toast = null;
  }

  function tabKeydown(event, index) {
    let next = index;
    if (event.key === 'ArrowRight') next = (index + 1) % settingsTabs.length;
    else if (event.key === 'ArrowLeft') next = (index - 1 + settingsTabs.length) % settingsTabs.length;
    else if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = settingsTabs.length - 1;
    else return;
    event.preventDefault();
    activeTab = settingsTabs[next].id;
    document.getElementById(`settings-tab-${activeTab}`)?.focus();
  }

  function normalizePromptPresetSettings() {
    if (!settings?.model) return;
    if (!Array.isArray(settings.model.system_prompt_presets)) settings.model.system_prompt_presets = [];
    if (!settings.model.system_prompt_preset) settings.model.system_prompt_preset = '';
    if (!settings.context) settings.context = { enabled: true, window_tokens: 0, compact_at_percent: 80, output_reserve: 8192, safety_margin: 4096, recent_tokens: 32768, image_tokens: 2048 };
    if (!settings.asr) settings.asr = { enabled: true, ffmpeg_endpoint: 'http://127.0.0.1:8698', endpoint: 'http://127.0.0.1:8694', model: 'qwen3-asr', language: 'auto', prompt: '', filter_fillers: true, timeout: '30m' };
    if (settings.asr.filter_fillers === undefined) settings.asr.filter_fillers = true;
    if (!settings.tts) settings.tts = { enabled: true, endpoint: 'http://127.0.0.1:8692', model: 'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice', language: 'Korean', voice: 'Sohee', instructions: '', seed: -1, auto_play: false, timeout: '10m' };
    if (settings.tools && settings.tools.media_import_enabled === undefined) settings.tools.media_import_enabled = true;
    if (!settings.extra) settings.extra = { ssh_enabled: false, ssh_endpoint: 'http://127.0.0.1:8699' };
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
    try {
      const result = await cleanupMedia([...keepMediaIds, ...avatarKeepIds]);
      mediaUsage = result.usage;
      notify(`미사용 미디어 ${result.removed.files}개(${formatBytes(result.removed.bytes)})를 정리했습니다.`);
    } catch (error) { notify(error.message, 'error'); }
    finally { cleaningMedia = false; }
  }

  async function persistSettings() {
    try {
      const result = await saveConfig({
        server: settings.server,
        model: settings.model,
        asr: settings.asr,
        tts: settings.tts,
        context: settings.context,
        tools: settings.tools,
        extra: settings.extra,
        appearance: settings.appearance,
        api_key: settingsAPIKey,
        clear_api_key: clearAPIKey,
      });
      settings = result.config;
      await onsaved(settings);
      notify(result.restart_required
        ? '저장했습니다. 주소 또는 DB 변경은 앱을 재시작하면 반영됩니다.'
        : '저장했으며 즉시 반영했습니다.');
      settingsAPIKey = '';
      clearAPIKey = false;
    } catch (error) { notify(error.message, 'error'); }
  }
</script>

<div class="modal-backdrop" role="presentation" onclick={(event) => event.target === event.currentTarget && onclose()}>
  <SettingsToast {toast} onclose={closeToast} />
  <div class="settings-modal" role="dialog" aria-modal="true" aria-labelledby="settings-title">
    <div class="modal-title"><h2 id="settings-title">설정</h2><button onclick={onclose} aria-label="닫기">×</button></div>
    <div class="settings-tabs" role="tablist" aria-label="설정 분류">
      {#each settingsTabs as tab, index}
        <button id={`settings-tab-${tab.id}`} type="button" role="tab" aria-selected={activeTab === tab.id} aria-controls={`settings-panel-${tab.id}`} class:active={activeTab === tab.id} tabindex={activeTab === tab.id ? 0 : -1} onclick={() => activeTab = tab.id} onkeydown={(event) => tabKeydown(event, index)}>{tab.label}</button>
      {/each}
    </div>
    <div class="settings-content">
      <div id="settings-panel-model" class="settings-tab-panel" class:active={activeTab === 'model'} role="tabpanel" aria-labelledby="settings-tab-model">
        <label>API endpoint<input bind:value={settings.model.endpoint} placeholder="http://192.168.100.61:8000" /></label>
        <label>Endpoint API key (선택)<input type="password" bind:value={settingsAPIKey} placeholder={settings.api_key_set ? '설정됨 — 변경할 때만 입력' : '인증이 필요할 때만 입력'} /></label>
        {#if settings.api_key_set}<label class="check"><input type="checkbox" bind:checked={clearAPIKey} /> 저장된 Endpoint API key 제거</label>{/if}
        <label>기본 모델<input bind:value={settings.model.default_model} list="model-list" placeholder="비우면 첫 모델 자동 선택" /></label>
        <datalist id="model-list">{#each models as model}<option value={model}></option>{/each}</datalist>
        <label>기본 reasoning effort<input bind:value={settings.model.reasoning_effort} list="reasoning-levels" placeholder="medium 또는 0.0~0.99" /></label>
        <PromptPresetManager model={settings.model} onnotify={notify} />
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
      </div>

      <div id="settings-panel-voice" class="settings-tab-panel" class:active={activeTab === 'voice'} role="tabpanel" aria-labelledby="settings-tab-voice">
        <fieldset>
          <legend>음성 인식</legend>
          <label class="check"><input type="checkbox" bind:checked={settings.asr.enabled} /> 음성·영상의 음성을 Qwen3-ASR로 전사</label>
          <div class="settings-section-title">인식 정확도</div>
          <label>기본 인식 언어<input bind:value={settings.asr.language} list="asr-languages" placeholder="auto 또는 Korean" /></label>
          <datalist id="asr-languages">{#each ['auto', 'Korean', 'Japanese', 'English', 'Chinese', 'Cantonese', 'French', 'German', 'Spanish', 'Portuguese', 'Italian', 'Russian', 'Thai', 'Vietnamese', 'Arabic', 'Turkish', 'Hindi', 'Indonesian', 'Malay', 'Dutch', 'Swedish', 'Danish', 'Finnish', 'Polish', 'Czech', 'Filipino', 'Persian', 'Greek', 'Romanian', 'Hungarian'] as language}<option value={language}></option>{/each}</datalist>
          <small class="settings-inline-help"><code>auto</code>는 혼합 언어에 적합합니다. 대부분 한 언어라면 <code>Korean</code>처럼 고정하는 편이 안정적입니다.</small>
          <label>문맥·전문용어 힌트<textarea rows="3" bind:value={settings.asr.prompt} placeholder="예: 한국어 기술 대화. 주요 용어: SparkTalk, DGX Spark, SGLang, Qwen3-ASR"></textarea></label>
          <small class="settings-inline-help">자주 말하는 제품명·인명·약어를 정확한 표기로 적으십시오. 긴 지시문보다 짧은 문맥과 용어 목록이 적합합니다.</small>
          <label class="check"><input type="checkbox" bind:checked={settings.asr.filter_fillers} /> 음성대기에서 단독 추임새·문장부호 무시</label>
          <details class="settings-advanced"><summary>연결 및 고급 설정</summary><div class="settings-advanced-body">
            <label>SparkTalk Extra Media endpoint<input bind:value={settings.asr.ffmpeg_endpoint} placeholder="http://127.0.0.1:8698" /></label>
            <label>ASR API endpoint<input bind:value={settings.asr.endpoint} placeholder="http://127.0.0.1:8694" /></label>
            <label>ASR 모델<input bind:value={settings.asr.model} placeholder="qwen3-asr" /></label>
            <label>처리 타임아웃<input bind:value={settings.asr.timeout} placeholder="30m" /></label>
            <small>생성 토큰·dtype·동시 처리 수는 Qwen3-ASR 컨테이너의 기동 설정이며 여기서 변경하지 않습니다.</small>
          </div></details>
          {#if serviceHealth?.asr}<div class="media-usage"><span>Media API · {serviceHealth.asr.ffmpeg?.status === 'ok' ? 'online' : serviceHealth.asr.ffmpeg?.status}</span><span>ASR API · {serviceHealth.asr.asr?.status === 'ok' ? 'online' : serviceHealth.asr.asr?.status}{serviceHealth.asr.asr?.model ? ` · ${serviceHealth.asr.asr.model}` : ''}</span></div>{/if}
          <small>음성 원본은 모델에 보내지 않고 전사문으로 대체합니다. 영상은 화면 정보와 전사문을 함께 보냅니다.</small>
        </fieldset>
        <fieldset>
          <legend>답변 음성</legend>
          <label class="check"><input type="checkbox" bind:checked={settings.tts.enabled} /> Qwen3-TTS로 AI 답변 읽기</label>
          <label class="check"><input type="checkbox" bind:checked={settings.tts.auto_play} disabled={!settings.tts.enabled} /> 답변 완료 후 자동 재생</label>
          <div class="settings-form-row three">
            <label>언어<select bind:value={settings.tts.language}><option>Korean</option><option>English</option><option>Chinese</option><option>Japanese</option><option>Auto</option></select></label>
            <label>화자<select bind:value={settings.tts.voice}><option>Sohee</option><option>Vivian</option><option>Serena</option><option>Ryan</option><option>Aiden</option><option>Ono_Anna</option></select></label>
            <label>시드<input type="number" min="-1" max="2147483647" bind:value={settings.tts.seed} /></label>
          </div>
          <label>기본 연기 지시<textarea rows="2" bind:value={settings.tts.instructions} placeholder="예: 차분하고 또렷한 목소리로 읽어 주세요."></textarea></label>
          <small class="settings-inline-help">시드 -1은 답변마다 임의값을 만들고 한 답변의 음성 묶음에서 공유합니다. 재생 중에는 마이크 판정을 멈춥니다.</small>
          <details class="settings-advanced"><summary>연결 및 고급 설정</summary><div class="settings-advanced-body">
            <label>TTS API endpoint<input bind:value={settings.tts.endpoint} placeholder="http://127.0.0.1:8692" /></label>
            <label>TTS 모델<input bind:value={settings.tts.model} placeholder="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice" /></label>
            <label>처리 타임아웃<input bind:value={settings.tts.timeout} placeholder="10m" /></label>
          </div></details>
          {#if serviceHealth?.tts}<div class="media-usage"><span>TTS API · {serviceHealth.tts.status === 'ok' ? 'online' : serviceHealth.tts.status}{serviceHealth.tts.model ? ` · ${serviceHealth.tts.model}` : ''}</span></div>{/if}
        </fieldset>
      </div>

      <div id="settings-panel-appearance" class="settings-tab-panel" class:active={activeTab === 'appearance'} role="tabpanel" aria-labelledby="settings-tab-appearance">
        <AvatarSettings bind:appearance={settings.appearance} onnotify={notify} onuploaded={(id) => avatarKeepIds = [...avatarKeepIds, id]} />
      </div>

      <div id="settings-panel-tools" class="settings-tab-panel" class:active={activeTab === 'tools'} role="tabpanel" aria-labelledby="settings-tab-tools">
        <fieldset>
          <legend>웹·미디어 도구</legend>
          <label class="check"><input type="checkbox" bind:checked={settings.tools.enabled} /> web_search / web_fetch 활성화</label>
          <label class="check"><input type="checkbox" bind:checked={settings.tools.media_import_enabled} /> URL 미디어 자동 가져오기</label>
          <label>최대 호출 라운드<input type="number" min="1" max="8" bind:value={settings.tools.max_rounds} /></label>
          <label>검색 결과 수<input type="number" min="1" max="10" bind:value={settings.tools.search_results} /></label>
          <label>도구 타임아웃<input bind:value={settings.tools.timeout} placeholder="15s" /></label>
        </fieldset>
        <fieldset>
          <legend>SparkTalk Extra</legend>
          <label class="check"><input type="checkbox" bind:checked={settings.extra.ssh_enabled} /> 승인형 SSH 도구 활성화</label>
          <label>Extra SSH endpoint<input bind:value={settings.extra.ssh_endpoint} placeholder="http://127.0.0.1:8699" /></label>
          {#if serviceHealth?.extra?.ssh}<div class="media-usage"><span>Extra SSH · {serviceHealth.extra.ssh.status === 'ok' ? 'online' : serviceHealth.extra.ssh.status}</span></div>{/if}
          <SSHSettings onnotify={notify} />
        </fieldset>
      </div>

      <div id="settings-panel-app" class="settings-tab-panel" class:active={activeTab === 'app'} role="tabpanel" aria-labelledby="settings-tab-app">
        <label>Listen address<input bind:value={settings.server.listen_addr} placeholder="0.0.0.0:8585" /></label>
        <label>SQLite 파일<input bind:value={settings.server.database} placeholder="sparktalk.db" /></label>
        <fieldset>
          <legend>미디어 보관</legend>
          {#if mediaUsage}<div class="media-usage"><span>전체 {mediaUsage.files}개 · {formatBytes(mediaUsage.bytes)}</span><span>미사용 {mediaUsage.unused_files}개 · {formatBytes(mediaUsage.unused_bytes)}</span></div><button class="media-cleanup" onclick={removeUnusedMedia} disabled={cleaningMedia || !mediaUsage.unused_files}>{cleaningMedia ? '정리 중…' : '미사용 미디어 정리'}</button>{:else}<span class="media-loading">보관 현황을 불러오는 중…</span>{/if}
          <small>현재 대화에 첨부됐거나 전송 대기 중인 이미지·음성·비디오는 유지합니다.</small>
        </fieldset>
        <p class="settings-help">Endpoint·모델·reasoning·시스템 프롬프트는 즉시 반영됩니다. Listen address와 DB 파일 변경은 재시작 후 반영됩니다.</p>
      </div>
    </div>
    <div class="modal-actions"><button class="secondary" onclick={onclose}>닫기</button><button class="primary" onclick={persistSettings}>저장</button></div>
  </div>
</div>
