<script>
  import { onDestroy, onMount } from 'svelte';
  import { cleanupMedia, getHealth, getMediaUsage, saveConfig } from '../api.js';
  import PromptPresetManager from './settings/PromptPresetManager.svelte';
  import AvatarSettings from './settings/AvatarSettings.svelte';
  import ThemeSettings from './settings/ThemeSettings.svelte';
  import SSHSettings from './settings/SSHSettings.svelte';
  import SettingsToast from './settings/SettingsToast.svelte';
  import { normalizePublicSettings } from '../lib/settings.js';
  import { modelCapabilities, normalizeReasoningEffort, thinkingToggleValue } from '../lib/model-capabilities.js';

  export let settings;
  export let runtime = null;
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
  let activeTab = 'chat';
  const settingsTabs = [
    { id: 'chat', label: '대화' },
    { id: 'voice', label: '음성' },
    { id: 'features', label: '기능' },
    { id: 'appearance', label: '외형' },
    { id: 'system', label: '시스템' },
  ];
  $: modelProfile = modelCapabilities(settings?.model?.model_type);
  $: gemmaThinkingValue = thinkingToggleValue(settings?.model?.reasoning_effort);
  $: if (settings?.model && modelProfile.family === 'qwen3.8') settings.model.reasoning_effort = normalizeReasoningEffort(settings.model.model_type, settings.model.reasoning_effort);
  $: currentBundle = runtime?.bundles?.find((bundle) => bundle.id === runtime?.selected_bundle) || runtime?.bundles?.[0];
  const ttsLanguages = ['auto', 'ko-KR', 'en-US', 'ja-JP', 'zh-CN', 'ar-MSA', 'ar-AE', 'ar-SA', 'de-DE', 'es-ES', 'fr-FR', 'hi-IN', 'it-IT', 'pt-BR', 'vi-VN'];
  const ttsVoices = [
      { value: 'Sofia', label: 'Sofia · 여성' },
      { value: 'Aria', label: 'Aria · 여성' },
      { value: 'John', label: 'John · 남성' },
      { value: 'Jason', label: 'Jason · 남성' },
      { value: 'Leo', label: 'Leo · 남성' },
    ];

  onMount(async () => {
    normalizePublicSettings(settings);
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

  function toggleDefaultThinking() {
    settings.model.reasoning_effort = gemmaThinkingValue === 'on' ? 'none' : 'on';
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

  function formatBytes(value) {
    if (!value) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB'];
    const index = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1);
    return `${(value / (1024 ** index)).toFixed(index ? 1 : 0)} ${units[index]}`;
  }

  function formatGiB(value) {
    return Number.isFinite(Number(value)) ? Number(value).toFixed(1) : '—';
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
        version: settings.version,
        server: settings.server,
        runtime: settings.runtime,
        model: settings.model,
        asr: settings.asr,
        tts: settings.tts,
        context: settings.context,
        tools: settings.tools,
        image: settings.image,
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
      <div id="settings-panel-chat" class="settings-tab-panel" class:active={activeTab === 'chat'} role="tabpanel" aria-labelledby="settings-tab-chat">
        <fieldset>
          <legend>현재 AI 세트</legend>
          <div class="settings-bundle-card"><span><strong>{currentBundle?.name || '관리형 세트'}</strong><small>{currentBundle?.description || settings.model.default_model}</small></span><div><b>{currentBundle?.model_id || settings.model.default_model}</b><small>{currentBundle?.context_tokens ? `${Math.round(currentBundle.context_tokens / 1024)}K context` : ''}</small></div></div>
          <small>모델과 엔진 연결은 우상단의 운영 패널에서 관리합니다. 설정에는 대화 동작만 저장됩니다.</small>
        </fieldset>
        <fieldset>
          <legend>추론 기본값</legend>
          {#if modelProfile.reasoning === 'toggle'}
            <div class="settings-toggle-field"><span>기본 Thinking</span><button type="button" class:active={gemmaThinkingValue === 'on'} onclick={toggleDefaultThinking} aria-pressed={gemmaThinkingValue === 'on'}>{gemmaThinkingValue === 'on' ? 'Thinking 켜짐' : 'Thinking 꺼짐'}</button><small>Gemma 4는 단계별 effort를 지원하지 않습니다.</small></div>
            <label>Thinking 예산<input type="number" min="0" step="128" bind:value={settings.model.thinking_budget} /><small>최대 생각 토큰 수입니다. 512 권장, 0이면 제한하지 않습니다.</small></label>
          {:else if modelProfile.family === 'qwen3.8'}
            <label>기본 reasoning effort<select bind:value={settings.model.reasoning_effort} aria-label="기본 reasoning effort">{#each modelProfile.reasoningLevels as level}<option value={level}>{level === 'none' ? '꺼짐' : level === 'low' ? 'Low' : level === 'medium' ? 'Medium' : 'XHigh'}</option>{/each}</select><small>Qwen3.8은 꺼짐·Low·Medium·XHigh 네 단계만 사용합니다.</small></label>
          {:else}
            <label>기본 reasoning effort<input bind:value={settings.model.reasoning_effort} list="settings-reasoning-levels" placeholder="직접 입력 또는 목록에서 선택" /></label>
            <datalist id="settings-reasoning-levels">{#each modelProfile.reasoningLevels as level}<option value={level}></option>{/each}</datalist>
          {/if}
        </fieldset>
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
          <label class="check"><input type="checkbox" bind:checked={settings.asr.enabled} /> 마이크와 첨부 미디어의 음성을 전사</label>
          <div class="settings-section-title">인식 정확도</div>
          <div class="settings-form-row two">
            <label>마이크 발화 언어<input bind:value={settings.asr.voice_language} list="asr-languages" placeholder="ko-KR" /></label>
            <label>첨부 미디어 언어<input bind:value={settings.asr.media_language} list="asr-languages" placeholder="auto" /></label>
          </div>
          <datalist id="asr-languages">{#each ['auto', 'ko-KR', 'ja-JP', 'en-US', 'en-GB', 'zh-CN', 'es-US', 'es-ES', 'fr-FR', 'fr-CA', 'it-IT', 'pt-BR', 'pt-PT', 'nl-NL', 'de-DE', 'tr-TR', 'ru-RU', 'ar-AR', 'hi-IN', 'vi-VN', 'uk-UA', 'pl-PL', 'sv-SE', 'cs-CZ', 'nb-NO', 'da-DK', 'bg-BG', 'fi-FI', 'hr-HR', 'sk-SK', 'hu-HU', 'ro-RO', 'et-EE', 'Filipino', 'Cantonese', 'Thai', 'Indonesian', 'Malay', 'Persian', 'Greek'] as language}<option value={language}></option>{/each}</datalist>
          <small class="settings-inline-help">마이크는 <code>ko-KR</code>, 어떤 언어가 나올지 모르는 영상·음성은 <code>auto</code>가 기본입니다. 직접 입력도 가능합니다.</small>
          <label>문맥·전문용어 힌트<textarea rows="3" bind:value={settings.asr.prompt} placeholder="예: 한국어 기술 대화. 주요 용어: SparkTalk, DGX Spark, SGLang, Qwen3-ASR"></textarea></label>
          <small class="settings-inline-help">자주 말하는 제품명·인명·약어를 정확한 표기로 적으십시오. 긴 지시문보다 짧은 문맥과 용어 목록이 적합합니다.</small>
          <label class="check"><input type="checkbox" bind:checked={settings.asr.filter_fillers} /> 음성대기에서 단독 추임새·문장부호 무시</label>
          {#if serviceHealth?.asr}<div class="media-usage"><span>Media API · {serviceHealth.asr.ffmpeg?.status === 'ok' ? 'online' : serviceHealth.asr.ffmpeg?.status}</span><span>ASR API · {serviceHealth.asr.asr?.status === 'ok' ? 'online' : serviceHealth.asr.asr?.status}</span></div>{/if}
          <small>현재 엔진: Nemotron ASR · SparkTalk Extra Media</small>
          <small>음성 원본은 모델에 보내지 않고 전사문으로 대체합니다. 영상은 화면 정보와 전사문을 함께 보냅니다.</small>
        </fieldset>
        <fieldset>
          <legend>답변 음성</legend>
          <label class="check"><input type="checkbox" bind:checked={settings.tts.enabled} /> TTS로 AI 답변 읽기</label>
          <label class="check"><input type="checkbox" bind:checked={settings.tts.auto_play} disabled={!settings.tts.enabled} /> 답변 완료 후 자동 재생</label>
          <label class="check"><input type="checkbox" bind:checked={settings.tts.omit_parentheticals} disabled={!settings.tts.enabled} /> 괄호 속 부연설명 읽지 않기</label>
          <small class="settings-inline-help">켜면 자동·수동 읽기 모두에서 <code>(한경)</code>, <code>(온라인)</code> 같은 괄호 내용을 제외합니다. 화면 원문은 바뀌지 않습니다.</small>
          <div class="settings-form-row three">
            <label>언어<select bind:value={settings.tts.language}>{#each ttsLanguages as language}<option value={language}>{language}</option>{/each}</select></label>
            <label>자동 한자 독음<select bind:value={settings.tts.hanja_reading}><option value="korean">한국어</option><option value="japanese">일본어</option><option value="chinese">중국어</option></select></label>
            <label>화자<select bind:value={settings.tts.voice}>{#each ttsVoices as voice}<option value={typeof voice === 'string' ? voice : voice.value}>{typeof voice === 'string' ? voice : voice.label}</option>{/each}</select></label>
          </div>
          <small class="settings-inline-help"><code>auto</code>는 지원 문자를 직접 구분하고 라틴 문자 문장은 지원 언어 안에서 판별합니다. 순수 한자 구간은 선택한 한국어·일본어·중국어 독음으로 읽습니다. 가나가 포함된 문장은 자동으로 일본어로 판별합니다.</small>
          <small class="settings-inline-help">Magpie는 22,050 Hz PCM과 고정 음성을 사용합니다.</small>
          <small class="settings-inline-help">재생 중에는 마이크 판정을 멈춥니다.</small>
          {#if serviceHealth?.tts}<div class="media-usage"><span>TTS API · {serviceHealth.tts.status === 'ok' ? 'online' : serviceHealth.tts.status}{serviceHealth.tts.model ? ` · ${serviceHealth.tts.model}` : ''}</span></div>{/if}
          <small>현재 엔진: Magpie TTS</small>
        </fieldset>
      </div>

      <div id="settings-panel-appearance" class="settings-tab-panel" class:active={activeTab === 'appearance'} role="tabpanel" aria-labelledby="settings-tab-appearance">
        <ThemeSettings bind:appearance={settings.appearance} />
        <AvatarSettings bind:appearance={settings.appearance} onnotify={notify} onuploaded={(id) => avatarKeepIds = [...avatarKeepIds, id]} />
      </div>

      <div id="settings-panel-features" class="settings-tab-panel" class:active={activeTab === 'features'} role="tabpanel" aria-labelledby="settings-tab-features">
        <fieldset>
          <legend>이미지 생성</legend>
          <label class="check"><input type="checkbox" bind:checked={settings.image.enabled} /> 대화형 이미지 생성·편집 도구 활성화</label>
          <label>기본 해상도<input bind:value={settings.image.default_size} placeholder="1024x1024" /></label>
          <label>기능 수준<select bind:value={settings.image.mode}>
            <option value="basic">기본 생성</option>
            <option value="extended">확장 생성·편집</option>
          </select></label>
          {#if serviceHealth?.image}<div class="media-usage"><span>이미지 API · {serviceHealth.image.status === 'ok' ? 'online' : serviceHealth.image.status}{serviceHealth.image.model ? ` · ${serviceHealth.image.model}` : ''}</span></div>{/if}
          <small>현재 엔진: FLUX.2 Klein 4B. 엔진 기동과 상태는 운영 패널에서 관리합니다.</small>
        </fieldset>
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
          {#if serviceHealth?.extra?.ssh}<div class="media-usage"><span>Extra SSH · {serviceHealth.extra.ssh.status === 'ok' ? 'online' : serviceHealth.extra.ssh.status}</span></div>{/if}
          {#if !settings.extra.ssh_enabled}
            <small class="ssh-empty">SSH 도구가 꺼져 있습니다.</small>
          {:else if serviceHealth?.extra?.ssh?.status === 'ok'}
            <SSHSettings onnotify={notify} />
          {:else if serviceHealth}
            <small class="ssh-security-note">SparkTalk Extra가 오프라인입니다. 서비스를 기동하면 키와 서버 설정을 불러옵니다.</small>
          {:else}
            <small class="media-loading">SparkTalk Extra 연결을 확인하는 중…</small>
          {/if}
        </fieldset>
      </div>

      <div id="settings-panel-system" class="settings-tab-panel" class:active={activeTab === 'system'} role="tabpanel" aria-labelledby="settings-tab-system">
        <fieldset>
          <legend>DGX Spark 운영</legend>
          <label>기본 AI 세트<select bind:value={settings.runtime.bundle}>{#each runtime?.bundles || [] as bundle}<option value={bundle.id}>{bundle.name}</option>{/each}</select><small>실행 중 세트 전환은 우상단 운영 패널에서 진행합니다.</small></label>
          <label class="check"><input type="checkbox" bind:checked={settings.runtime.auto_start} /> SparkTalk 시작 시 기본 세트 자동 기동</label>
          <label>최소 확보 메모리<input type="number" min="1" max="64" step="1" bind:value={settings.runtime.memory_reserve_gib} /><small>새 엔진을 올릴 때 남겨둘 통합메모리 GiB입니다.</small></label>
          <label>운영 데이터 폴더<input bind:value={settings.runtime.data_dir} /></label>
          <label>모델 캐시 폴더<input bind:value={settings.runtime.model_cache} /></label>
          <div class="media-usage"><span>Docker · {runtime?.docker === 'online' ? 'online' : 'offline'}</span><span>시스템 가용 · {formatGiB(runtime?.memory?.available_gib)} GiB</span><span>즉시 여유 · {formatGiB(runtime?.memory?.free_gib)} GiB</span></div>
        </fieldset>
        <fieldset>
          <legend>앱 서버</legend>
          <label>Listen address<input bind:value={settings.server.listen_addr} placeholder="0.0.0.0:8585" /></label>
          <label>SQLite 파일<input bind:value={settings.server.database} placeholder="sparktalk.db" /></label>
        </fieldset>
        <fieldset>
          <legend>미디어 보관</legend>
          {#if mediaUsage}<div class="media-usage"><span>전체 {mediaUsage.files}개 · {formatBytes(mediaUsage.bytes)}</span><span>미사용 {mediaUsage.unused_files}개 · {formatBytes(mediaUsage.unused_bytes)}</span></div><button class="media-cleanup" onclick={removeUnusedMedia} disabled={cleaningMedia || !mediaUsage.unused_files}>{cleaningMedia ? '정리 중…' : '미사용 미디어 정리'}</button>{:else}<span class="media-loading">보관 현황을 불러오는 중…</span>{/if}
          <small>현재 대화에 첨부됐거나 전송 대기 중인 이미지·음성·비디오는 유지합니다.</small>
        </fieldset>
        <p class="settings-help">대화·음성·도구 설정은 즉시 반영됩니다. Listen address와 DB 파일 변경은 재시작 후 반영됩니다.</p>
      </div>
    </div>
    <div class="modal-actions"><button class="secondary" onclick={onclose}>닫기</button><button class="primary" onclick={persistSettings}>저장</button></div>
  </div>
</div>
