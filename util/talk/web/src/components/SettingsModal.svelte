<script>
  import MicrophoneHelp from './MicrophoneHelp.svelte';
  import SettingsField from './settings/SettingsField.svelte';
  import SettingsHelp from './settings/SettingsHelp.svelte';
  import { onDestroy, onMount } from 'svelte';
  import { cleanupMedia, getHealth, getMediaUsage, saveConfig } from '../api.js';
  import CharacterSettings from './settings/CharacterSettings.svelte';
  import AvatarSettings from './settings/AvatarSettings.svelte';
  import ThemeSettings from './settings/ThemeSettings.svelte';
  import SupportServices from './settings/SupportServices.svelte';
  import ModelDownloads from './settings/ModelDownloads.svelte';
  import RuntimeSetEditor from './settings/RuntimeSetEditor.svelte';
  import SSHSettings from './settings/SSHSettings.svelte';
  import SettingsToast from './settings/SettingsToast.svelte';
  import MemorySettings from './settings/MemorySettings.svelte';
  import ToolDiscoverySettings from './settings/ToolDiscoverySettings.svelte';
  import { applyExternalModelType, normalizePublicSettings } from '../lib/settings.js';
  import { modelCapabilities, normalizeReasoningEffort, reasoningEffortLabel, thinkingToggleValue } from '../lib/model-capabilities.js';

  export let settings;
  export let runtime = null;
  export let keepMediaIds = [];
  export let onclose = () => {};
  export let onManageSkills = () => {};
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
  let systemSection = 'connection';
  let featureSection = 'web';
  let profileSection = 'character';
  const settingsTabs = [
    { id: 'chat', label: '대화' },
    { id: 'profile', label: '프로필' },
    { id: 'memory', label: '기억' },
    { id: 'voice', label: '음성' },
    { id: 'features', label: '기능' },
    { id: 'system', label: '시스템' },
  ];
  $: modelProfile = modelCapabilities(settings?.model?.model_type);
  $: gemmaThinkingValue = thinkingToggleValue(settings?.model?.reasoning_effort);
  $: if (settings?.model && (modelProfile.family === 'qwen3.8' || modelProfile.family === 'qwen3.8-exl3' || (modelProfile.family === 'glm5.3' || modelProfile.family === 'deepseek-v4'))) settings.model.reasoning_effort = normalizeReasoningEffort(settings.model.model_type, settings.model.reasoning_effort);
  $: externalMode = settings?.runtime?.mode === 'external';
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

  function selectExternalModelType(event) {
    applyExternalModelType(settings, event.currentTarget.value);
    settings = settings;
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
        memory: settings.memory,
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

<div class="modal-backdrop" role="presentation">
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
          {#if externalMode}
            <div class="settings-bundle-card"><span><strong>{(modelProfile.family === 'glm5.3' || modelProfile.family === 'deepseek-v4') ? (modelProfile.family === 'deepseek-v4' ? 'DeepSeek V4 Flash Vision Exp' : 'GLM-5.3 Flash') : '외부 모델'}</strong><small>{settings.model.endpoint}</small></span><div><b>{settings.model.default_model}</b><small>{settings.context.window_tokens ? `${Math.round(settings.context.window_tokens / 1024)}K context` : 'context 자동 감지'}</small></div></div>
            <small>외부 API는 SparkTalk가 기동하거나 중지하지 않습니다.</small>
          {:else}
            <div class="settings-bundle-card"><span><strong>{currentBundle?.name || '관리형 세트'}</strong><small>{currentBundle?.description || settings.model.default_model}</small></span><div><b>{currentBundle?.model_id || settings.model.default_model}</b><small>{currentBundle?.context_tokens ? `${Math.round(currentBundle.context_tokens / 1024)}K context` : ''}</small></div></div>
            <small>모델과 엔진 연결은 우상단의 운영 패널에서 관리합니다. 설정에는 대화 동작만 저장됩니다.</small>
          {/if}
        </fieldset>
        <fieldset>
          <legend><span>추론 기본값</span> <SettingsHelp title="추론 기본값">
            {#if modelProfile.family === 'gemma4'}<p>Gemma 4는 Thinking을 켜거나 끄며, 단계별 추론 강도는 지원하지 않습니다.</p><p>Thinking 예산은 최대 생각 토큰 수입니다. 512 권장, 0이면 제한하지 않습니다.</p>
            {:else if modelProfile.family === 'qwen3.8' || modelProfile.family === 'qwen3.8-exl3'}<p>Qwen3.8은 Thinking 꺼짐과 Low·Medium·XHigh 단계를 지원합니다.</p>
            {:else if modelProfile.family === 'glm5.3' || modelProfile.family === 'deepseek-v4'}<p>{modelProfile.family === 'deepseek-v4' ? 'DeepSeek V4' : 'GLM-5.3 Flash'}는 꺼짐·Low·High·Max를 사용합니다.</p>
            {:else}<p>연결한 모델이 지원하는 추론 강도를 선택하거나 직접 입력합니다.</p>{/if}
          </SettingsHelp></legend>
          {#if modelProfile.reasoning === 'toggle'}
            <div class="settings-toggle-field"><span>기본 Thinking</span><button type="button" class:active={gemmaThinkingValue === 'on'} onclick={toggleDefaultThinking} aria-pressed={gemmaThinkingValue === 'on'}>{gemmaThinkingValue === 'on' ? 'Thinking 켜짐' : 'Thinking 꺼짐'}</button></div>
            {#if modelProfile.family === 'gemma4'}<label class="settings-field-row settings-number-row"><span>Thinking 예산</span><input type="number" min="0" step="128" bind:value={settings.model.thinking_budget} /></label>{/if}
          {:else if modelProfile.family === 'qwen3.8' || modelProfile.family === 'qwen3.8-exl3'}
            <label class="settings-field-row"><span>기본 추론 강도</span><select bind:value={settings.model.reasoning_effort} aria-label="기본 reasoning effort">{#each modelProfile.reasoningLevels as level}<option value={level}>{reasoningEffortLabel(level, settings.model.model_type)}</option>{/each}</select></label>
          {:else if (modelProfile.family === 'glm5.3' || modelProfile.family === 'deepseek-v4')}
            <label class="settings-field-row"><span>기본 추론 강도</span><select bind:value={settings.model.reasoning_effort} aria-label="기본 reasoning effort">{#each modelProfile.reasoningLevels as level}<option value={level}>{reasoningEffortLabel(level)}</option>{/each}</select></label>
          {:else}
            <label class="settings-field-row"><span>기본 추론 강도</span><input bind:value={settings.model.reasoning_effort} list="settings-reasoning-levels" placeholder="직접 입력 또는 목록에서 선택" /></label>
            <datalist id="settings-reasoning-levels">{#each modelProfile.reasoningLevels as level}<option value={level}></option>{/each}</datalist>
          {/if}
        </fieldset>
        <fieldset class="context-settings">
          <legend><span>지능형 문맥 관리</span> <SettingsHelp title="지능형 문맥 관리"><p>문맥 한도는 실행 중인 모델 서버가 지원하는 값 이내로 설정하세요. 0이면 서버에서 자동 감지합니다.</p><p>입력 예산 = 문맥 한도 − 최대 출력 − 안전 여유. 입력 예산에 자동 정리 시작 비율(50~95%)을 곱한 지점에서 오래된 원문을 요약합니다.</p><p>최대 출력은 생각 과정을 포함한 한 번의 응답 상한입니다. 최근 원문 유지와 이미지 추정값도 토큰 단위입니다.</p><p>대화와 첨부 원본은 보관되며, 모델로 보내는 활성 문맥만 정리합니다.</p></SettingsHelp></legend>
          <label class="check"><input type="checkbox" bind:checked={settings.context.enabled} /> 오래된 원문을 구조화 요약으로 전환</label>
          <div class="context-fields">
            <label class="context-row"><span>문맥 한도 <small>0 = 자동 감지</small></span><span class="context-number"><input aria-label="모델 context window (0은 백엔드 자동 감지)" type="number" min="0" step="1024" bind:value={settings.context.window_tokens} /><small>토큰</small></span></label>
            <label class="context-row"><span>자동 정리 시작 비율</span><span class="context-number"><input aria-label="자동 정리 시작 비율" type="number" min="50" max="95" bind:value={settings.context.compact_at_percent} /><small>%</small></span></label>
            <label class="context-row"><span>최대 출력 <small>생각 과정 포함</small></span><span class="context-number"><input aria-label="최대 출력 토큰 (생각 과정 포함)" type="number" min="256" step="256" bind:value={settings.context.output_reserve} /><small>토큰</small></span></label>
            <label class="context-row"><span>안전 여유</span><span class="context-number"><input aria-label="안전 여유 토큰" type="number" min="256" step="256" bind:value={settings.context.safety_margin} /><small>토큰</small></span></label>
            <label class="context-row"><span>최근 원문 유지</span><span class="context-number"><input aria-label="최근 원문 유지 토큰" type="number" min="256" step="256" bind:value={settings.context.recent_tokens} /><small>토큰</small></span></label>
            <label class="context-row"><span>이미지 1장당 추정</span><span class="context-number"><input aria-label="이미지 장당 보수적 추정 토큰" type="number" min="1" step="128" bind:value={settings.context.image_tokens} /><small>토큰</small></span></label>
          </div>

        </fieldset>
      </div>

      <div id="settings-panel-profile" class="settings-tab-panel" class:active={activeTab === 'profile'} role="tabpanel" aria-labelledby="settings-tab-profile">
        <div class="settings-section-navigation" role="group" aria-label="프로필 구분">
          <button type="button" aria-pressed={profileSection === 'character'} onclick={() => profileSection = 'character'}>AI 캐릭터</button>
          <button type="button" aria-pressed={profileSection === 'user'} onclick={() => profileSection = 'user'}>내 프로필</button>
        </div>
        <div id="settings-profile-character" class="settings-section" hidden={profileSection !== 'character'}>
          <CharacterSettings bind:model={settings.model} bind:appearance={settings.appearance} onnotify={notify} onuploaded={(id) => avatarKeepIds = [...avatarKeepIds, id]} />
        </div>
        <div class="settings-section" hidden={profileSection !== 'user'}>
          <fieldset><legend>내 프로필</legend><label class="settings-field-row"><span>내 이름</span><input bind:value={settings.appearance.user_name} maxlength="80" placeholder="나" /></label></fieldset>
          <AvatarSettings role="user" bind:appearance={settings.appearance} onnotify={notify} onuploaded={(id) => avatarKeepIds = [...avatarKeepIds, id]} />
        </div>
      </div>

      <div id="settings-panel-memory" class="settings-tab-panel" class:active={activeTab === 'memory'} role="tabpanel" aria-labelledby="settings-tab-memory">
        <MemorySettings config={settings.memory} />
      </div>

      <div id="settings-panel-voice" class="settings-tab-panel" class:active={activeTab === 'voice'} role="tabpanel" aria-labelledby="settings-tab-voice">
        <fieldset>
          <legend><span>음성 인식</span> <SettingsHelp title="음성 인식"><p>마이크는 <code>ko-KR</code>, 어떤 언어가 나올지 모르는 영상·음성은 <code>auto</code>가 기본입니다. 직접 입력도 가능합니다.</p><p>자주 말하는 제품명·인명·약어를 정확한 표기로 적으십시오. 긴 지시문보다 짧은 문맥과 용어 목록이 적합합니다.</p><p>음성 원본은 모델에 보내지 않고 전사문으로 대체합니다. 영상은 화면 정보와 전사문을 함께 보냅니다.</p></SettingsHelp></legend>
          <label class="check"><input type="checkbox" bind:checked={settings.asr.enabled} /> 마이크와 첨부 미디어의 음성을 전사</label>
          <MicrophoneHelp />
          <div class="settings-section-title">인식 정확도</div>
          <div class="settings-form-row two">
            <label class="settings-field-row"><span>마이크 발화 언어</span><input bind:value={settings.asr.voice_language} list="asr-languages" placeholder="ko-KR" /></label>
            <label class="settings-field-row"><span>첨부 미디어 언어</span><input bind:value={settings.asr.media_language} list="asr-languages" placeholder="auto" /></label>
          </div>
          <datalist id="asr-languages">{#each ['auto', 'ko-KR', 'ja-JP', 'en-US', 'en-GB', 'zh-CN', 'es-US', 'es-ES', 'fr-FR', 'fr-CA', 'it-IT', 'pt-BR', 'pt-PT', 'nl-NL', 'de-DE', 'tr-TR', 'ru-RU', 'ar-AR', 'hi-IN', 'vi-VN', 'uk-UA', 'pl-PL', 'sv-SE', 'cs-CZ', 'nb-NO', 'da-DK', 'bg-BG', 'fi-FI', 'hr-HR', 'sk-SK', 'hu-HU', 'ro-RO', 'et-EE', 'Filipino', 'Cantonese', 'Thai', 'Indonesian', 'Malay', 'Persian', 'Greek'] as language}<option value={language}></option>{/each}</datalist>

          <label>문맥·전문용어 힌트<textarea rows="3" bind:value={settings.asr.prompt} placeholder="예: 한국어 기술 대화. 주요 용어: SparkTalk, DGX Spark, SGLang, Qwen3-ASR"></textarea></label>

          <label class="check"><input type="checkbox" bind:checked={settings.asr.filter_fillers} /> 음성대기에서 단독 추임새·문장부호 무시</label>
          {#if serviceHealth?.asr}<div class="media-usage"><span>Media API · {serviceHealth.asr.ffmpeg?.status === 'ok' ? 'online' : serviceHealth.asr.ffmpeg?.status}</span><span>ASR API · {serviceHealth.asr.asr?.status === 'ok' ? 'online' : serviceHealth.asr.asr?.status}</span></div>{/if}
          <small>현재 엔진: Nemotron ASR · SparkTalk Extra Media</small>

        </fieldset>
        <fieldset>
          <legend><span>답변 음성</span> <SettingsHelp title="답변 음성"><p>켜면 자동·수동 읽기 모두에서 <code>(한경)</code>, <code>(온라인)</code> 같은 괄호 내용을 제외합니다. 화면 원문은 바뀌지 않습니다.</p><p><code>auto</code>는 지원 문자를 직접 구분하고 라틴 문자 문장은 지원 언어 안에서 판별합니다. 순수 한자 구간은 선택한 한국어·일본어·중국어 독음으로 읽습니다. 가나가 포함된 문장은 자동으로 일본어로 판별합니다.</p><p>Magpie는 22,050 Hz PCM과 고정 음성을 사용합니다.</p><p>재생 중에는 마이크 판정을 멈춥니다.</p></SettingsHelp></legend>
          <label class="check"><input type="checkbox" bind:checked={settings.tts.enabled} /> TTS로 AI 답변 읽기</label>
          <label class="check"><input type="checkbox" bind:checked={settings.tts.auto_play} disabled={!settings.tts.enabled} /> 답변 완료 후 자동 재생</label>
          <label class="check"><input type="checkbox" bind:checked={settings.tts.omit_parentheticals} disabled={!settings.tts.enabled} /> 괄호 속 부연설명 읽지 않기</label>

          <div class="settings-form-row three">
            <label class="settings-field-row"><span>언어</span><select bind:value={settings.tts.language}>{#each ttsLanguages as language}<option value={language}>{language}</option>{/each}</select></label>
            <label class="settings-field-row"><span>자동 한자 독음</span><select bind:value={settings.tts.hanja_reading}><option value="korean">한국어</option><option value="japanese">일본어</option><option value="chinese">중국어</option></select></label>
            <label class="settings-field-row"><span>화자</span><select bind:value={settings.tts.voice}>{#each ttsVoices as voice}<option value={typeof voice === 'string' ? voice : voice.value}>{typeof voice === 'string' ? voice : voice.label}</option>{/each}</select></label>
          </div>



          {#if serviceHealth?.tts}<div class="media-usage"><span>TTS API · {serviceHealth.tts.status === 'ok' ? 'online' : serviceHealth.tts.status}{serviceHealth.tts.model ? ` · ${serviceHealth.tts.model}` : ''}</span></div>{/if}
          <small>현재 엔진: Magpie TTS</small>
        </fieldset>
      </div>

      <div id="settings-panel-features" class="settings-tab-panel" class:active={activeTab === 'features'} role="tabpanel" aria-labelledby="settings-tab-features">
        <div class="settings-section-navigation" role="group" aria-label="기능 설정 분류">
          <button type="button" aria-pressed={featureSection === 'web'} onclick={() => { featureSection = 'web'; }}>웹·미디어</button>
          <button type="button" aria-pressed={featureSection === 'documents'} onclick={() => { featureSection = 'documents'; }}>문서</button>
          <button type="button" aria-pressed={featureSection === 'image'} onclick={() => { featureSection = 'image'; }}>이미지</button>
          <button type="button" aria-pressed={featureSection === 'skills'} onclick={() => { featureSection = 'skills'; }}>스킬·기록</button>
          <button type="button" aria-pressed={featureSection === 'ssh'} onclick={() => { featureSection = 'ssh'; }}>SSH·키</button>
        </div>
        <div class="settings-section" hidden={featureSection !== 'documents'}>
          <fieldset><legend><span>문서 생성</span> <SettingsHelp title="문서 생성"><p>보고서는 본문·표·첨부 이미지, 발표자료는 제목·목록을 지원합니다. 스프레드시트는 여러 시트·기본 수식·숫자 서식을 지원합니다. Office 파일과 같은 내용의 PDF도 생성합니다. 원본과 배치가 다를 수 있으며 PDF 생성 실패 시 원본만 반환합니다. 서비스 주소와 실행 상태는 시스템의 지원 서비스에서 관리합니다.</p></SettingsHelp></legend>
            <label class="check"><input type="checkbox" bind:checked={settings.extra.documents_enabled} /> DOCX·PPTX·XLSX·PDF·HWP·HWPX 파일 생성 활성화</label>
            <button type="button" onclick={() => { activeTab = "system"; systemSection = "services"; }}>문서 서비스 상태·실행 관리</button>

          </fieldset>
        </div>
        <div class="settings-section" hidden={featureSection !== 'image'}>
        <fieldset>
          <legend>이미지 생성</legend>
          <label class="check"><input type="checkbox" bind:checked={settings.image.enabled} /> 대화형 이미지 생성·편집 도구 활성화</label>
          <label class="settings-field-row"><span>기본 해상도</span><input bind:value={settings.image.default_size} placeholder="1024x1024" /></label>
          <label class="settings-field-row"><span>기능 수준</span><select bind:value={settings.image.mode}>
            <option value="basic">기본 생성</option>
            <option value="extended">확장 생성·편집</option>
          </select></label>
          {#if serviceHealth?.image}<div class="media-usage"><span>이미지 API · {serviceHealth.image.status === 'ok' ? 'online' : serviceHealth.image.status}{serviceHealth.image.model ? ` · ${serviceHealth.image.model}` : ''}</span></div>{/if}
          <small>현재 엔진: FLUX.2 Klein 4B. 엔진 기동과 상태는 운영 패널에서 관리합니다.</small>
        </fieldset>
        </div>
        <div class="settings-section" hidden={featureSection !== 'web'}>
        <fieldset>
          <legend><span>웹·미디어 도구</span> <SettingsHelp title="웹·미디어 도구"><p>기본 24 · 최대 64. 한 라운드에 여러 도구를 호출할 수 있으며, 작업 절차에서는 단계마다 적용됩니다.</p><p>검색 1회당 최대 건수입니다. 기본 15 · 최대 30이며, 검색엔진의 실제 결과 수에 따라 줄어듭니다.</p><p>0보다 큰 시간으로 입력합니다. 예: 15s(15초), 1m(1분). 기본 15초이며 웹 검색·페이지 읽기에 적용됩니다.</p></SettingsHelp></legend>
          <button type="button" onclick={() => { activeTab = "system"; systemSection = "services"; }}>웹·미디어 서비스 상태·실행 관리</button>
          <label class="check"><input type="checkbox" bind:checked={settings.tools.enabled} /> web_search / web_fetch 활성화</label>
          <label class="check"><input type="checkbox" bind:checked={settings.tools.media_import_enabled} /> URL 미디어 자동 가져오기</label>
          <label class="check"><input type="checkbox" bind:checked={settings.extra.collector_enabled} /> 격리 브라우저 Collector 활성화</label>
          <label class="settings-field-row settings-number-row"><span>최대 호출 라운드 (≤ 64)</span><input type="number" min="1" max="64" bind:value={settings.tools.max_rounds} /></label>

          <label class="settings-field-row settings-number-row"><span>검색 결과 수 (≤ 30)</span><input type="number" min="1" max="30" bind:value={settings.tools.search_results} /></label>

          <label class="settings-field-row"><span>도구 타임아웃 (> 0초)</span><input bind:value={settings.tools.timeout} placeholder="15s" /></label>

        </fieldset>
        </div>
        <div class="settings-section" hidden={featureSection !== 'skills'}>
        <ToolDiscoverySettings onManage={onManageSkills} bind:enabled={settings.tools.skills_enabled} onnotify={notify} />
        </div>
        <div class="settings-section" hidden={featureSection !== 'ssh'}>
        <fieldset>
          <legend>SSH 도구·인증 키</legend>
          <button type="button" onclick={() => { activeTab = "system"; systemSection = "services"; }}>SSH 서비스 상태·실행 관리</button>
          <label class="check"><input type="checkbox" bind:checked={settings.extra.ssh_enabled} /> 승인형 SSH 도구 활성화</label>
          {#if serviceHealth?.extra?.ssh}<div class="media-usage"><span>Extra SSH · {serviceHealth.extra.ssh.status === 'ok' ? 'online' : serviceHealth.extra.ssh.status}</span></div>{/if}
          {#if settings.runtime.key_store_hosts?.length || (settings.extra.ssh_enabled && serviceHealth?.extra?.ssh?.status === 'ok')}
            <SSHSettings onnotify={notify} onkeystorechange={(state) => { settings.runtime.key_store_hosts = state.hosts; settings.runtime.key_store_peers = state.peers; settings = settings; }} />
          {:else if !settings.extra.ssh_enabled}
            <small class="ssh-empty">SSH 도구가 꺼져 있습니다.</small>
          {:else if serviceHealth}
            <small class="ssh-security-note">SparkTalk Extra가 오프라인입니다. 서비스를 기동하면 키와 서버 설정을 불러옵니다.</small>
          {:else}
            <small class="media-loading">SparkTalk Extra 연결을 확인하는 중…</small>
          {/if}
        </fieldset>
        </div>
      </div>

      <div id="settings-panel-system" class="settings-tab-panel" class:active={activeTab === 'system'} role="tabpanel" aria-labelledby="settings-tab-system">
        <div class="settings-section-navigation system-navigation" role="group" aria-label="시스템 설정 분류">
          <button type="button" aria-pressed={systemSection === 'connection'} onclick={() => { systemSection = 'connection'; }}>시작·연결</button>
          <button type="button" aria-pressed={systemSection === 'sets'} onclick={() => { systemSection = 'sets'; }}>AI 세트</button>
          <button type="button" aria-pressed={systemSection === 'services'} onclick={() => { systemSection = 'services'; }}>지원 서비스</button>
          <button type="button" aria-pressed={systemSection === 'downloads'} onclick={() => { systemSection = 'downloads'; }}>모델 준비</button>
          <button type="button" aria-pressed={systemSection === 'appearance'} onclick={() => { systemSection = 'appearance'; }}>외형</button>
          <button type="button" aria-pressed={systemSection === 'storage'} onclick={() => { systemSection = 'storage'; }}>앱·저장소</button>
        </div>
        <div class="settings-section" hidden={systemSection !== 'connection'}>
        <fieldset>
          <legend><span>모델 연결</span> <SettingsHelp title="모델 연결"><p>외부 API 주소에는 /v1을 붙이지 않습니다. 모델 유형에 맞는 응답 제어를 사용합니다.</p><p>세트 관리형에서는 기본 세트와 자동 기동을 설정합니다. 실행 중 세트 전환은 운영 패널에서 진행합니다.</p><p>최소 확보 메모리는 새 엔진을 올릴 때 남겨둘 통합메모리 GiB입니다.</p></SettingsHelp></legend>
          <label class="settings-field-row"><span>실행 방식</span><select bind:value={settings.runtime.mode}><option value="managed">세트 관리형 (로컬·원격)</option><option value="external">외부 OpenAI 호환 API</option></select></label>
          {#if externalMode}
            <label class="settings-field-row"><span>모델 API 주소</span><input bind:value={settings.model.endpoint} placeholder="http://서버주소:8000" /></label>
            <label class="settings-field-row"><span>모델 ID</span><input bind:value={settings.model.default_model} /></label>
            <label class="settings-field-row"><span>모델 유형</span><select value={settings.model.model_type} onchange={selectExternalModelType}><option value="glm5.3">GLM-5.3 Flash</option><option value="qwen3.8">Qwen3.8</option><option value="gemma4">Gemma 4</option><option value="deepseek-v4">DeepSeek V4</option><option value="generic">일반 OpenAI 호환</option></select></label>
            <label class="settings-field-row"><span>API 키</span><input type="password" bind:value={settingsAPIKey} autocomplete="new-password" placeholder={settings.api_key_set ? '저장된 키 유지' : '필요한 경우 입력'} /></label>
            {#if settings.api_key_set}<label class="check"><input type="checkbox" bind:checked={clearAPIKey} /> 저장된 API 키 삭제</label>{/if}
            <small>GLM-5.3 Flash를 선택하면 512K 문맥과 Max 리즈닝을 적용합니다. ASR·TTS·이미지 생성·Extra 등 부가 기능은 각 설정에서 개별 관리합니다.</small>
          {:else}
            <SettingsField title="기본 AI 세트"><select bind:value={settings.runtime.bundle}>{#each settings.runtime.catalog?.bundles || runtime?.bundles || [] as bundle}<option value={bundle.id}>{bundle.name}</option>{/each}</select><svelte:fragment slot="help"><p>실행 중 세트 전환은 우상단 운영 패널에서 진행합니다.</p></svelte:fragment></SettingsField>
            <label class="check"><input type="checkbox" bind:checked={settings.runtime.auto_start} /> SparkTalk 시작 시 기본 세트 자동 기동</label>
            <details class="system-advanced"><summary>메모리·모델 경로</summary>
            <label class="settings-field-row settings-number-row"><span>최소 확보 메모리</span><input type="number" min="1" max="64" step="1" bind:value={settings.runtime.memory_reserve_gib} /></label>
            <label class="settings-field-row"><span>운영 데이터 폴더</span><input bind:value={settings.runtime.data_dir} /></label>
            <label class="settings-field-row"><span>모델 캐시 폴더</span><input bind:value={settings.runtime.model_cache} /></label>
            </details>
            <div class="media-usage"><span>Docker · {runtime?.docker === 'online' ? '정상' : runtime?.docker === 'offline' ? '연결 실패' : '확인 중'}</span><span>시스템 가용 · {formatGiB(runtime?.memory?.available_gib)} GiB</span><span>즉시 여유 · {formatGiB(runtime?.memory?.free_gib)} GiB</span></div>
          {/if}
        </fieldset>
        </div>
        <div class="settings-section" hidden={systemSection !== 'sets'}>
          <RuntimeSetEditor bind:catalog={settings.runtime.catalog} initialSelection={settings.runtime.bundle} />
        </div>
        {#if systemSection === 'services'}<SupportServices bind:settings onstatus={rows => { serviceHealth = { ...serviceHealth, extra: Object.fromEntries(rows.map(row => [row.key, { status: row.health === 'online' ? 'ok' : row.health, enabled: row.enabled }])) }; }} />{/if}
        <div class="settings-section" hidden={systemSection !== 'downloads'}><ModelDownloads catalog={settings.runtime.catalog} /></div>
        <div class="settings-section" hidden={systemSection !== 'appearance'}><ThemeSettings bind:appearance={settings.appearance} /></div>
        <div class="settings-section" hidden={systemSection !== 'storage'}>
        <fieldset>
          <legend>앱 서버</legend>
          <label class="settings-field-row"><span>Listen address</span><input bind:value={settings.server.listen_addr} placeholder="0.0.0.0:8585" /></label>
          <label class="settings-field-row"><span>SQLite 파일</span><input bind:value={settings.server.database} placeholder="sparktalk.db" /></label>
        </fieldset>
        <fieldset>
          <legend>미디어 보관</legend>
          {#if mediaUsage}<div class="media-usage"><span>전체 {mediaUsage.files}개 · {formatBytes(mediaUsage.bytes)}</span><span>미사용 {mediaUsage.unused_files}개 · {formatBytes(mediaUsage.unused_bytes)}</span></div><button class="media-cleanup" onclick={removeUnusedMedia} disabled={cleaningMedia || !mediaUsage.unused_files}>{cleaningMedia ? '정리 중…' : '미사용 미디어 정리'}</button>{:else}<span class="media-loading">보관 현황을 불러오는 중…</span>{/if}
          <small>현재 대화에 첨부됐거나 전송 대기 중인 이미지·음성·비디오는 유지합니다.</small>
        </fieldset>
        </div>
        <p class="settings-help">변경 사항은 아래 저장 버튼으로 적용합니다. 앱 서버 주소와 DB 변경은 재시작이 필요합니다.</p>
      </div>
    </div>
    <div class="modal-actions"><button class="secondary" onclick={onclose}>닫기</button><button class="primary" onclick={persistSettings}>저장</button></div>
  </div>
</div>

<style>
  .context-fields { display: grid; gap: 6px; }
  .context-settings .context-row { display: grid; grid-template-columns: minmax(0, 1fr) 154px; align-items: center; gap: 10px; margin: 0; }
  .context-row > span:first-child { min-width: 0; }
  .context-row > span:first-child small { display: inline; margin-left: 5px; font-size: 11px; }
  .context-number { display: grid; grid-template-columns: minmax(0, 1fr) 26px; align-items: center; gap: 6px; }
  .context-number input { width: 100%; min-width: 0; box-sizing: border-box; text-align: right; padding: 7px 8px; }
  .context-number small { margin: 0; font-size: 11px; }
  @media (max-width: 520px) {
    .context-settings .context-row { grid-template-columns: minmax(0, 1fr) 128px; gap: 6px; }
    .context-row > span:first-child small { display: block; margin-left: 0; }
  }

  .settings-section-navigation { display: flex; gap: 4px; margin-bottom: 16px; padding: 4px; border: 1px solid #80808040; border-radius: 10px; }
  .settings-section-navigation button { flex: 1; padding: 9px 6px; border: 0; border-radius: 7px; background: transparent; color: inherit; font: inherit; font-size: 13px; }
  .settings-section-navigation button[aria-pressed=true] { background: #6584ed22; color: inherit; font-weight: 650; }
  .system-navigation { overflow-x: auto; scrollbar-width: thin; }
  .system-navigation button { flex: 1 0 auto; white-space: nowrap; padding: 9px 10px; }
  .settings-section-navigation button:focus-visible { outline: 2px solid #6584ed; }
  .settings-section[hidden] { display: none; }
  .system-advanced { border-top: 1px solid #80808040; padding-top: 10px; }
  .system-advanced summary { cursor: pointer; font-size: 13px; }
  .system-advanced label { margin-top: 12px; }
</style>
