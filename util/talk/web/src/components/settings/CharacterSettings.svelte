<script>
  import SettingsHelp from './SettingsHelp.svelte';
  import AvatarSettings from './AvatarSettings.svelte';
  import PromptPresetManager from './PromptPresetManager.svelte';
  import { ensureComposer } from '../../lib/prompt-composer.js';
  export let model;
  export let appearance;
  export let onnotify = () => {};
  export let onuploaded = () => {};
  $: c = ensureComposer(model);
  function set(field, value) { model = { ...model, prompt_composer: { ...c, [field]: value } }; }
</script>

<AvatarSettings role="assistant" bind:appearance {onnotify} {onuploaded} />
<fieldset>
  <legend><span>이름과 소개</span> <SettingsHelp title="이름과 소개"><p>이름과 외형은 모델을 바꿔도 유지됩니다. 조합 모드에서는 이름과 소개가 페르소나·공통 조건과 함께 적용됩니다.</p></SettingsHelp></legend>
  <label class="settings-field-row"><span>AI 이름</span><input value={c.character_name || ''} oninput={e => set('character_name', e.currentTarget.value)} maxlength="80" placeholder="SparkTalk" /></label>
  <label class="settings-field-row"><span>짧은 소개</span><input value={c.character_description || ''} oninput={e => set('character_description', e.currentTarget.value)} maxlength="1000" placeholder="예: 차분하고 호기심 많은 대화 상대" /></label>

</fieldset>
<PromptPresetManager bind:model {onnotify} />
