<script>
  export let model;
  export let notice = '';

  let managerOpen = false;
  let filter = '';

  function selectPreset(name) {
    model.system_prompt_preset = name;
    if (!name) return;
    const preset = model.system_prompt_presets.find((item) => item.name === name);
    if (preset) model.system_prompt = preset.prompt;
  }

  function addPreset() {
    const name = window.prompt('새 시스템 프롬프트 프리셋 이름을 입력하세요.');
    if (name === null || !name.trim()) return;
    const trimmed = name.trim();
    if (model.system_prompt_presets.some((item) => item.name === trimmed)) {
      notice = '같은 이름의 시스템 프롬프트 프리셋이 있습니다.';
      return;
    }
    model.system_prompt_presets = [...model.system_prompt_presets, { name: trimmed, prompt: model.system_prompt || '' }];
    model.system_prompt_preset = trimmed;
    notice = `'${trimmed}' 프리셋을 추가했습니다. 설정 저장을 누르면 반영됩니다.`;
  }

  function savePreset() {
    const name = model.system_prompt_preset;
    if (!name) {
      addPreset();
      return;
    }
    model.system_prompt_presets = model.system_prompt_presets.map((item) =>
      item.name === name ? { ...item, prompt: model.system_prompt || '' } : item);
    notice = `'${name}' 프리셋의 내용을 갱신했습니다. 설정 저장을 누르면 반영됩니다.`;
  }

  function renamePreset() {
    const current = model.system_prompt_preset;
    if (!current) return;
    const name = window.prompt('프리셋 이름을 수정하세요.', current);
    if (name === null || !name.trim() || name.trim() === current) return;
    const trimmed = name.trim();
    if (model.system_prompt_presets.some((item) => item.name === trimmed)) {
      notice = '같은 이름의 시스템 프롬프트 프리셋이 있습니다.';
      return;
    }
    model.system_prompt_presets = model.system_prompt_presets.map((item) =>
      item.name === current ? { ...item, name: trimmed } : item);
    model.system_prompt_preset = trimmed;
    notice = `'${trimmed}'으로 이름을 변경했습니다. 설정 저장을 누르면 반영됩니다.`;
  }

  function movePresetTo(index, rawTarget) {
    const numericTarget = Number(rawTarget);
    if (!Number.isInteger(numericTarget)) return;
    const target = Math.max(0, Math.min(model.system_prompt_presets.length - 1, numericTarget));
    const reordered = [...model.system_prompt_presets];
    const [preset] = reordered.splice(index, 1);
    reordered.splice(target, 0, preset);
    model.system_prompt_presets = reordered;
    notice = `'${preset.name}' 프리셋을 ${target + 1}번째로 이동했습니다. 설정 저장을 누르면 반영됩니다.`;
  }

  function movePresetToPosition(index, value) {
    if (!String(value).trim()) return;
    movePresetTo(index, Number(value) - 1);
  }

  function filteredPresets() {
    const query = filter.trim().toLocaleLowerCase();
    if (!query) return model.system_prompt_presets;
    return model.system_prompt_presets.filter((preset) => preset.name.toLocaleLowerCase().includes(query));
  }

  function removePreset() {
    const name = model.system_prompt_preset;
    if (!name || !confirm(`'${name}' 시스템 프롬프트 프리셋을 삭제할까요? 현재 프롬프트 내용은 유지됩니다.`)) return;
    model.system_prompt_presets = model.system_prompt_presets.filter((item) => item.name !== name);
    model.system_prompt_preset = '';
    notice = `'${name}' 프리셋을 삭제했습니다. 설정 저장을 누르면 반영됩니다.`;
  }

  function presetDirty() {
    const name = model?.system_prompt_preset;
    if (!name) return false;
    const preset = model.system_prompt_presets.find((item) => item.name === name);
    return Boolean(preset) && preset.prompt !== (model.system_prompt || '');
  }
</script>

<fieldset class="prompt-presets">
  <legend>전역 시스템 프롬프트</legend>
  <div class="preset-select-row">
    <label>프리셋
      <select value={model.system_prompt_preset || ''} onchange={(event) => selectPreset(event.currentTarget.value)}>
        <option value="">직접 입력</option>
        {#each model.system_prompt_presets as preset}<option value={preset.name}>{preset.name}</option>{/each}
      </select>
    </label>
    <button class="preset-manage-button" onclick={() => managerOpen = !managerOpen}>순서 관리</button>
  </div>
  {#if managerOpen}
    <div class="preset-manager">
      <div class="preset-manager-head">
        <input bind:value={filter} placeholder="프리셋 이름 검색" aria-label="프리셋 이름 검색" />
        <small>{filteredPresets().length}/{model.system_prompt_presets.length}개</small>
      </div>
      <div class="preset-manager-list">
        {#each filteredPresets() as preset (preset.name)}
          {@const index = model.system_prompt_presets.findIndex((item) => item.name === preset.name)}
          <div class="preset-manager-row">
            <input class="preset-position" type="number" min="1" max={model.system_prompt_presets.length} value={index + 1} aria-label={`${preset.name} 순서`} onchange={(event) => movePresetToPosition(index, event.currentTarget.value)} />
            <button class="preset-manager-name" class:active={model.system_prompt_preset === preset.name} onclick={() => selectPreset(preset.name)}>{preset.name}</button>
            <button title="맨 위로" aria-label={`${preset.name} 맨 위로 이동`} onclick={() => movePresetTo(index, 0)} disabled={index === 0}><span class="edge-arrow to-top" aria-hidden="true">↑</span></button>
            <button title="한 칸 위로" aria-label={`${preset.name} 한 칸 위로 이동`} onclick={() => movePresetTo(index, index - 1)} disabled={index === 0}>↑</button>
            <button title="한 칸 아래로" aria-label={`${preset.name} 한 칸 아래로 이동`} onclick={() => movePresetTo(index, index + 1)} disabled={index === model.system_prompt_presets.length - 1}>↓</button>
            <button title="맨 아래로" aria-label={`${preset.name} 맨 아래로 이동`} onclick={() => movePresetTo(index, model.system_prompt_presets.length - 1)} disabled={index === model.system_prompt_presets.length - 1}><span class="edge-arrow to-bottom" aria-hidden="true">↓</span></button>
          </div>
        {:else}
          <small class="preset-empty">일치하는 프리셋이 없습니다.</small>
        {/each}
      </div>
      <small>순서 번호를 입력하면 해당 위치로 바로 이동합니다.</small>
    </div>
  {/if}
  <textarea class="system-prompt" bind:value={model.system_prompt} rows="6" placeholder="예: 모든 답변은 한국어 존댓말로 작성한다."></textarea>
  {#if presetDirty()}<small class="preset-dirty">선택한 프리셋에서 내용이 변경되었습니다.</small>{/if}
  <div class="preset-actions">
    <button onclick={addPreset}>＋ 새 프리셋</button>
    <button onclick={savePreset}>{model.system_prompt_preset ? '현재 내용 저장' : '현재 내용으로 만들기'}</button>
    <button onclick={renamePreset} disabled={!model.system_prompt_preset}>이름 변경</button>
    <button class="danger" onclick={removePreset} disabled={!model.system_prompt_preset}>삭제</button>
  </div>
</fieldset>
