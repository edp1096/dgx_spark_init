<script>
  import { defaultAttachmentLimits } from '../../lib/attachments.js';
  export let limits;
  $: if (!limits) limits = { ...defaultAttachmentLimits };
  const kinds = [['image', '이미지'], ['audio', '음성'], ['video', '비디오'], ['document', '문서·ZIP·소스']];
  function override(kind, enabled) {
    const entries = { ...limits.type_limits_mb };
    if (enabled) entries[kind] = limits.max_file_mb;
    else delete entries[kind];
    limits = { ...limits, type_limits_mb: entries };
  }
</script>
<fieldset>
  <legend>첨부 파일 제한</legend>
  <label class="settings-field-row"><span>공통 파일 크기 (MiB)</span><input type="number" min="1" max="4096" bind:value={limits.max_file_mb} /></label>
  <label class="settings-field-row"><span>메시지당 파일 수</span><input type="number" min="1" max="20" bind:value={limits.max_files} /></label>
  <small>업로드·URL 가져오기·SSH 다운로드·생성 파일에 공통으로 적용합니다. 개별 설정을 켠 형식만 별도 크기를 사용합니다.</small>
  <details>
    <summary>형식별 크기 설정 (선택)</summary>
    {#each kinds as [kind, label]}
      <div class="type-limit">
        <label><input type="checkbox" checked={Object.hasOwn(limits.type_limits_mb || {}, kind)} onchange={(event) => override(kind, event.currentTarget.checked)} />{label} 개별 설정</label>
        {#if Object.hasOwn(limits.type_limits_mb || {}, kind)}
          <label>{label} 크기 (MiB)<input type="number" min="1" max="4096" bind:value={limits.type_limits_mb[kind]} /></label>
        {:else}<small>공통 {limits.max_file_mb}MiB 사용</small>{/if}
      </div>
    {/each}
  </details>
</fieldset>
<style>
  .type-limit {display:flex; align-items:center; flex-wrap:wrap; gap:12px; margin:12px 0}
  .type-limit label {display:flex; align-items:center; gap:6px}
  .type-limit input[type="number"] {width:100px}
  small {display:block; opacity:.75; margin:8px 0}
</style>
