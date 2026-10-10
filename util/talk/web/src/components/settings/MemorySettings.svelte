<script>
  import SettingsHelp from './SettingsHelp.svelte';
  export let config;
  export let embedding = { enabled: false };
  let status = null;
  let checking = false;
  async function refreshIndex() {
    checking = true;
    try { const response = await fetch('/api/retrieval/status'); if (response.ok) status = await response.json(); }
    finally { checking = false; }
  }
</script>

<fieldset>
  <legend><span>기억 회수</span> <SettingsHelp title="기억 회수"><p>항상 참고 기억은 관련성 검색과 별도 한도를 사용합니다. 현재 대화와 실패한 응답은 관련 회수에서 제외합니다.</p><p>항목 수는 1~12개, 토큰 예산은 256~8,192 범위에서 설정합니다.</p></SettingsHelp></legend>
  <label class="check"><input type="checkbox" bind:checked={config.enabled} /> 저장된 기억을 대화에 적용</label>
  <label class="check"><input type="checkbox" bind:checked={config.recall_sessions} disabled={!config.enabled} /> 관련 있는 과거 대화도 검색</label>
  <label class="check"><input type="checkbox" bind:checked={config.allow_proposals} disabled={!config.enabled} /> 모델의 기억 제안 허용 · 저장 전 항상 확인</label>
  <div class="settings-number-list">
    <label class="settings-field-row settings-number-row"><span>항상 참고 최대 항목</span><input type="number" min="1" max="12" bind:value={config.always_max_results} /></label>
    <label class="settings-field-row settings-number-row"><span>항상 참고 토큰 예산</span><input type="number" min="256" max="8192" step="256" bind:value={config.always_token_budget} /></label>
  </div>
  <div class="settings-number-list">
    <label class="settings-field-row settings-number-row"><span>관련 회수 최대 항목</span><input type="number" min="1" max="12" bind:value={config.max_results} /></label>
    <label class="settings-field-row settings-number-row"><span>관련 회수 토큰 예산</span><input type="number" min="256" max="8192" step="256" bind:value={config.token_budget} /></label>
  </div>

</fieldset>

<fieldset>
  <legend><span>의미 검색</span> <SettingsHelp title="의미 검색"><p>키워드 검색과 의미 검색의 결과를 함께 사용합니다. 기억·과거 대화·문서에서 표현이 달라도 관련 내용을 찾는 데 도움이 됩니다.</p><p>EXL3·NVFP4 세트에서는 검색 모델을 상주시킵니다. QAD 등 검색 모델이 없는 세트에서는 키워드 검색을 사용합니다. 서비스가 응답하지 않아도 키워드 검색으로 동작합니다.</p></SettingsHelp></legend>
  <label class="check"><input type="checkbox" bind:checked={embedding.enabled} /> 키워드 검색에 의미 검색 추가</label>
  <button type="button" onclick={refreshIndex} disabled={checking}>색인 상태 확인</button>
  {#if status}
    {#if embedding.enabled && !status.enabled}<p>현재 세트에서는 키워드 검색을 사용합니다.</p>{/if}
    <p>기억 {status.counts.memory_ready}/{status.counts.memory_total} · 대화 {status.counts.message_ready}/{status.counts.message_total} · 문서 {status.counts.knowledge_ready}/{status.counts.knowledge_total}</p>
    {#if status.error}<small>의미 검색 대기: {status.error}</small>{/if}
  {/if}
</fieldset>
