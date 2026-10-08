<script>
  import SettingsHelp from './SettingsHelp.svelte';
  export let config;
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
