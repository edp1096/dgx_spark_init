<script>
  export let config;
</script>

<fieldset>
  <legend>기억 회수</legend>
  <label class="check"><input type="checkbox" bind:checked={config.enabled} /> 저장된 기억을 대화에 적용</label>
  <label class="check"><input type="checkbox" bind:checked={config.recall_sessions} disabled={!config.enabled} /> 관련 있는 과거 대화도 검색</label>
  <label class="check"><input type="checkbox" bind:checked={config.allow_proposals} disabled={!config.enabled} /> 모델의 기억 제안 허용 · 저장 전 항상 확인</label>
  <div class="settings-form-row two">
    <label>항상 참고 최대 항목<input type="number" min="1" max="12" bind:value={config.always_max_results} /></label>
    <label>항상 참고 토큰 예산<input type="number" min="256" max="8192" step="256" bind:value={config.always_token_budget} /></label>
  </div>
  <div class="settings-form-row two">
    <label>관련 회수 최대 항목<input type="number" min="1" max="12" bind:value={config.max_results} /></label>
    <label>관련 회수 토큰 예산<input type="number" min="256" max="8192" step="256" bind:value={config.token_budget} /></label>
  </div>
  <small>항상 참고 기억은 관련성 검색과 별도 한도를 사용합니다. 현재 대화와 실패한 응답은 관련 회수에서 제외합니다.</small>
</fieldset>
