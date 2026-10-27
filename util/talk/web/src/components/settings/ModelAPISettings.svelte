<script>
  import { onMount } from 'svelte';
  let info = null;
  let error = '';
  let copyStatus = '';
  let addressInput;
  const baseURL = `${window.location.origin}/v1`;
  onMount(() => {
    const controller = new AbortController();
    async function refresh() {
      try {
        const response = await fetch('/api/model-api', { signal: controller.signal, cache: 'no-store' });
        if (!response.ok) throw new Error('연결 모델 정보를 불러오지 못했습니다.');
        info = await response.json();
        error = '';
      } catch (e) { if (!controller.signal.aborted) error = '연결 모델 정보를 불러오지 못했습니다.'; }
    }
    refresh();
    const timer = setInterval(refresh, 5000);
    return () => { clearInterval(timer); controller.abort(); };
  });
  async function copyAddress() {
    try {
      if (navigator.clipboard?.writeText) await navigator.clipboard.writeText(baseURL);
      else {
        addressInput.focus(); addressInput.select();
        if (!document.execCommand('copy')) throw new Error('copy failed');
      }
      copyStatus = '주소를 복사했습니다.';
    } catch { addressInput.focus(); addressInput.select(); copyStatus = '선택된 주소를 복사하세요.'; }
  }
</script>

<fieldset>
  <legend>외부 툴 연결</legend>
  <label class="settings-field-row"><span>공통 API 주소</span><input aria-label="공통 API 주소" bind:this={addressInput} value={baseURL} readonly /></label>
  <button type="button" onclick={copyAddress}>API 주소 복사</button>
  <p role="status">{copyStatus}</p>
  <label class="settings-field-row"><span>고정 모델 ID</span><input aria-label="고정 모델 ID" value="sparktalk" readonly /></label>
  <p>현재 연결 대상: <strong>{info?.active_model || '확인 중'}</strong></p>
  {#if error}<p role="alert">{error}</p>{/if}
  <small>외부 툴에 위 주소와 모델 ID를 입력하세요. 모델을 전환해도 그대로 사용할 수 있습니다. API 키는 필요하지 않습니다. 입력이 필수인 툴에는 임의의 값을 입력하세요.</small>
  <small>Chat Completions와 스트리밍을 전달합니다. Responses API는 연결 모델 서버가 지원해야 합니다. Talk의 기억과 내장 도구는 실행하지 않습니다.</small>
</fieldset>

<style>
  fieldset { min-width: 0; }
  input { min-width: 0; max-width: 100%; }
  p { overflow-wrap: anywhere; }
  small { display: block; margin-top: 8px; }
</style>
