<script>
  import SettingsHelp from './settings/SettingsHelp.svelte';
  export let compact = false;
  const origin = window.location.origin;
  const insecureHTTP = window.location.protocol === 'http:' && !window.isSecureContext;
  const isEdge = /Edg(?:A|iOS)?\//.test(navigator.userAgent);
  const browserName = isEdge ? 'Edge' : 'Chrome';
  const flagsAddress = `${isEdge ? 'edge' : 'chrome'}://flags/#unsafely-treat-insecure-origin-as-secure`;
  let copyStatus = '';
  async function copy(value, event) {
    const button = event.currentTarget;
    const field = button.parentElement.querySelector('input');
    let copied = false;
    try {
      if (navigator.clipboard?.writeText) { await navigator.clipboard.writeText(value); copied = true; }
    } catch { /* Use the selected field on HTTP or when clipboard permission is denied. */ }
    if (!copied) {
      field.focus(); field.select(); field.setSelectionRange(0, field.value.length);
      try { copied = document.execCommand('copy'); } catch { /* Keep text selected for manual copying. */ }
    }
    copyStatus = copied ? '복사했습니다. 브라우저 주소창 또는 설정 입력칸에 붙여넣으세요.' : '자동 복사가 차단되었습니다. 선택된 주소를 직접 복사하세요.';
    if (copied) button.focus();
  }
</script>

<SettingsHelp title="마이크 사용 설정" buttonText={compact ? '' : insecureHTTP ? 'HTTP 마이크 설정 방법' : '마이크 사용 방법'}>
  {#if insecureHTTP}
    <p>Chrome·Edge에서 현재 HTTP 주소를 안전한 출처 예외로 등록하면 마이크를 사용할 수 있습니다. 이 설정은 브라우저에서 직접 변경해야 합니다.</p>
    <div class="microphone-guide">
      <strong>1. 브라우저 설정 주소를 복사해 주소창에 붙여넣기</strong>
      <div class="copy-row"><label>{browserName}<input readonly value={flagsAddress} aria-label={`${browserName} 설정 주소`} onclick={event => event.currentTarget.select()} /></label><button type="button" onclick={event => copy(flagsAddress, event)} aria-label={`${browserName} 설정 주소 복사`}>복사</button></div>
      <small>웹페이지에서 chrome://·edge:// 설정 화면을 직접 열 수 없어 주소창에 붙여넣어야 합니다.</small>
      <strong>2. ‘Insecure origins treated as secure’ 입력칸에 아래 주소 추가</strong>
      <div class="copy-row"><label>등록할 현재 화면 주소<input readonly value={origin} aria-label="등록할 현재 화면 주소" onclick={event => event.currentTarget.select()} /></label><button type="button" onclick={event => copy(origin, event)} aria-label="현재 화면 주소 복사">복사</button></div>
      <small>http://와 포트 번호까지 포함합니다. 다른 주소가 이미 있으면 쉼표로 구분해 추가하세요.</small>
      <strong>3. Enabled 선택 → Relaunch / Restart로 브라우저 재시작</strong>
      <strong>4. SparkTalk를 다시 열고 사이트의 마이크 권한 허용</strong>
      <p class="copy-status" role="status">{copyStatus}</p>
      <small>직접 관리하는 주소에만 적용하세요. 이 설정이 HTTP 통신을 암호화하지는 않습니다.</small>
      <small>iPhone·iPad 또는 해당 플래그가 없는 브라우저에서는 HTTPS로 접속하세요.</small>
    </div>
  {:else}
    <p>현재 주소는 보안 환경으로 인식됩니다. 별도의 HTTP 예외 설정은 필요하지 않습니다.</p>
    <p>마이크가 동작하지 않으면 주소창의 사이트 권한과 운영체제의 마이크 권한을 허용하고, 연결된 마이크를 확인하세요. 브라우저가 녹음 기능을 지원해야 합니다.</p>
  {/if}
</SettingsHelp>

<style>
  .microphone-guide { display: grid; gap: 12px; margin-top: 14px; }
  .microphone-guide strong { font-size: 12px; }
  .copy-row { display: grid; grid-template-columns: minmax(0, 1fr) auto; align-items: end; gap: 7px; min-width: 0; }
  .copy-row label { display: grid; gap: 5px; min-width: 0; font-size: 11px; }
  .copy-row input { width: 100%; min-width: 0; height: 34px; padding: 0 8px; border: 1px solid #80808050; border-radius: 7px; background: #80808012; color: inherit; font: inherit; }
  .copy-row button { height: 34px; padding: 0 10px; border: 1px solid #80808050; border-radius: 7px; background: #80808020; color: inherit; font: inherit; font-size: 11px; }
  .copy-row button:hover { background: #80808035; }
  .copy-row button:focus-visible, .copy-row input:focus-visible { outline: 2px solid #6584ed; outline-offset: 2px; }
  small { font-size: 11px; opacity: .8; line-height: 1.6; }
  .copy-status { margin: 0; font-size: 12px; }
  .copy-status:empty { display: none; }
</style>
