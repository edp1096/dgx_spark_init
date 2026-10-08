<script>
 import { onMount } from 'svelte';
 let status = {}, token = '', error = '', busy = false;
 let showToken = false, copyStatus = '', tokenInput;
 const address = window.location.origin;
 async function refresh() {
  try { const r = await fetch('/api/browser'); if (!r.ok) throw new Error(await r.text()); status = await r.json(); } catch (e) { error = e.message; }
 }
 async function change(method) {
  busy = true; error = ''; copyStatus = ''; showToken = false;
  try { const r = await fetch('/api/browser', { method }); if (!r.ok) throw new Error(await r.text()); const data = await r.json(); token = data.token || ''; await refresh(); } catch (e) { error = e.message; } finally { busy = false; }
 }
 async function copyToken() {
  copyStatus = '';
  try {
   if (navigator.clipboard?.writeText) {
    try { await navigator.clipboard.writeText(token); copyStatus = '복사됨'; return; } catch { /* HTTP or denied clipboard: use selection fallback. */ }
   }
   const field = document.createElement('textarea');
   field.value = token;
   field.style.cssText = 'position:fixed;left:-9999px;top:0';
   document.body.appendChild(field);
   try { field.select(); if (!document.execCommand('copy')) throw new Error('copy failed'); }
   finally { field.remove(); }
   copyStatus = '복사됨';
  } catch {
   showToken = true;
   copyStatus = '표시된 키를 선택해 직접 복사하세요.';
   requestAnimationFrame(() => { tokenInput?.focus(); tokenInput?.select(); });
  }
 }
 onMount(() => { refresh(); const t = setInterval(refresh, 5000); return () => clearInterval(t); });
</script>
<fieldset>
 <legend>브라우저 연결</legend>
 <p>로그인된 Chrome의 네이버 구매상품을 조회하고, 확인한 리뷰를 등록합니다.</p>
 <div class="media-usage">{status.connected ? '● 연결됨' : '연결 안 됨'}</div>
 {#if status.update_required}<p role="status">확장 업데이트가 필요합니다. 아래 ZIP을 기존 확장 폴더에 덮어쓴 뒤 Chrome 확장 관리에서 새로고침하세요. 연결 키는 그대로 사용합니다.</p>{/if}
 <p><a class="extension-download" href="/api/browser/extension.zip" download="sparktalk-browser.zip">Chrome 확장 다운로드</a></p>
 <small>ZIP 압축을 풀고 Chrome 확장 프로그램에서 개발자 모드 → 압축해제된 확장 프로그램 로드로 설치하세요.</small>
 <label class="settings-field-row"><span>Talk 주소</span><input readonly value={address} /></label>
 <div class="browser-actions"><button type="button" disabled={busy} onclick={() => change('POST')}>{status.paired ? '연결 키 재발급' : '연결 키 발급'}</button><button type="button" disabled={busy || !status.paired} onclick={() => change('DELETE')}>연결 해제</button></div>
 {#if token}
  <label class="settings-field-row"><span>새 연결 키</span><input bind:this={tokenInput} readonly type={showToken ? 'text' : 'password'} value={token} onclick={e => e.currentTarget.select()} /></label>
  <div class="browser-actions">
   <button type="button" onclick={copyToken}>연결 키 복사</button>
   <button type="button" aria-pressed={showToken} onclick={() => { showToken = !showToken; }}>{showToken ? '키 숨기기' : '키 표시'}</button>
   <small role="status">{copyStatus}</small>
  </div>
  <small>복사한 키를 확장에 입력하세요. 재발급하면 기존 연결은 해제됩니다.</small>
 {/if}
 <p><small>Chrome에서 네이버 구매내역을 열고 Talk에 “리뷰 쓸 상품 찾아줘”라고 요청하세요. 실제 사용 소감과 별점으로 초안을 만든 뒤, 대화의 “이대로 등록”으로 승인합니다.</small></p>
 {#if error}<p role="alert">{error}</p>{/if}
</fieldset>
<style>
 .extension-download { display:inline-block; padding:8px 12px; border:1px solid #80808040; border-radius:7px; background:#6584ed12; color:inherit; text-decoration:none; font-size:13px; }
 .extension-download:hover { background:#6584ed22; }
 .browser-actions { display:flex; flex-wrap:wrap; gap:.5rem; margin:.75rem 0; }
 input { min-width:0; }
 p,small { overflow-wrap:anywhere; }
</style>
