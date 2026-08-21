<script>
  import { onMount } from 'svelte';
  import { createSSHHost, deleteSSHHost, deleteSSHKey, generateSSHKey, importSSHKey, listSSHHosts, listSSHKeys, testSSHHost, trustSSHHost, updateSSHHost } from '../../api.js';

  export let onnotify = () => {};

  let hosts = [];
  let keys = [];
  let editing = null;
  let busy = '';
  let keyBusy = '';
  let newKeyID = '';
  let keyFile = null;
  let keyFileInput;
  let observedKeys = {};
  let form = emptyForm();
  const privateKeyUploadAllowed = typeof window !== 'undefined' && (window.isSecureContext || ['localhost', '127.0.0.1', '::1'].includes(window.location.hostname));

  onMount(() => { loadHosts(); loadKeys(); });

  function emptyForm() {
    return { alias: '', name: '', hostname: '', port: 22, username: '', key_id: '', timeout_seconds: 60 };
  }

  async function loadHosts() {
    try { hosts = await listSSHHosts(); }
    catch (error) { onnotify(error.message, 'error'); }
  }

  async function loadKeys() {
    try { keys = await listSSHKeys(); }
    catch (error) { onnotify(error.message, 'error'); }
  }

  async function generateKey() {
    const keyID = newKeyID.trim();
    if (!keyID) return;
    keyBusy = 'generate';
    try {
      const key = await generateSSHKey(keyID);
      keys = [...keys, key].sort((a, b) => a.id.localeCompare(b.id));
      newKeyID = '';
      onnotify(`${key.id}: Ed25519 키를 생성했습니다. 공개키를 대상 서버에 등록하세요.`);
    } catch (error) { onnotify(error.message, 'error'); }
    finally { keyBusy = ''; }
  }

  async function importKey() {
    const keyID = newKeyID.trim();
    if (!keyID || !keyFile || !privateKeyUploadAllowed) return;
    keyBusy = 'import';
    try {
      const key = await importSSHKey(keyID, keyFile);
      keys = [...keys, key].sort((a, b) => a.id.localeCompare(b.id));
      newKeyID = '';
      keyFile = null;
      if (keyFileInput) keyFileInput.value = '';
      onnotify(`${key.id}: 개인키를 안전한 외부 키 폴더로 가져왔습니다.`);
    } catch (error) { onnotify(error.message, 'error'); }
    finally { keyBusy = ''; }
  }

  async function removeKey(key) {
    if (!confirm(`${key.id} 개인키를 삭제할까요? 복구할 수 없습니다.`)) return;
    keyBusy = key.id;
    try {
      await deleteSSHKey(key.id);
      keys = keys.filter((item) => item.id !== key.id);
      onnotify(`${key.id}: 개인키를 삭제했습니다.`);
    } catch (error) { onnotify(error.message, 'error'); }
    finally { keyBusy = ''; }
  }

  async function copyPublicKey(key) {
    try {
      if (navigator.clipboard && window.isSecureContext) await navigator.clipboard.writeText(key.public_key);
      else {
        const field = document.createElement('textarea');
        field.value = key.public_key;
        field.style.position = 'fixed';
        field.style.opacity = '0';
        document.body.appendChild(field);
        field.select();
        document.execCommand('copy');
        field.remove();
      }
      onnotify(`${key.id}: 공개키를 복사했습니다.`);
    } catch { onnotify('자동 복사가 차단되었습니다. 표시된 공개키를 직접 복사하세요.', 'error'); }
  }

  function edit(host) {
    editing = host.id;
    form = { ...host };
  }

  function add() {
    editing = 'new';
    form = emptyForm();
  }

  function cancel() {
    editing = null;
    form = emptyForm();
  }

  async function save() {
    if (!form.alias.trim() || !form.name.trim() || !form.hostname.trim() || !form.username.trim() || !form.key_id.trim()) return;
    busy = editing;
    try {
      if (editing === 'new') await createSSHHost(form);
      else await updateSSHHost(editing, form);
      await loadHosts();
      cancel();
      onnotify('SSH 서버 설정을 저장했습니다.');
    } catch (error) { onnotify(error.message, 'error'); }
    finally { busy = ''; }
  }

  async function remove(host) {
    if (!confirm(`${host.name} SSH 설정을 삭제할까요?`)) return;
    busy = host.id;
    try {
      await deleteSSHHost(host.id);
      hosts = hosts.filter((item) => item.id !== host.id);
      onnotify('SSH 서버 설정을 삭제했습니다.');
    } catch (error) { onnotify(error.message, 'error'); }
    finally { busy = ''; }
  }

  async function test(host) {
    busy = host.id;
    observedKeys = { ...observedKeys, [host.id]: null };
    try {
      await testSSHHost(host.id);
      onnotify(`${host.name}: SSH 연결이 정상입니다.`);
    } catch (error) {
      const hostKey = error.details?.host_key;
      if (hostKey?.fingerprint) {
        observedKeys = { ...observedKeys, [host.id]: hostKey };
        onnotify(`${host.name}: 처음 보는 호스트 키입니다. 지문을 확인한 뒤 신뢰를 선택하세요.`, 'error');
      } else onnotify(`${host.name}: ${error.message}`, 'error');
    } finally { busy = ''; }
  }

  async function trust(host) {
    const hostKey = observedKeys[host.id];
    if (!hostKey || !confirm(`${host.name}의 호스트 키 ${hostKey.fingerprint}를 신뢰할까요?`)) return;
    busy = host.id;
    try {
      await trustSSHHost(host.id, hostKey.public_key);
      observedKeys = { ...observedKeys, [host.id]: null };
      await test(host);
    } catch (error) { onnotify(`${host.name}: ${error.message}`, 'error'); }
    finally { busy = ''; }
  }
</script>

<div class="ssh-settings">
  <div class="ssh-settings-head"><span>인증 키</span><small>개인키는 DB가 아닌 Extra 외부 폴더에 저장</small></div>
  <div class="ssh-key-create">
    <label>새 키 ID<input bind:value={newKeyID} placeholder="dgx-main" /></label>
    <button type="button" onclick={generateKey} disabled={keyBusy || !newKeyID.trim()}>Ed25519 생성</button>
    <label class:disabled={!privateKeyUploadAllowed}>기존 개인키<input bind:this={keyFileInput} type="file" onchange={(event) => keyFile = event.currentTarget.files?.[0] || null} disabled={!privateKeyUploadAllowed || keyBusy} /></label>
    <button type="button" onclick={importKey} disabled={keyBusy || !newKeyID.trim() || !keyFile || !privateKeyUploadAllowed}>가져오기</button>
  </div>
  {#if !privateKeyUploadAllowed}<small class="ssh-security-note">현재 HTTP 원격 접속에서는 개인키 업로드를 차단합니다. Extra 내부에서 생성하는 ‘Ed25519 생성’을 사용하세요.</small>{/if}
  {#if keys.length}
    <div class="ssh-key-list">
      {#each keys as key}
        <div class="ssh-key-row">
          <div><strong>{key.id}</strong><span>{key.type} · {key.fingerprint}</span><code>{key.public_key}</code></div>
          <div><button type="button" onclick={() => copyPublicKey(key)}>공개키 복사</button><button type="button" class="danger" onclick={() => removeKey(key)} disabled={keyBusy}>삭제</button></div>
        </div>
      {/each}
    </div>
  {:else}<small class="ssh-empty">등록된 인증 키가 없습니다. 새 키를 UI에서 바로 생성할 수 있습니다.</small>{/if}
  <datalist id="ssh-key-ids">{#each keys as key}<option value={key.id}></option>{/each}</datalist>
  <div class="ssh-settings-head"><span>등록 서버</span><button type="button" onclick={add} disabled={editing !== null}>＋ 서버 추가</button></div>
  {#if hosts.length}
    <div class="ssh-host-list">
      {#each hosts as host}
        <div class="ssh-host-row">
          <div><strong>{host.name}</strong><span>{host.username}@{host.hostname}:{host.port}</span><small>{host.alias} · 키 {host.key_id} · {host.timeout_seconds}초</small></div>
          <div class="ssh-host-actions">
            <button type="button" onclick={() => test(host)} disabled={busy || editing !== null}>시험</button>
            <button type="button" onclick={() => edit(host)} disabled={busy || editing !== null}>수정</button>
            <button type="button" class="danger" onclick={() => remove(host)} disabled={busy || editing !== null}>삭제</button>
          </div>
          {#if observedKeys[host.id]}
            <div class="ssh-host-key"><span>SHA256 지문</span><code>{observedKeys[host.id].fingerprint}</code><button type="button" onclick={() => trust(host)} disabled={busy}>이 키 신뢰</button></div>
          {/if}
        </div>
      {/each}
    </div>
  {:else if editing !== 'new'}
    <small class="ssh-empty">등록된 SSH 서버가 없습니다.</small>
  {/if}
  {#if editing !== null}
    <div class="ssh-host-form">
      <label>표시 이름<input bind:value={form.name} placeholder="DGX Spark" /></label>
      <label>별칭<input bind:value={form.alias} placeholder="dgx-main" /></label>
      <label class="ssh-wide">호스트<input bind:value={form.hostname} placeholder="192.168.100.61" /></label>
      <label>포트<input type="number" min="1" max="65535" bind:value={form.port} /></label>
      <label>사용자<input bind:value={form.username} placeholder="edp1096" /></label>
      <label>키 ID<input bind:value={form.key_id} list="ssh-key-ids" placeholder="dgx-main" /></label>
      <label>제한시간(초)<input type="number" min="1" max="86400" bind:value={form.timeout_seconds} /></label>
      <div class="ssh-form-actions"><button type="button" onclick={cancel}>취소</button><button type="button" class="primary" onclick={save} disabled={busy}>저장</button></div>
    </div>
  {/if}
  <small>공개키는 대상 서버의 <code>~/.ssh/authorized_keys</code>에 등록합니다. 비밀번호와 개인키 본문은 대화·YAML·SQLite에 저장하지 않습니다.</small>
</div>
