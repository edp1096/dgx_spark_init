<script>
  import SettingsHelp from './SettingsHelp.svelte';
  import { onMount, onDestroy } from 'svelte';
  export let settings;
  export let onstatus = () => {};
  let snapshot = null;
  let error = '';
  let loading = false;
  let acting = false;
  let timer;
  let disposed = false;
  let requestController;
  const enabledFields = { collector: 'collector_enabled', documents: 'documents_enabled', ssh: 'ssh_enabled' };
  const endpointFields = { media: 'media_endpoint', collector: 'collector_endpoint', documents: 'documents_endpoint', ssh: 'ssh_endpoint' };
  $: busy = acting || snapshot?.operation?.state === 'running';

  async function refresh() {
    if (loading || disposed) return;
    loading = true;
    requestController = new AbortController();
    try {
      const response = await fetch('/api/support', { signal: requestController.signal });
      if (!response.ok) throw new Error(await response.text());
      snapshot = await response.json();
      onstatus(snapshot.services || []);
    } catch (e) { if (e.name !== 'AbortError') error = e.message; }
    finally { loading = false; }
  }
  function enabled(key) { return key === 'media' ? settings.tools.media_import_enabled : settings.extra[enabledFields[key]]; }
  function setEnabled(key, value) {
    if (key === 'media') settings = { ...settings, tools: { ...settings.tools, media_import_enabled: value } };
    else settings = { ...settings, extra: { ...settings.extra, [enabledFields[key]]: value } };
  }
  function deployment(row) {
    if (!snapshot?.managed) return { endpoint: settings.extra[endpointFields[row.key]] || (row.key === 'media' ? settings.asr.ffmpeg_endpoint : ''), host: 'external' };
    const definition = settings.runtime.catalog?.components?.find(c => c.id === row.id) || row;
    const bundle = settings.runtime.catalog?.bundles?.find(b => b.id === snapshot.bundle_id);
    return { ...definition, ...(bundle?.components?.includes(row.id) ? bundle.bindings?.[row.id] : {}) };
  }
  function edit(row, field, value) {
    if (!snapshot.managed) {
      settings = { ...settings, extra: { ...settings.extra, [endpointFields[row.key]]: value } };
      if (row.key === 'media') settings.asr.ffmpeg_endpoint = value;
      return;
    }
    const catalog = structuredClone(settings.runtime.catalog);
    const bundle = catalog.bundles.find(b => b.id === snapshot.bundle_id);
    const target = bundle?.components?.includes(row.id)
      ? ((bundle.bindings ||= {})[row.id] ||= {})
      : catalog.components.find(c => c.id === row.id);
    target[field] = value;
    target.auto_address = false;
    if (field === 'endpoint') target.health_url = value.replace(/\/$/, '') + '/health';
    settings = { ...settings, runtime: { ...settings.runtime, catalog } };
  }
  function dirty(row) { const staged = deployment(row); return staged.host !== row.host || staged.endpoint !== row.endpoint || (snapshot?.managed && staged.port !== row.port); }
  function status(row) {
    if (row.health === 'online') return '준비 완료';
    if (row.health === 'failed') return '오류';
    if (row.status === 'running') return '실행 중 · 준비 확인 중';
    return row.status === 'external' ? '연결되지 않음' : '정지';
  }
  async function action(row, operation) {
    if (busy || dirty(row)) return;
    acting = true; error = '';
    try {
      const response = await fetch(`/api/runtime/components/${encodeURIComponent(row.id)}/${operation}`, { method: 'POST' });
      if (!response.ok) throw new Error(await response.text());
      const state = await response.json();
      if (snapshot) snapshot = { ...snapshot, operation: state.operation };
      await refresh();
    } catch (e) { error = e.message; }
    finally { acting = false; }
  }
  onMount(() => { refresh(); timer = setInterval(refresh, 5000); });
  onDestroy(() => { disposed = true; clearInterval(timer); requestController?.abort(); });
</script>

<section aria-label="지원 서비스 관리" class="support-services">
  <div>지원 서비스 <SettingsHelp title="지원 서비스"><p>기능 사용 설정과 서비스 실행은 별개입니다. 지원 서비스는 모델 세트와 독립적으로 시작·중지하며, 모델 전환이나 세트 중지 시 계속 실행됩니다.</p></SettingsHelp></div>
  <button type="button" onclick={refresh} disabled={loading}>서비스 상태 새로고침</button>
  {#if error}<p role="alert">{error}</p>{/if}
  {#if snapshot?.operation?.state === 'running'}
    <div role="status"><progress aria-label="서비스 작업 진행 중"></progress> {snapshot.operation.phase} · 처리 중</div>
  {:else if snapshot?.operation?.state === 'failed'}<p role="alert">{snapshot.operation.error}</p>{/if}
  {#each snapshot?.services || [] as row (row.id)}
    {@const staged = deployment(row)}
    {@const running = row.status === 'running'}
    <fieldset aria-label={`지원 서비스 ${row.key}`}>
      <legend>{row.name}</legend>
      <p>{row.description}</p>
      <label class="check"><input type="checkbox" checked={enabled(row.key)} onchange={event => setEnabled(row.key, event.currentTarget.checked)} /> 기능 사용 허용</label>
      <div class="service-state">
        <span>설치: {row.installed === 'ready' ? '이미지 준비됨' : row.installed === 'missing' ? '이미지 준비 필요' : row.installed === 'external' ? '외부 서비스' : '확인 불가'}</span>
        <strong>상태: {status(row)}</strong>
        <span>준비 버전: {row.version}</span>
        {#if row.running_image}<span>실행 이미지: {row.running_image}</span>{/if}
      </div>
      {#if snapshot.managed}<label class="settings-field-row"><span>실행 호스트</span><select value={staged.host} disabled={running || busy} onchange={event => edit(row, 'host', event.currentTarget.value)}>{#each Object.keys(settings.runtime.catalog?.hosts || {}) as host}<option value={host}>{host}</option>{/each}</select></label>{/if}
      {#if snapshot.managed}<label class="settings-field-row settings-number-row"><span>서버 포트</span><input type="number" min="1" max="65535" value={staged.port || ''} disabled={running || busy} onchange={event => edit(row, 'port', Number(event.currentTarget.value))} /></label>{/if}
      <label class="settings-field-row"><span>API 주소</span><input value={staged.endpoint || ''} disabled={snapshot.managed && (running || busy)} onchange={event => edit(row, 'endpoint', event.currentTarget.value)} /></label>
      {#if running && snapshot.managed}<small>실행 위치나 주소를 바꾸려면 먼저 서비스를 중지하세요.</small>{/if}
      {#if dirty(row)}<small>배치 변경을 저장한 뒤 서비스를 조작하세요.</small>{/if}
      {#if snapshot.managed}
        <div class="buttons">
          <button type="button" disabled={busy || dirty(row) || row.controller === 'external'} onclick={() => action(row, 'prepare')}>이미지 준비</button>
          <button type="button" disabled={busy || dirty(row) || running || row.controller === 'external'} onclick={() => action(row, 'start')}>시작</button>
          <button type="button" disabled={busy || dirty(row) || !running || row.controller === 'external'} onclick={() => action(row, 'stop')}>중지</button>
          <button type="button" disabled={busy || dirty(row) || !running || row.controller === 'external'} onclick={() => action(row, 'restart')}>재시작</button>
        </div>
      {/if}
      {#if row.error || row.installation_error}<small class="service-error">{row.error || row.installation_error}</small>{/if}
    </fieldset>
  {/each}
  <small>기능 사용 및 배치 변경은 아래 저장 버튼으로 적용합니다. 시작·중지·재시작·이미지 준비 버튼은 현재 저장된 구성에 즉시 적용됩니다.</small>
</section>

<style>
  .support-services { display: grid; gap: 12px; }
  .service-state { display: flex; flex-wrap: wrap; gap: 8px 18px; margin: 10px 0; font-size: .85rem; }
  fieldset { min-width: 0; }
  .service-error { overflow-wrap: anywhere; color: #ef8e8e; }
  progress { width: 90px; }
</style>
