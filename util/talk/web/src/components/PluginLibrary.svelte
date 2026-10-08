<script>
  import { onMount, onDestroy } from 'svelte';
  import { listPlugins, pluginRuns, pluginAction, installPlugin } from '../api/plugins.js';
  export let onnotify = () => {};
  let packageFile = null, purgeData = false;
  let items = [], loading = true, busy = false, selected = null, config = '{}', grants = [], runs = [], input = '{}', sessionId = '', result = '', timer, alive = true;
  const labels = { storage: '전용 저장 공간', tools: '대화에 도구 제공', jobs: '백그라운드 작업', 'model.complete': '모델 호출', 'tool.call': 'Talk 도구 호출' };
  const status = { active: '활성', disabled: '비활성', failed: '오류', starting: '시작 중', stopping: '종료 중', running: '실행 중', completed: '완료', canceled: '취소', interrupted: '재시작으로 중단', timed_out: '시간 초과' };
  onMount(() => { load(); timer = setInterval(refreshRuns, 3000); });
  onDestroy(() => { alive = false; clearInterval(timer); });
  async function install() {
    if (!packageFile || busy) return;
    busy = true;
    try { await installPlugin(packageFile); await load(); onnotify('설치 완료. 설정과 권한을 확인한 후 활성화하세요.', 'success'); }
    catch (e) { onnotify(e.message, 'error'); }
    finally { busy = false; }
  }
  async function remove() {
    if (!confirm(purgeData ? '플러그인과 저장 데이터·실행 기록을 삭제할까요?' : '데이터를 보존하고 플러그인을 제거할까요?')) return;
    await act('remove', {purge:purgeData});
  }
  async function rollback() {
    if (!confirm('이전 버전과 업데이트 직전 데이터로 복구할까요? 업데이트 이후 변경한 플러그인 데이터는 사라집니다.')) return;
    await act('rollback');
  }
  async function load() {
    try { const data = await listPlugins(); if (alive) { items = data; if (selected) { selected = items.find(x => x.manifest.id === selected.manifest.id) || null; if (selected) { config = JSON.stringify(selected.settings.config, null, 2); grants = [...(selected.settings.grants || [])]; } } } }
    catch (e) { if (alive) onnotify(e.message, 'error'); } finally { if (alive) loading = false; }
  }
  async function refreshRuns() {
    if (!alive || !selected || busy) return;
    const id = selected.manifest.id;
    try { const [data, catalog] = await Promise.all([pluginRuns(id), listPlugins()]); if (alive && selected?.manifest.id === id) { runs = data; items = catalog; selected = items.find(x => x.manifest.id === id) || null; } }
    catch (e) { if (alive) onnotify(e.message, 'error'); }
  }
  async function select(item) { purgeData = false; selected = item; config = JSON.stringify(item.settings.config, null, 2); grants = [...(item.settings.grants || [])]; runs = []; result = ''; await refreshRuns(); }
  async function act(action, body = {}) {
    if (!selected || busy) return;
    busy = true;
    try { const value = await pluginAction(selected.manifest.id, action, body); if (action === 'call' || action === 'submit') result = JSON.stringify(value, null, 2); await load(); }
    catch (e) { onnotify(e.message, 'error'); }
    finally { busy = false; await refreshRuns(); }
  }
  async function save() {
    try { await act('configure', { config: JSON.parse(config), grants }); }
    catch (e) { onnotify(e.message, 'error'); }
  }
  async function invoke(name) {
    const op = selected.manifest.operations.find(o => o.name === name);
    try { await act(op.background ? 'submit' : 'call', { operation: name, request: { input: JSON.parse(input), session_id: sessionId } }); }
    catch (e) { onnotify(e.message, 'error'); }
  }
  $: editable = selected && ['disabled', 'failed'].includes(selected.status) && !busy;
</script>
<section class="plugin-library">
  <header><div><h3>플러그인</h3><p>설치된 기능의 설정, 권한과 실행 상태를 관리합니다.</p></div><button onclick={load} disabled={busy}>새로고침</button></header>
  <div class="actions"><label>플러그인 패키지 (.zip)<input type="file" accept=".zip,application/zip" onchange={e => packageFile = e.currentTarget.files?.[0] || null} disabled={busy} /></label><button onclick={install} disabled={busy || !packageFile}>설치·업데이트</button></div>
  <p>업데이트하려면 먼저 비활성화하세요. 설치 후 권한을 다시 허용해야 합니다.</p>
  {#if loading}<p>불러오는 중…</p>
  {:else if !items.length}<p class="empty">등록된 플러그인이 없습니다.</p>
  {:else}
    <div class="catalog">{#each items as item}<button class:chosen={selected?.manifest.id === item.manifest.id} onclick={() => select(item)} disabled={busy}><strong>{item.manifest.name}</strong> · {status[item.status] || item.status}<small>{item.manifest.description}</small></button>{/each}</div>
    {#if selected}
      <h4>{selected.manifest.name} <small>v{selected.manifest.version}</small></h4>
      {#if selected.error}<p role="alert">{selected.error}</p>{/if}
      <div class="actions">
        <button onclick={() => act('enable')} disabled={busy || !['disabled', 'failed'].includes(selected.status)}>활성화</button>
        <button onclick={() => act('disable')} disabled={busy || (selected.status === 'disabled' && !selected.settings.enabled)}>비활성화</button>
        {#if selected.settings.data_version !== selected.manifest.data_version}<button onclick={() => act('migrate')} disabled={!editable || selected.settings.enabled}>저장 데이터 업데이트</button>{/if}
      </div>
      {#if selected.external}
        <div class="actions">
          {#if selected.rollback_available}<button onclick={rollback} disabled={!editable || selected.settings.enabled}>이전 버전·데이터 복구</button>{/if}
          <label class="grant"><input type="checkbox" bind:checked={purgeData} disabled={!editable} />제거할 때 데이터도 삭제</label>
          <button onclick={remove} disabled={!editable || selected.settings.enabled}>제거</button>
        </div>
      {/if}
      <fieldset disabled={!editable}><legend>허용할 권한</legend>
        {#each selected.manifest.permissions || [] as permission}<label class="grant"><input type="checkbox" bind:group={grants} value={permission} />{labels[permission] || permission}</label>{/each}
        {#if !selected.manifest.permissions?.length}<p>요청 권한 없음</p>{/if}
      </fieldset>
      <label>설정 (JSON)<textarea rows="6" bind:value={config} disabled={!editable}></textarea></label>
      <button onclick={save} disabled={!editable}>설정·권한 저장</button>
      {#if selected.status === 'active' && selected.manifest.panels?.length}
        <label>작업 입력 (JSON)<textarea rows="4" bind:value={input} disabled={busy}></textarea></label>
        <label>연결할 대화 ID (선택)<input bind:value={sessionId} disabled={busy} /></label>
        {#each selected.manifest.panels as panel}<article><h4>{panel.title}</h4><p>{panel.description}</p><div class="actions">{#each panel.operations as op}<button onclick={() => invoke(op)} disabled={busy || selected.active_runs > 0}>{selected.manifest.operations.find(x => x.name === op)?.description || op}</button>{/each}</div></article>{/each}
      {/if}
      {#if result}<pre>{result}</pre>{/if}
      <h4>최근 실행</h4>
      {#if !runs.length}<p>실행 기록이 없습니다.</p>{/if}
      {#each runs as run}<article><strong>{run.operation}</strong> · {status[run.status] || run.status} <small>{new Date(run.created_at).toLocaleString()}</small>
        {#if run.status === 'running'}<button onclick={() => act('cancel', { run_id: run.id })} disabled={busy}>취소</button>{/if}
        {#if run.error}<p role="alert">{run.error}</p>{/if}
        {#if run.result !== undefined}<details><summary>결과</summary><pre>{JSON.stringify(run.result, null, 2)}</pre></details>{/if}
      </article>{/each}
    {/if}
  {/if}
</section>
<style>
  .plugin-library{padding:20px;max-width:1000px;margin:auto}header,.actions{display:flex;align-items:center;gap:10px;flex-wrap:wrap}header{justify-content:space-between}.catalog{display:grid;gap:8px;margin:16px 0}.catalog button{text-align:left}.catalog small{display:block;margin-top:4px}.chosen{outline:2px solid #688fee}label{display:grid;gap:6px;margin:12px 0}.grant{display:flex;align-items:center}fieldset{margin:16px 0;border:1px solid var(--border,#555);border-radius:8px}textarea,input{box-sizing:border-box;max-width:100%}textarea{width:100%;font-family:monospace}article{border:1px solid var(--border,#555);border-radius:8px;padding:12px;margin:12px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:360px;overflow:auto}small,p{opacity:.8}.empty{padding:30px 0}button{cursor:pointer;padding:7px 12px;border:1px solid var(--border,#555);border-radius:8px;background:transparent;color:inherit}button:disabled{opacity:.45;cursor:default}[role=alert]{color:#e78383}h3,h4{margin:8px 0}
</style>
