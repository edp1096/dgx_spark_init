<script>
 import { onMount } from 'svelte';
 import { getKeyStore, keyStoreAction } from '../../api.js';
 export let onnotify = () => {};
 export let onchange = () => {};
 let state = null;
 let selected = [];
 let target = '';
 let busy = false;
 let error = '';
 async function refresh() {
  try { state = await getKeyStore(); error = ''; }
  catch(e) { error = e.message; }
 }
 onMount(() => { refresh(); const timer = setInterval(() => { if (!busy) refresh(); }, 30000); return () => clearInterval(timer); });
 async function act(action) {
  busy = true;
  try {
   const report = await keyStoreAction({ action, hosts: [...new Set([...(state?.hosts || []), ...selected])], target });
   await refresh(); await onchange(state);
   const pending = report.error || report.replicas?.some(r => r.error);
   onnotify(pending ? '변경은 반영됐지만 일부 호스트의 동기화가 대기 중입니다.' : action === 'handoff' ? '키 관리 권한을 이전했습니다.' : '키 저장소 동기화를 완료했습니다.', pending ? 'error' : 'info');
  } catch(e) { error = e.message; onnotify(e.message, 'error'); }
  finally { busy = false; }
 }
</script>
<details class="key-sync">
 <summary>키 저장소 동기화 <span>{state?.hosts?.length ? `관리: ${state.report?.authority_host || '연결 확인 필요'}` : '설정 안 됨'}</span></summary>
 <small>세트와 별개로 키·신뢰한 서버 정보를 복제합니다. 등록·교체·삭제는 현재 관리 호스트에 반영됩니다.</small>
 {#if error}<p role="alert">{error}</p>{/if}
 {#if state?.hosts?.length}
  {#if state.report?.error}<p role="alert">{state.report.error}</p>{/if}
  <div class="replicas">
   {#each state.report?.replicas || [] as replica}
    <div><strong>{replica.host}{#if state.peers?.[replica.host]?.address}<small>{state.peers[replica.host].address}</small>{/if}</strong><span>{replica.error ? '동기화 대기' : `${replica.manifest.epoch}.${replica.manifest.version} · ${Object.keys(replica.manifest.keys || {}).length}개 키`}</span>{#if replica.error}<small>{replica.error}</small>{/if}</div>
   {/each}
  </div>
  <div class="sync-actions">
   <button type="button" disabled={busy} onclick={() => act('sync')}>지금 동기화</button>
   <label>관리 권한 이전 대상<select bind:value={target}><option value="">호스트 선택</option>{#each state.hosts as host}{#if host !== state.report?.authority_host}<option value={host}>{host}</option>{/if}{/each}</select></label>
   <button type="button" disabled={busy || !target} onclick={() => act('handoff')}>관리 권한 이전</button>
  </div>
  <details><summary>복제 호스트 추가</summary>
   <div class="sync-hosts">{#each state.available_hosts.filter(host => !state.hosts.includes(host)) as host}<label><input type="checkbox" value={host} bind:group={selected} />{host}</label>{/each}</div>
   <button type="button" disabled={busy || !selected.length} onclick={() => act('configure')}>호스트 추가 및 동기화</button>
  </details>
  <small>오프라인 호스트는 30초마다 다시 확인합니다. 관리 호스트를 끄기 전에 권한을 이전하세요.</small>
 {:else if state}
  <div class="sync-hosts">{#each state.available_hosts as host}<label><input type="checkbox" value={host} bind:group={selected} />{host}</label>{/each}</div>
  <button type="button" disabled={busy || !selected.length} onclick={() => act('configure')}>선택한 호스트에 동기화 설정</button>
  <small>각 호스트에 최신 Extra SSH 이미지와 SSH 연결이 필요합니다. 서로 다른 기존 저장소는 자동 병합하지 않습니다.</small>
 {:else}<small>저장소 정보를 불러오는 중…</small>{/if}
</details>
<style>
 button { padding: 7px 10px; border: 1px solid #7775; border-radius: 7px; background: #7771; color: inherit; font-size: 12px; } button:disabled { opacity: .45; cursor: default; }
 .key-sync { min-width: 0; margin: 10px 0; padding: 10px; border: 1px solid #7774; border-radius: 9px; }
 summary { cursor: pointer; font-size: 13px; } summary span { margin-left: 8px; font-size: 11px; opacity: .7; }
 small { display: block; margin-top: 8px; line-height: 1.5; overflow-wrap: anywhere; }
 .replicas { display: grid; gap: 6px; margin: 10px 0; } .replicas > div { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 6px; padding: 8px; background: #7771; border-radius: 6px; font-size: 12px; } .replicas small { width: 100%; }
 .sync-actions, .sync-hosts { display: flex; flex-wrap: wrap; align-items: end; gap: 8px; margin: 10px 0; } .sync-actions label { flex: 1; min-width: 140px; } .sync-hosts label { display: flex; align-items: center; gap: 5px; } p { overflow-wrap: anywhere; font-size: 12px; }
</style>
