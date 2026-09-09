<script>
  import SettingsField from './SettingsField.svelte';
  import { request } from '../../api/request.js';
  import { resolveSetMembers, setDeploymentValue, resetDeployment, hostIsUsed } from '../../lib/runtime-sets.js';
  export let catalog;
  export let initialSelection = '';
  let selected = initialSelection;
  let choosingServices = false;
  let selectedHost = 'local';
  let discovering = false;
  let networkResult = null;
  let chosenWorker = '';
  function networkEnabled(enabled) {
    catalog.network = { ...catalog.network, enabled };
    catalog = catalog;
  }
  async function discoverNetwork() {
    discovering = true;
    try {
      networkResult = await request('/api/runtime/network/discover', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({catalog, worker_id:chosenWorker}) });
      if (networkResult.catalog) {
        catalog = networkResult.catalog;
        chosenWorker = catalog.network.worker_id;
        message = '탐색한 장비와 자동 주소를 반영했습니다. 저장하면 적용됩니다.';
      } else { message = networkResult.error || '워커를 선택하세요.'; }
    } catch (error) { message=error.message; }
    finally { discovering=false; }
  }
  $: if (!catalog?.hosts?.[selectedHost]) selectedHost = catalog?.hosts?.local ? 'local' : Object.keys(catalog?.hosts || {})[0] || '';
  let probeResults = {};
  let resultBundle = selected;
  $: if (resultBundle !== selected) { resultBundle = selected; probeResults = {}; }
  let probingID = '';
  $: members = resolveSetMembers(catalog, bundle);
  $: displayContext = bundle?.context_tokens;
  let message = '';
  let source = '';
  let busy = false;
  $: if (catalog && !catalog.bundles?.some(b => b.id === selected)) selected = catalog.bundles?.some(b => b.id === initialSelection) ? initialSelection : catalog.bundles?.[0]?.id || '';
  $: bundle = catalog?.bundles?.find(b => b.id === selected);
  const roles = ['llm', 'asr', 'tts', 'image', 'media', 'collector', 'ssh'];

  function updateDeployment(id, field, value) {
    setDeploymentValue(bundle, id, field, value);
    catalog = catalog;
    probeResults = {};
  }
  function updateDefinition(id, field, value) {
    const definition = catalog.components.find(c => c.id === id);
    definition[field] = value;
    catalog = catalog;
  }

  function uniqueID(base, list) {
    let id = base; let n = 2;
    while (list.some(item => item.id === id)) id = `${base}-${n++}`;
    return id;
  }
  function duplicateBundle() {
    const copy = structuredClone(bundle);
    copy.id = uniqueID(`${copy.id}-copy`, catalog.bundles);
    copy.name += ' 복사';
    catalog.bundles = [...catalog.bundles, copy];
    selected = copy.id;
    catalog = catalog;
  }
  function toggle(id, checked) {
    bundle.components = checked ? [...bundle.components, id] : bundle.components.filter(value => value !== id);
    if (!checked) resetDeployment(bundle, id);
    catalog = catalog;
  }
  function addComponent() {
    const id = uniqueID('service', catalog.components);
    catalog.components = [...catalog.components, { id, name: '새 서비스', role: 'collector', controller: 'external', host: Object.keys(catalog.hosts)[0], endpoint: 'http://127.0.0.1:8695', health_url: 'http://127.0.0.1:8695/health', memory_gib: 0, startup_timeout_seconds: 120 }];
    catalog = catalog;
    choosingServices = true;
  }
  function duplicateComponent(component) {
    const copy = structuredClone(component);
    copy.id = uniqueID(`${copy.id}-copy`, catalog.components);
    copy.name += ' 복사';
    if (copy.container) copy.container += '-copy';
    catalog.components = [...catalog.components, copy];
    catalog = catalog;
    choosingServices = true;
    message = '새 서비스 정의를 만들었습니다. 실행 위치만 바꾸려면 기존 서비스의 세트별 배치를 수정하면 됩니다.';
  }
  function addHost() {
    const id = uniqueID('host', Object.keys(catalog.hosts).map(id => ({ id })));
    catalog.hosts = { ...catalog.hosts, [id]: { address: '', user: '', port: 22, data_dir: '', memory_reserve_gib: 8 } };
    selectedHost = id;
    catalog = catalog;
  }
  async function probe(component) {
    probingID = component.id;
    const targetBundle = selected;
    const stillCurrent = () => selected === targetBundle && members.some(c => c.id === component.id && c.health_url === component.health_url && c.endpoint === component.endpoint);
    try {
      const result = await request('/api/runtime/probe', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ url: component.health_url }) });
      if (stillCurrent()) probeResults = { ...probeResults, [component.id]: result.error || result.status };
    } catch (error) { if (stillCurrent()) probeResults = { ...probeResults, [component.id]: error.message }; }
    finally { probingID = ''; }
  }
  async function importSource() {
    busy = true;
    try {
      catalog = await request('/api/runtime/catalog/parse', { method: 'POST', headers: { 'Content-Type': 'text/plain' }, body: source });
      message = '검증한 구성을 편집 화면에 불러왔습니다. 아래 저장 버튼으로 적용하세요.';
    } catch (error) { message = error.message; }
    finally { busy = false; }
  }
  async function importFile(event) {
    const file = event.currentTarget.files?.[0];
    if (!file) return;
    if (file.size > 1024 * 1024) { message = '파일은 1MiB 이하여야 합니다.'; return; }
    source = await file.text();
    await importSource();
    event.target.value = '';
  }
  function download() {
    const url = URL.createObjectURL(new Blob([JSON.stringify(catalog, null, 2)], { type: 'application/json' }));
    const link = document.createElement('a'); link.href = url; link.download = 'sparktalk-sets.json'; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
</script>

{#if catalog}
<fieldset class="set-editor">
  <legend>AI 세트 편집</legend>
  <label class="settings-field-row"><span>편집할 세트</span><select bind:value={selected} onchange={() => { choosingServices = false; }}>{#each catalog.bundles as item}<option value={item.id}>{item.name}</option>{/each}</select></label>
  {#if bundle}
    <div class="set-heading"><span>{members.length}개 서비스 · {displayContext ? Math.round(displayContext / 1024) + 'K 문맥' : '문맥 자동'}</span><button type="button" onclick={duplicateBundle}>세트 복제</button></div>
    <label class="settings-field-row"><span>세트 이름</span><input bind:value={bundle.name} /></label>
    <details class="set-profile">
      <summary>모델·세트 상세 설정</summary>
    <label class="settings-field-row"><span>세트 ID</span><input value={bundle.id} onchange={(event) => { bundle.id = event.currentTarget.value; selected = bundle.id; catalog = catalog; }} /></label>
    <small>지원 서비스의 세트별 설정은 실행 위치·주소 선택에 사용됩니다. 지원 서비스는 모델 세트와 독립적으로 시작·중지합니다.</small>
    <label class="settings-field-row"><span>설명</span><input bind:value={bundle.description} /></label>
    <div class="grid">
      <label class="settings-field-row"><span>모델 ID</span><input bind:value={bundle.model_id} /></label>
      <label class="settings-field-row"><span>모델 유형</span><select bind:value={bundle.model_type}><option value="glm5.3">GLM 5.3</option><option value="qwen3.8">Qwen3.8</option><option value="gemma4">Gemma 4</option><option value="deepseek-v4">DeepSeek V4</option><option value="generic">일반 OpenAI 호환</option></select></label>
      <label class="settings-field-row settings-number-row"><span>문맥 토큰 수</span><input type="number" min="0" bind:value={bundle.context_tokens} /></label>
    </div>

      <button type="button" class="danger" disabled={catalog.bundles.length < 2} onclick={() => { catalog.bundles = catalog.bundles.filter(b => b.id !== selected); catalog = catalog; }}>세트 삭제</button>
    </details>
    <div class="set-heading"><strong>서비스 구성</strong><button type="button" aria-expanded={choosingServices} onclick={() => { choosingServices = !choosingServices; }}>{choosingServices ? '선택 마치기' : '서비스 선택'}</button></div>
    {#if choosingServices}
      <div class="service-picker">
        <small>이 세트에서 사용할 서비스를 선택하세요. 같은 역할은 하나만 선택합니다.</small>
        {#each catalog.components as component}
          <div class="picker-row"><label class="check"><input type="checkbox" checked={bundle.components.includes(component.id)} onchange={(event) => toggle(component.id, event.currentTarget.checked)} /><span>{component.name}<small>{component.role === 'tool' ? 'Extra' : component.role}</small></span></label>
          {#if !catalog.bundles.some(b => b.components.includes(component.id))}<button type="button" aria-label={component.name + ' 미사용 서비스 삭제'} onclick={() => { catalog.components = catalog.components.filter(c => c.id !== component.id); catalog = catalog; }}>삭제</button>{/if}</div>
        {/each}
        <button type="button" onclick={addComponent}>새 서비스 정의</button>
      </div>
    {/if}
    {#each members as component (component.id)}
      <details class="service-card">
        <summary><span class="service-summary"><strong>{component.name}</strong><small>{component.endpoint}</small></span><span class="host-badge">{component.host === 'local' ? '로컬' : component.host}</span></summary>
        <div class="service-connection">
          {#if component.controller !== 'external'}
            <label class="check"><input type="checkbox" checked={component.auto_address ?? false} onchange={(event) => updateDeployment(component.id, 'auto_address', event.currentTarget.checked)} /> 헤드·워커 주소 자동 연동</label>
          {/if}
          <label class="settings-field-row"><span>실행 호스트</span><select value={component.host ?? ""} oninput={(event) => updateDeployment(component.id, "host", event.currentTarget.value)}>{#each Object.keys(catalog.hosts) as host}<option value={host}>{host === 'local' ? '로컬 · 이 컴퓨터' : host}</option>{/each}</select></label>
          <label class="settings-field-row"><span>API 주소</span><input value={component.endpoint ?? ""} oninput={(event) => updateDeployment(component.id, "endpoint", event.currentTarget.value)} placeholder="http://서버:포트" /></label>
          <label class="settings-field-row"><span>상태 확인 URL</span><input value={component.health_url ?? ""} oninput={(event) => updateDeployment(component.id, "health_url", event.currentTarget.value)} /></label>
          <button type="button" disabled={!!probingID} onclick={() => probe(component)}>{probingID === component.id ? '확인 중…' : 'API 연결 시험'}</button>
          {#if probeResults[component.id]}<p class="probe-result" role="status">{probeResults[component.id]}</p>{/if}
          <small>호스트와 API 주소는 이 세트에만 적용됩니다.</small>
          {#if Object.keys(bundle.bindings?.[component.id] || {}).length}<button type="button" onclick={() => { resetDeployment(bundle, component.id); probeResults = {}; catalog = catalog; }}>기본 배치로 되돌리기</button>{/if}
        </div>
        <details class="service-advanced">
          <summary>실행·포트 상세 설정</summary>
          <small>공통 역할·Compose 레시피는 같은 정의를 쓰는 모든 세트에 적용됩니다. 나머지 배치는 이 세트에 저장합니다.</small>
          <div class="grid">
          <label class="settings-field-row"><span>서비스 ID</span><input value={component.id} readonly /></label>
          <label class="settings-field-row"><span>이름</span><input value={component.name ?? ""} oninput={(event) => updateDeployment(component.id, "name", event.currentTarget.value)} /></label>
          <label class="settings-field-row"><span>공통 역할</span><select value={component.role} onchange={(event) => updateDefinition(component.id, "role", event.currentTarget.value)}>{#if component.role === 'tool'}<option value="tool">Extra (레시피에서 역할 결정)</option>{/if}{#each roles as role}<option value={role}>{role}</option>{/each}</select></label>
          <label class="settings-field-row"><span>제어 방식</span><select value={component.controller ?? ""} oninput={(event) => updateDeployment(component.id, "controller", event.currentTarget.value)}><option value="compose">Docker Compose</option><option value="glm53-cluster">GLM Head + Worker</option><option value="dspark-cluster">DSpark Head + Worker</option><option value="external">연결 전용</option></select></label>
          <label class="settings-field-row"><span>서비스 모델 ID</span><input value={component.model ?? ""} oninput={(event) => updateDeployment(component.id, "model", event.currentTarget.value)} /></label>
          {#if component.controller !== 'external'}
            <label class="settings-field-row"><span>컨테이너 이름</span><input value={component.container ?? ""} oninput={(event) => updateDeployment(component.id, "container", event.currentTarget.value)} /></label>
            <label class="settings-field-row settings-number-row"><span>예상 메모리 GiB</span><input type="number" min="0" step="0.1" value={component.memory_gib ?? ""} oninput={(event) => updateDeployment(component.id, "memory_gib", Number(event.currentTarget.value))} /></label>
            <label class="settings-field-row settings-number-row"><span>시작 제한시간(초)</span><input type="number" min="1" value={component.startup_timeout_seconds ?? ""} oninput={(event) => updateDeployment(component.id, "startup_timeout_seconds", Number(event.currentTarget.value))} /></label>
          {/if}
          {#if component.controller === 'compose'}
            <label class="settings-field-row"><span>공통 Compose 레시피</span><input value={component.compose_asset} onchange={(event) => updateDefinition(component.id, "compose_asset", event.currentTarget.value)} placeholder="compose.extra-collector.yaml" /></label>
            <label class="settings-field-row"><span>서버 바인딩 주소</span><input value={component.bind_address ?? ""} oninput={(event) => updateDeployment(component.id, "bind_address", event.currentTarget.value)} placeholder="127.0.0.1" /></label>
            <label class="settings-field-row settings-number-row"><span>서버 공개 포트</span><input type="number" min="0" max="65535" value={component.port ?? ""} oninput={(event) => updateDeployment(component.id, "port", Number(event.currentTarget.value))} placeholder="0: 레시피 기본값" /></label>
            {#if component.compose_asset === 'compose.flash-next.yaml'}
              <SettingsField title="초안 어휘"><select value={component.runtime_options?.DRAFT_VOCAB ?? 'ko64k'} oninput={(event) => updateDeployment(component.id, "runtime_options", {...component.runtime_options, DRAFT_VOCAB:event.currentTarget.value})}><option value="ko64k">한국어 포함 64K (기본)</option><option value="off">전체 어휘</option></select><svelte:fragment slot="help"><p>초안 생성 속도를 높이는 설정입니다. 변경 후 서비스를 다시 시작해야 합니다.</p></svelte:fragment></SettingsField>
            {/if}
          {:else if ['glm53-cluster', 'dspark-cluster'].includes(component.controller)}
            <label class="settings-field-row settings-number-row"><span>API 공개 포트</span><input type="number" min="0" max="65535" value={component.port ?? ""} oninput={(event) => updateDeployment(component.id, "port", Number(event.currentTarget.value))} placeholder="0: 기본 포트" /></label>
            <label class="settings-field-row"><span>가중치</span><select value={component.runtime_options?.MODEL_VARIANT ?? 'official'} oninput={(event) => updateDeployment(component.id, "runtime_options", {...component.runtime_options, MODEL_VARIANT:event.currentTarget.value})}><option value="official">공식 원본</option><option value="abliterated">Abliterated</option></select></label>
            {#if component.controller === 'dspark-cluster'}
              <SettingsField title="도구 호출 형식 복구"><select value={component.runtime_options?.DSPARK_ENABLE_DSML_RECOVERY ?? '1'} oninput={(event) => updateDeployment(component.id, "runtime_options", {...component.runtime_options, DSPARK_ENABLE_DSML_RECOVERY:event.currentTarget.value})}><option value="1">사용 (기본)</option><option value="0">사용 안 함</option></select><svelte:fragment slot="help"><p>도구 호출 표기가 일부 누락된 응답을 복구합니다. 변경 후 서비스를 다시 시작해야 합니다.</p></svelte:fragment></SettingsField>
              <SettingsField title="반복 요청 캐시 보완"><select value={component.runtime_options?.DSPARK_ENABLE_DSPARK_SWA_PREFIX ?? '0'} oninput={(event) => updateDeployment(component.id, "runtime_options", {...component.runtime_options, DSPARK_ENABLE_DSPARK_SWA_PREFIX:event.currentTarget.value})}><option value="0">사용 안 함 (기본)</option><option value="1">사용</option></select><svelte:fragment slot="help"><p>반복 질문에서 응답이 끊길 때 사용합니다. 짧은 요청의 지연이 늘 수 있습니다.</p></svelte:fragment></SettingsField>
            {/if}
            {#each [['HEAD_RAIL_IP','헤드 통신 IP','10.200.0.1'],['WORKER_RAIL_IP','워커 통신 IP','10.200.0.2'],['HEAD_NCCL_IF','헤드 통신 인터페이스','enp1s0f1np1'],['WORKER_NCCL_IF','워커 통신 인터페이스','enp1s0f1np1'],['HEAD_NCCL_HCA','헤드 HCA','rocep1s0f1'],['WORKER_NCCL_HCA','워커 HCA','rocep1s0f1']] as [key,label,fallback]}
              <label class="settings-field-row"><span>{label}</span><input value={component.runtime_options?.[key] ?? fallback} oninput={(event) => updateDeployment(component.id, "runtime_options", {...component.runtime_options,[key]:event.currentTarget.value})} /></label>
            {/each}
            <label class="settings-field-row"><span>워커 호스트</span><select value={component.worker_host ?? ""} oninput={(event) => updateDeployment(component.id, "worker_host", event.currentTarget.value)}>{#each Object.keys(catalog.hosts) as host}<option value={host}>{host}</option>{/each}</select></label>
            <label class="settings-field-row"><span>워커 컨테이너</span><input value={component.worker_container ?? ""} oninput={(event) => updateDeployment(component.id, "worker_container", event.currentTarget.value)} /></label>
            <label class="settings-field-row settings-number-row"><span>워커 예상 메모리 GiB</span><input type="number" min="0" step="0.1" value={component.worker_memory_gib ?? ""} oninput={(event) => updateDeployment(component.id, "worker_memory_gib", Number(event.currentTarget.value))} /></label>
            <small>앱에 내장된 실행 패키지를 사용합니다. 통신망 설정은 실제 연결과 일치해야 합니다.</small>
          {/if}

          </div>
          <div class="buttons"><button type="button" onclick={() => duplicateComponent(component)}>서비스 정의 복제</button><button type="button" onclick={() => toggle(component.id, false)}>이 세트에서 제외</button></div>
        </details>
      </details>
    {/each}
    {#if !members.length}<p>서비스 선택을 눌러 사용할 모델과 부가 기능을 추가하세요.</p>{/if}
  {/if}
  <details class="host-editor">
    <summary>실행 호스트 편집</summary>
    <label class="check"><input type="checkbox" checked={catalog.network?.enabled ?? false} onchange={(event) => networkEnabled(event.currentTarget.checked)} /> 이 컴퓨터를 헤드로 사용하고 워커 자동 탐색</label>
    {#if catalog.network?.enabled}
      <small>앱 시작과 세트 기동 시 장비 ID·SSH·QSFP를 확인합니다. 주소 자동 연동을 켠 서비스에 적용되며 외부 서비스는 유지됩니다.</small>
      <div class="network-actions">
        <button type="button" disabled={discovering} onclick={discoverNetwork}>{discovering ? '장비 확인 중…' : '워커 탐색 · 주소 맞추기'}</button>
        {#if networkResult?.candidates?.length}
          <label class="settings-field-row"><span>워커 장비</span><select aria-label="워커 장비" bind:value={chosenWorker}><option value="">자동 선택</option>{#each networkResult.candidates as node}<option value={node.id}>{node.hostname} · {node.address}</option>{/each}</select></label>
        {/if}
      </div>
      {#if networkResult?.local}<small>현재 헤드: {networkResult.local.hostname} · {networkResult.local.address || '이 컴퓨터'}</small>{/if}
      {#if catalog.network.last_error}<p role="status">{catalog.network.last_error}</p>{/if}
      {#each networkResult?.warnings || [] as warning}<small>{warning}</small>{/each}
    {/if}
    <small>주소를 비우면 SparkTalk 호스트에서 실행합니다. 원격 호스트는 앱 실행 계정의 SSH 키와 known_hosts를 사용합니다. Extra SSH 서비스와는 별도 연결입니다.</small>
    <div class="host-selection">
      <label class="settings-field-row"><span>편집할 실행 호스트</span><select bind:value={selectedHost}>{#each Object.keys(catalog.hosts) as id}<option value={id}>{id === 'local' ? '로컬 (local)' : id === 'worker' ? '워커 (worker)' : id}</option>{/each}</select></label>
      <button type="button" onclick={addHost}>호스트 추가</button>
    </div>
    {#each Object.entries(catalog.hosts).filter(([id]) => id === selectedHost) as [id, host] (id)}
      <fieldset><legend>{id}</legend><div class="grid">
        <label class="settings-field-row"><span>SSH 주소</span><input bind:value={host.address} placeholder="빈 값: 로컬" /></label>
        <label class="settings-field-row"><span>SSH 사용자</span><input bind:value={host.user} /></label>
        <label class="settings-field-row settings-number-row"><span>SSH 포트</span><input type="number" min="0" max="65535" bind:value={host.port} /></label>
        <label class="settings-field-row"><span>SSH 개인키 경로</span><input bind:value={host.identity_file} placeholder="빈 값: 기본 SSH 설정" /></label>
        <label class="settings-field-row"><span>호스트의 운영 데이터 경로</span><input bind:value={host.data_dir} /></label>
        <label class="settings-field-row"><span>호스트의 모델 캐시 경로</span><input bind:value={host.model_cache} /></label>
        <label class="settings-field-row settings-number-row"><span>최소 확보 메모리 GiB</span><input type="number" min="0" step="0.5" bind:value={host.memory_reserve_gib} placeholder="0: 공통 설정" /></label>
      </div><button type="button" disabled={hostIsUsed(catalog, id)} onclick={() => { const next = { ...catalog.hosts }; delete next[id]; catalog.hosts = next; catalog = catalog; }}>미사용 호스트 삭제</button></fieldset>
    {/each}
  </details>
  <details>
    <summary>JSON / YAML 가져오기·내보내기</summary>
    <small>가져오기는 전체 세트 목록을 교체합니다. 불러온 뒤 시작·연결에서 기본 세트를 확인하세요.</small>
    <label class="settings-field-row"><span>세트 파일</span><input type="file" accept=".json,.yaml,.yml" onchange={importFile} /></label>
    <div class="buttons"><button type="button" onclick={download}>JSON 내보내기</button><button type="button" onclick={() => { source = JSON.stringify(catalog, null, 2); }}>현재 구성을 편집기에 복사</button></div>
    <label>전체 세트 정의<textarea rows="14" bind:value={source} spellcheck="false"></textarea></label>
    <button type="button" disabled={busy || !source.trim()} onclick={importSource}>검증 후 불러오기</button>
  </details>
  {#if message}<p role="status">{message}</p>{/if}
</fieldset>
{/if}

<style>
  .network-actions { display: flex; align-items: end; gap: 8px; flex-wrap: wrap; }
  .network-actions button { align-self: flex-end; }
  .set-editor { min-width: 0; }
  .set-editor details { border: 1px solid #80808040; border-radius: 9px; padding: 10px 12px; margin: 0; min-width: 0; }
  .set-editor summary { cursor: pointer; overflow-wrap: anywhere; font-size: 13px; }
  .set-editor .service-card { padding: 0; overflow: hidden; }
  .service-card > summary { display: flex; align-items: center; gap: 10px; padding: 12px; list-style: none; }
  .service-card > summary::-webkit-details-marker { display: none; }
  .service-card > summary::before { content: '›'; font-size: 18px; opacity: .65; }
  .service-card[open] > summary::before { transform: rotate(90deg); }
  .service-summary { flex: 1; min-width: 0; }
  .service-summary strong { display: block; font-size: 13px; font-weight: 600; }
  .service-summary small { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; margin-top: 4px; }
  .host-badge { background: #6584ed15; border: 1px solid #6584ed30; border-radius: 5px; padding: 3px 7px; font-size: 11px; }
  .service-connection { padding: 0 14px 14px; }
  .set-editor .service-advanced { border-width: 1px 0 0; border-radius: 0; background: #80808006; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(200px, 100%), 1fr)); gap: 8px; }
  .set-editor label { display: flex; flex-direction: column; gap: 4px; margin: 8px 0; }
  .set-editor label.check { flex-direction: row; }
  .set-editor input, .set-editor select, .set-editor textarea { min-width: 0; width: 100%; box-sizing: border-box; }
  .set-editor input[type=checkbox] { width: auto; }
  .set-heading { display: flex; align-items: center; justify-content: space-between; gap: 8px; font-size: 12px; }
  .set-heading strong { font-size: 13px; }
  .set-editor button { padding: 6px 10px; border: 1px solid #80808050; border-radius: 6px; background: transparent; color: inherit; font-size: 12px; }
  .set-editor button:hover:not(:disabled) { background: #6584ed15; }
  .set-editor button:disabled { opacity: .5; }
  .set-editor .danger { color: #ce5050; margin-top: 10px; }
  .buttons { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0; }
  .service-picker { padding: 10px; border: 1px solid #80808040; border-radius: 8px; }
  .host-selection { display: flex; align-items: end; gap: 10px; margin: 8px 0 12px; }
  .host-selection label { flex: 1; min-width: 0; margin: 0; }
  .host-selection button { flex-shrink: 0; min-height: 36px; }
  .picker-row { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
  .service-picker .check { margin: 0; padding: 8px 0; gap: 10px; }
  .service-picker small { margin-top: 3px; }
  .probe-result { font-size: 12px; overflow-wrap: anywhere; }
  textarea { font-family: monospace; }
  small { display: block; opacity: .85; font-size: 11px; line-height: 1.5; }
</style>
