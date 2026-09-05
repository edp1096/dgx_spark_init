<script>
 import { onMount, onDestroy } from 'svelte';
 export let catalog;
 let token = '', configured = false, saving = false, message = '', state = '';
 let logs = [], elapsed = '', startedAt = '';
 let component = '', variant = 'abliterated', timer;
 $: models = (catalog?.components || []).filter(c => ['glm53-cluster','dspark-cluster'].includes(c.controller) || ['compose.qwen27-exl3.yaml','compose.flash-next-exl3.yaml'].includes(c.compose_asset));
 $: if (!component && models.length) component = models[0].id;
 $: selectedModel = models.find(model => model.id === component);
 $: repositories = modelRepositories(selectedModel, variant);
 function modelRepositories(model, variant) {
  if (model?.controller === 'dspark-cluster') return [variant === 'abliterated' ? 'drowzeys/keys-DeepSeekV4Flash-Vision-EXP-ablit' : 'deepseek-ai/DeepSeek-V4-Flash-Vision-Exp'];
  if (model?.controller === 'glm53-cluster') return ['brandonmusic/GLM-5.3-Flash-tr3-4bpw', 'local-inference-lab/GLM-5.3-Flash-DFlash2-MXFP8', ...(variant === 'abliterated' ? ['lovesenko/GLM-5.3-Flash-tr3-4bpw-Abliterated'] : [])];
  if (model?.compose_asset === 'compose.qwen27-exl3.yaml') return ['Lygodactylus/Qwen3.8-27B-Uncensored-exl3-4bpw'];
  if (model?.compose_asset === 'compose.flash-next-exl3.yaml') return ['turboderp/Qwen3.8-Flash-Next-exl3', ...(variant === 'abliterated' ? ['Qwen/Qwen3.8-Flash-Next', 'windowsxp811203/Qwen3.8-Flash-Next-Abliterated'] : [])];
  return [];
 }
 async function request(path, method = 'GET', body) {
  const r = await fetch(path, {method, headers: {'Content-Type':'application/json'}, ...(body ? {body:JSON.stringify(body)} : {})});
  if (!r.ok) throw new Error(await r.text());
  return r.json();
 }
 async function status() {
  try { const r = await request('/api/models/prepare'); state = r.state; logs = r.logs || []; startedAt = r.started_at; const seconds = startedAt ? Math.max(0, Math.floor((Date.now()-Date.parse(startedAt))/1000)) : 0; elapsed = `${Math.floor(seconds/60)}분 ${seconds%60}초`; if (r.detail) message = r.detail; } catch (e) { message = e.message; }
 }
 async function save(remove = false) {
  saving = true;
  try { const r = await request('/api/credentials/huggingface', remove ? 'DELETE' : 'PUT', remove ? null : {token}); configured = r.configured; token = ''; message = remove ? '토큰을 삭제했습니다.' : '토큰을 저장했습니다.'; }
  catch (e) { message = e.message; } finally { saving = false; }
 }
 async function prepare(action) {
  try { await request('/api/models/prepare','POST',{component,variant,action}); state = 'running'; message = '모델 준비 중'; } catch (e) { message = e.message; }
 }
 onMount(async () => { try { configured = (await request('/api/credentials/huggingface')).configured; } catch (e) { message=e.message; } await status(); timer=setInterval(status,3000); });
 onDestroy(() => clearInterval(timer));
</script>
<fieldset>
 <legend>Hugging Face 인증</legend>
 <p>{configured ? '토큰 등록됨' : '등록된 토큰 없음'}</p>
 <label>{configured ? '새 토큰으로 교체' : '다운로드 토큰'}<input type="password" autocomplete="new-password" bind:value={token} placeholder="hf_…" /></label>
 <div class="buttons"><button type="button" disabled={saving || !token.trim()} onclick={() => save()}>토큰 저장</button><button type="button" disabled={saving || !configured} onclick={() => save(true)}>토큰 삭제</button></div>
 <small>앱 서버에 별도로 보관하며 설정 내보내기에 포함하지 않습니다. 접근 제한 모델은 Hugging Face에서 먼저 이용 조건에 동의해야 합니다.</small>
</fieldset>
<fieldset>
 <legend>모델 준비</legend>
 <label>모델<select bind:value={component}>{#each models as model}<option value={model.id}>{model.name}</option>{/each}</select></label>
 <label>가중치<select bind:value={variant}><option value="official">공식 원본</option><option value="abliterated">Abliterated</option></select></label>
 {#if repositories.length}
  <div class="model-sources">
   <strong>다운로드할 모델 저장소</strong>
   {#each repositories as repository}
    <a href={'https://huggingface.co/' + repository} target="_blank" rel="noopener noreferrer">{repository} ↗</a>
   {/each}
   <small>저장소를 열어 로그인하세요. 접근 동의·승인 요청이 표시되면 먼저 완료하고, 같은 계정의 읽기 권한 토큰을 등록하세요.</small>
  </div>
 {/if}
 <div class="buttons"><button type="button" disabled={!component || state === 'running'} onclick={() => prepare('model')}>모델만 준비</button><button type="button" disabled={!component || state === 'running'} onclick={() => prepare('setup')}>전체 준비</button></div>
 <small>앱에 내장된 절차로 준비합니다. 해당 모델을 중지한 뒤 실행하세요. 완료 후 AI 세트의 가중치를 선택해 기동합니다.</small>
 {#if state === 'running'}<div class="preparation-progress"><progress aria-label="모델 준비 진행 중"></progress><span>준비 중 · 경과 {elapsed}</span></div>{/if}
 <p role="status">{message}</p>
 {#if logs.length}<details open><summary>실시간 작업 로그</summary><pre class="preparation-log">{logs.join('\n')}</pre></details>{/if}
</fieldset>

<style>
 .preparation-progress { display: flex; gap: 0.75rem; align-items: center; }
 .preparation-log { max-height: 16rem; overflow: auto; white-space: pre-wrap; overflow-wrap: anywhere; font-size: 0.8rem; }
 .model-sources { display: grid; gap: 0.5rem; margin: 0.75rem 0; }
 .model-sources a { overflow-wrap: anywhere; }
</style>
