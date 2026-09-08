<script>
 import {onMount} from 'svelte';
 export let onnotify=()=>{};
 let items=[],skills=[],draft=null,original='',busy=false,search='',deleting=false;
 const tools={'':'작성 결과 확인',ssh_exec:'SSH 실행 성공',web_search:'웹 검색',web_fetch:'페이지 읽기',web_collect:'웹 수집',media_import:'미디어 가져오기',image_generate:'이미지 생성',knowledge_search:'지식 검색',document_generate:'문서 생성'};
 async function api(path,method='GET',body){const r=await fetch(path,{method,headers:{'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined});if(!r.ok)throw Error(await r.text());return r.status===204?null:r.json();}
 async function load(){try{[items,skills]=await Promise.all([api('/api/workflows'),api('/api/skills')]);}catch(e){onnotify(e.message,'error');}}
 onMount(load);
 const step=()=>({name:'',skills:[],goal:'',done_when:'',verify_tool:'',verify_command:'',on_failure:-1,max_retries:0});
 function edit(x,copy=false){draft=structuredClone(x);original=copy?'':x.name;if(copy){draft.name=x.name.slice(0,50)+'-copy';draft.builtin=false;}deleting=false;}
 function create(){original='';draft={name:'',description:'',enabled:true,builtin:false,steps:[step()]};}
 function move(i,delta){const j=i+delta;[draft.steps[i],draft.steps[j]]=[draft.steps[j],draft.steps[i]];draft.steps=draft.steps.map((s,index)=>{let n=s.on_failure;if(n===i)n=j;else if(n===j)n=i;return {...s,on_failure:n<index?n:-1};});}
 function remove(i){draft.steps=draft.steps.filter((_,n)=>n!==i).map(s=>({...s,on_failure:s.on_failure===i?-1:s.on_failure>i?s.on_failure-1:s.on_failure}));}
 async function save(){busy=true;try{if(!original&&items.some(x=>x.name===draft.name))throw Error('이미 사용 중인 이름입니다.');await api('/api/workflows/'+draft.name,'PUT',draft);draft=null;await load();onnotify('작업 절차를 저장했습니다. 진행 중인 작업에는 기존 구성이 유지됩니다.');}catch(e){onnotify(e.message,'error');}finally{busy=false;}}
 async function toggle(x){try{await api('/api/workflows/'+x.name,'PUT',x.builtin?{enabled:!x.enabled}:{...x,enabled:!x.enabled});await load();}catch(e){onnotify(e.message,'error');}}
 async function destroy(){try{await api('/api/workflows/'+original,'DELETE');draft=null;deleting=false;await load();}catch(e){onnotify(e.message,'error');}}
</script>
<section class="workflow-library">
 {#if draft}
 <form onsubmit={e=>{e.preventDefault();save();}}>
 <header><h3>{draft.builtin?'내장 작업 절차':'작업 절차 편집'}</h3><button type="button" onclick={()=>draft=null}>목록으로</button></header>
 <label>절차 이름<input aria-label="절차 이름" bind:value={draft.name} disabled={!!original} pattern={'[a-z0-9][a-z0-9-]{0,63}'} required /></label>
 <label>사용할 상황<textarea aria-label="절차 설명" bind:value={draft.description} readonly={draft.builtin} required maxlength="500" rows="2"></textarea></label>
 <p>각 단계에는 현재 단계의 스킬과 이전 단계 결과를 전달합니다. 도구 권한은 기존 설정을 따릅니다.</p>
 {#each draft.steps as s,i}<fieldset><legend>{i+1}. {s.name||'새 단계'}</legend>
 <header><label>단계 이름<input aria-label={`단계 ${i+1} 이름`} bind:value={s.name} readonly={draft.builtin} maxlength="80" required /></label>{#if !draft.builtin}<button type="button" aria-label={`단계 ${i+1} 위로`} disabled={i===0} onclick={()=>move(i,-1)}>↑</button><button type="button" aria-label={`단계 ${i+1} 아래로`} disabled={i===draft.steps.length-1} onclick={()=>move(i,1)}>↓</button><button type="button" disabled={draft.steps.length===1} onclick={()=>remove(i)}>단계 삭제</button>{/if}</header>
 <label>목표·필요 입력<textarea aria-label={`단계 ${i+1} 목표`} bind:value={s.goal} readonly={draft.builtin} required maxlength="4000" rows="2"></textarea></label>
 <details><summary>사용 스킬 · {s.skills.join(', ')||'선택 필요'}</summary><div class="checks">{#each skills as skill}<label><input type="checkbox" bind:group={s.skills} value={skill.name} disabled={draft.builtin} />{skill.name}</label>{/each}</div></details>
 <label>완료 조건<textarea aria-label={`단계 ${i+1} 완료 조건`} bind:value={s.done_when} readonly={draft.builtin} required maxlength="2000" rows="2"></textarea></label>
 <details><summary>검증·재시도 · {tools[s.verify_tool]||s.verify_tool}</summary>
 <label>필요한 실행 근거<select bind:value={s.verify_tool} onchange={()=>{if(s.verify_tool!=='ssh_exec')s.verify_command='';}} disabled={draft.builtin}>{#each Object.entries(tools) as [value,label]}<option {value}>{label}</option>{/each}</select></label>
 {#if s.verify_tool==='ssh_exec'}<label>검증 명령 일치 (선택)<input bind:value={s.verify_command} readonly={draft.builtin} placeholder="예: go test ./..." /><small>입력하면 이 명령의 성공 결과만 검증 근거로 인정합니다.</small></label>{/if}
 <label>검증 실패 시 돌아갈 단계<select bind:value={s.on_failure} disabled={draft.builtin}><option value={-1}>중단</option>{#each draft.steps.slice(0,i) as prev,j}<option value={j}>{j+1}. {prev.name}</option>{/each}</select></label>
 {#if s.on_failure>=0}<label>자동 재시도 횟수<input type="number" min="0" max="2" bind:value={s.max_retries} readonly={draft.builtin} /></label>{/if}
 </details></fieldset>{/each}
 {#if draft.builtin}<button type="button" onclick={()=>edit(draft,true)}>복사해서 수정</button>{:else}<header><button type="button" disabled={draft.steps.length>=8} onclick={()=>draft.steps=[...draft.steps,step()]}>단계 추가</button><label class="check"><input type="checkbox" bind:checked={draft.enabled} />사용</label><button type="submit" disabled={busy}>저장</button>{#if original}<button type="button" onclick={()=>deleting=!deleting}>삭제</button>{/if}</header>{/if}
 {#if deleting}<p role="alert">기존 실행 기록은 남습니다. <button type="button" onclick={destroy}>삭제 확인</button></p>{/if}
 </form>
 {:else}
 <header><h3>작업 절차</h3><button onclick={create}>새 작업 절차</button></header>
 <p>스킬을 단계별로 묶어 실행합니다. 완료 기준에 대한 모델의 판단과 실제 도구 실행 근거를 함께 확인할 수 있습니다.</p>
 <input aria-label="작업 절차 검색" placeholder="작업 절차 검색" bind:value={search} />
 {#each items.filter(x=>(x.name+' '+x.description).toLowerCase().includes(search.toLowerCase())) as x}<article>
 <header><button onclick={()=>edit(x)}>{x.name}</button><small>{x.builtin?'내장':'사용자'} · {x.steps.length}단계</small><label class="check"><input type="checkbox" checked={x.enabled} onchange={()=>toggle(x)} />사용</label></header>
 <p>{x.description}</p><small>{x.steps.map(s=>s.name).join(' → ')}</small><header><button onclick={()=>edit(x)}>구성 보기</button><button onclick={()=>edit(x,true)}>복사</button></header>
 </article>{/each}
 {/if}
</section>
<style>
.workflow-library{max-width:960px;padding:20px;margin:auto}header{display:flex;gap:10px;align-items:center;justify-content:space-between;flex-wrap:wrap}h3{font-size:16px}p,small{font-size:12px;opacity:.8}article,fieldset{border:1px solid var(--border,#3b4557);border-radius:10px;padding:14px;margin:12px 0;min-width:0}label{display:grid;gap:6px;margin:10px 0;font-size:13px}label.check,.checks label{display:flex;align-items:center}.checks{display:flex;flex-wrap:wrap;gap:10px}input:not([type=checkbox]),textarea,select{width:100%;box-sizing:border-box}textarea{resize:vertical}details{margin:12px 0}button{cursor:pointer;background:transparent;color:inherit;border:1px solid var(--border,#3b4557);border-radius:7px;padding:7px 10px}button:disabled{opacity:.4}header>label{flex:1}input[type=checkbox]{width:auto}@media(max-width:600px){.workflow-library{padding:12px}header{justify-content:flex-start}}
</style>
