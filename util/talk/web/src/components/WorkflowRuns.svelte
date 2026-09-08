<script>
 import {onDestroy} from 'svelte';
 export let sessionId='';export let version=0;export let running=false;export let onResume=()=>{};
 let runs=[],error='',epoch=0;
 const status={pending:'대기',running:'진행 중',completed:'완료',paused:'중단',failed:'실패',blocked:'진행 불가',unverified:'검증 못함'};
 $: refresh(sessionId,version,running);
 async function refresh(id,_version,_running){const request=++epoch;if(!id){runs=[];return;}try{const r=await fetch(`/api/sessions/${id}/workflows`);if(!r.ok)throw Error(await r.text());const data=await r.json();if(request===epoch){runs=data;error='';}}catch(e){if(request===epoch)error=e.message;}}
 onDestroy(()=>epoch++);
</script>
{#if runs.length || error}<details class="workflow-runs" open={running}>
 <summary>작업 진행 · {runs[0]?.definition.name||''} {status[runs[0]?.status]||''}</summary>
 {#if error}<p role="alert">{error}</p>{/if}
 {#each runs as run}<details open={run.status==='running'}><summary>{run.definition.name} · {status[run.status]} · {Math.min(run.current+1,run.steps.length)}/{run.steps.length}</summary>
 {#each run.steps as step,i}<details><summary>{i+1}. {run.definition.steps[i].name} · {status[step.status]}{step.attempts>1?` · ${step.attempts}회 시도`:''}</summary>
 <p>{run.definition.steps[i].done_when}</p>{#if step.error}<p role="alert">{step.error}</p>{/if}<pre>{step.summary}</pre>
 {#if step.history?.length}<details><summary>이전 시도 {step.history.length}회</summary>{#each step.history as attempt}<details><summary>{status[attempt.status]||attempt.status}</summary><pre>{attempt.summary}</pre><pre>{attempt.error||''}</pre>{#each attempt.evidence||[] as e}<details><summary>{e.tool}</summary><pre>{e.arguments}</pre><pre>{e.error||e.result}</pre></details>{/each}</details>{/each}</details>{/if}
 {#each step.evidence||[] as e}<details><summary>{e.tool} · {e.id}{e.error?' · 오류':''}</summary><pre>{e.arguments}</pre><pre>{e.error||e.result}</pre></details>{/each}
 </details>{/each}
 {#if run.status!=='running'&&run.status!=='completed'}<button disabled={running} onclick={()=>onResume(run.id)}>이 작업 이어하기</button><small>입력창에서 요청을 확인하고 전송하면 재개됩니다.</small>{/if}
 </details>{/each}
 </details>{/if}
<style>.workflow-runs{border:1px solid var(--border,#3b4557);border-radius:8px;margin:8px 14px;padding:10px;max-height:40vh;overflow:auto;font-size:12px}details details{margin:8px 0 8px 10px}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:240px;overflow:auto}button{padding:6px 10px;border:1px solid var(--border,#3b4557);border-radius:6px;background:transparent;color:inherit}p[role=alert]{color:#ee9b80}</style>
