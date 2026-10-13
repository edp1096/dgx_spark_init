<script>
 import {onDestroy} from 'svelte';
 import DocumentTaskCard from './DocumentTaskCard.svelte';
 import { documentTaskFinished, documentTaskCompletionLabel } from '../lib/document-task.js';
 export let sessionId='';export let version=0;export let running=false;export let onResume=()=>{};
 let runs=[],error='',epoch=0,idleTicks=0,expanded=false,loadedSession='';
 const status={pending:'대기',running:'진행 중',completed:'완료',paused:'중단',failed:'실패',blocked:'진행 불가',unverified:'검증 못함',waiting_model:'모델 대기',needs_review:'검토 필요',canceled:'취소'};
 $: refresh(sessionId,version,running);
 async function refresh(id,_version,_running){const request=++epoch;if(!id){runs=[];loadedSession='';return;}try{const r=await fetch(`/api/sessions/${id}/workflows`);if(!r.ok)throw Error(await r.text());const data=await r.json();if(request===epoch){if(loadedSession!==id){expanded=_running||data.some(r=>r.document);loadedSession=id;}else if(data.some(r=>(r.document||_running)&&!runs.some(previous=>previous.id===r.id))){expanded=true;}runs=data;error='';}}catch(e){if(request===epoch)error=e.message;}}
 const poll=setInterval(()=>{if(!sessionId)return;idleTicks++;if(runs.some(r=>r.document&&(r.status==='running'||r.conversions?.some(item=>item.status==='running')))||idleTicks>=5){idleTicks=0;refresh(sessionId,version,running);}},3000);
 onDestroy(()=>{epoch++;clearInterval(poll);});
</script>
{#if runs.length || error}<details class="workflow-runs" bind:open={expanded}>
 <summary><span>작업{runs.length>1?` ${runs.length}개`:''}</span><span class="workflow-summary-state">{runs[0]?.document?({translate:'문서 번역',summarize:'문서 요약',compare:'자료 비교'})[runs[0].document.kind]||'문서 작업':runs[0]?.definition.name||''} · {documentTaskFinished(runs[0])?documentTaskCompletionLabel(runs[0]):status[runs[0]?.status]||''}</span></summary>
 {#if error}<p role="alert">{error}</p>{/if}
 {#each runs as run (run.id)}{#if run.document}<DocumentTaskCard {run} {sessionId} onChange={()=>refresh(sessionId,version,running)} />{:else}<details open={run.status==='running'}><summary>{run.definition.name} · {status[run.status]} · {Math.min(run.current+1,run.steps.length)}/{run.steps.length}</summary>
 {#each run.steps as step,i}<details><summary>{i+1}. {run.definition.steps[i].name} · {status[step.status]}{step.attempts>1?` · ${step.attempts}회 시도`:''}</summary>
 <p>{run.definition.steps[i].done_when}</p>{#if step.error}<p role="alert">{step.error}</p>{/if}<pre>{step.summary}</pre>
 {#if step.history?.length}<details><summary>이전 시도 {step.history.length}회</summary>{#each step.history as attempt}<details><summary>{status[attempt.status]||attempt.status}</summary><pre>{attempt.summary}</pre><pre>{attempt.error||''}</pre>{#each attempt.evidence||[] as e}<details><summary>{e.tool}</summary><pre>{e.arguments}</pre><pre>{e.error||e.result}</pre></details>{/each}</details>{/each}</details>{/if}
 {#each step.evidence||[] as e}<details><summary>{e.tool} · {e.id}{e.error?' · 오류':''}</summary><pre>{e.arguments}</pre><pre>{e.error||e.result}</pre></details>{/each}
 </details>{/each}
 {#if run.status!=='running'&&run.status!=='completed'}<button disabled={running} onclick={()=>onResume(run.id)}>이 작업 이어하기</button><small>입력창에서 요청을 확인하고 전송하면 재개됩니다.</small>{/if}
 </details>{/if}{/each}
 </details>{/if}
<style>
 .workflow-runs{width:min(820px,calc(100% - 32px));min-width:0;min-height:0;margin:0 auto 12px;max-height:min(48dvh,460px);overflow:auto;scrollbar-width:thin;font-size:12px}
 .workflow-runs > summary{padding:7px 2px;color:#a0aaba;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
 .workflow-summary-state{margin-left:10px;font-weight:400}
 .workflow-runs[open] > summary .workflow-summary-state{display:none}
 .workflow-runs :global(.document-task + .document-task){margin-top:10px}
 :global(html[data-theme="light"]) .workflow-runs > summary{color:#657287}
 details details{margin:8px 0 8px 10px}summary{cursor:pointer}summary:focus-visible{outline:2px solid #789bfa;outline-offset:-2px;border-radius:5px}
 pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:240px;overflow:auto}button{padding:6px 10px;border:1px solid var(--border,#3b4557);border-radius:6px;background:transparent;color:inherit}p[role=alert]{color:#ee9b80}
 @media(max-width:520px){.workflow-runs{width:calc(100% - 24px);margin-bottom:10px}.workflow-runs > summary{min-height:36px;padding:9px 2px}}
</style>
