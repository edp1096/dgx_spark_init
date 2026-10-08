<script>
 import { speechTextFromMarkdown } from '../lib/speech-text.js';
 export let original='';
 export let omitParentheticals=false;
 let opened=false,loading=false,error='',parts=[],converted='',generation=0;
 async function preview(){
  const id=++generation;
  converted=speechTextFromMarkdown(original,{omitParentheticals});parts=[];error='';loading=true;
  try{
   const response=await fetch('/api/tts/preview',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:converted})});
   if(!response.ok)throw Error(await response.text());
   const data=await response.json();if(id===generation)parts=data.parts||[];
  }catch(e){if(id===generation)error=e.message;}finally{if(id===generation)loading=false;}
 }
 $: if(opened && original!==undefined && omitParentheticals!==undefined) preview();
</script>
<details bind:open={opened} class="speech-preview">
 <summary>낭독문 확인</summary>
 {#if opened}
  <strong>응답 원문</strong><pre>{original}</pre>
  <strong>변환된 낭독문</strong><pre>{converted}</pre>
  <strong>TTS 전달 텍스트 · 현재 설정 기준</strong>
  {#if loading}<p>확인 중…</p>{:else if error}<p role="alert">{error}</p>{:else}
   {#each parts as part}<div>{part.language}</div><pre>{part.text}</pre>{/each}
  {/if}
 {/if}
</details>
<style>
 .speech-preview{margin-top:8px;font-size:13px;max-width:100%;}summary{cursor:pointer;opacity:.75;}strong{display:block;margin-top:12px;}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit;padding:10px;border:1px solid #80808040;border-radius:8px;max-height:240px;overflow:auto;}
</style>
