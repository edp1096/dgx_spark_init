<script>
 let state=null,busy=false,error='';
 async function request(action=''){
  busy=true;error='';
  try{
   const url='/api/media/yt-dlp'+(action==='check'?'?check=1':action?'/'+action:'');
   const response=await fetch(url,{method:action&&action!=='check'?'POST':'GET'});
   if(!response.ok)throw Error(await response.text());state=await response.json();
  }catch(e){error=e.message;}finally{busy=false;}
 }
</script>
<div class="runtime">
 <strong>yt-dlp 다운로드 엔진</strong>
 <p>{state?`현재 버전: ${state.current}`:'버전 확인을 누르면 현재 미디어 서비스의 상태를 조회합니다.'}{state?.latest?` · 최신: ${state.latest}`:''}</p>
 <div class="actions">
  <button type="button" disabled={busy} onclick={()=>request()}>버전 확인</button>
  <button type="button" disabled={busy} onclick={()=>request('check')}>업데이트 확인</button>
  <button type="button" disabled={busy||!state?.update_available} onclick={()=>request('update')}>최신 버전 설치</button>
  <button type="button" disabled={busy||!state?.can_rollback} onclick={()=>request('rollback')}>이전 버전 복원</button>
 </div>
 {#if busy}<p role="status">처리 중…</p>{/if}
 {#if error}<p role="alert">{error}</p>{/if}
 <small>버전 확인은 설치하지 않습니다. 설치 시 공식 배포 파일의 SHA256과 실행 버전을 검증하며, 적용한 버전은 서비스 재시작 후에도 유지됩니다.</small>
</div>
<style>.runtime{margin:14px 0;padding:12px;border:1px solid #80808040;border-radius:8px}.actions{display:flex;gap:8px;flex-wrap:wrap}p{font-size:13px}small{display:block;margin-top:10px;opacity:.75}</style>
