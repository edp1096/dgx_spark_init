// Runs in the page world at document_start, before the review app can block
// on its resume-draft confirm. Preserve all unrelated native dialogs.
(() => {
 if(window.__sparkTalkResumeDraft)return;
 const nativeConfirm=window.confirm;
 const isResume=message=>/리뷰|작성\s*중|임시\s*저장/.test(message)&&/이어서\s*(?:작성|쓰)|계속\s*작성|작성.*계속/.test(message);
 window.confirm=function(message){
  if(isResume(String(message||''))){
   document.documentElement?.setAttribute('data-sparktalk-resumed-draft','true');
   return true;
  }
  return nativeConfirm.call(window,message);
 };
 window.__sparkTalkResumeDraft=true;
})();
