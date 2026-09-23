const urlEl=document.querySelector('#url'),tokenEl=document.querySelector('#token'),status=document.querySelector('#status');
const touched=new Set(),writes=[];
// Store each draft field immediately: closing the popup must not discard a paste.
for(const [el,key] of [[urlEl,'connectionDraftURL'],[tokenEl,'connectionDraftToken']]){
 el.addEventListener('input',()=>{
  touched.add(key);
  const write=chrome.storage.local.set({[key]:el.value});
  writes.push(write);
  write.catch(()=>{status.textContent='임시 입력 저장 실패';});
 });
}
chrome.storage.local.get(['server','token','connectionDraftURL','connectionDraftToken']).then(v=>{
 if(!touched.has('connectionDraftURL'))urlEl.value=v.connectionDraftURL??v.server??'';
 if(!touched.has('connectionDraftToken'))tokenEl.value=v.connectionDraftToken??v.token??'';
}).catch(()=>{status.textContent='저장된 입력을 불러오지 못했습니다.';});
async function refresh(){try{const r=await chrome.runtime.sendMessage({type:'STATUS'});status.textContent=r.connected?'연결됨':'연결 안 됨';}catch{status.textContent='연결 상태 확인 실패';}}
refresh();
document.querySelector('#connect').onclick=async()=>{
 try{
  const u=new URL(urlEl.value);if(!['http:','https:'].includes(u.protocol)||u.username||u.password)throw Error('http 또는 https 서버 주소를 입력하세요.');
  const token=tokenEl.value.trim();if(!/^[a-f0-9]{64}$/.test(token))throw Error('Talk에서 새 연결 키를 발급해 입력하세요.');
  if(!await chrome.permissions.request({origins:[u.origin+'/*']}))throw Error('Talk 서버 접근 권한이 필요합니다.');
  await Promise.all(writes);
  await chrome.storage.local.set({server:u.origin,token});
  await chrome.storage.local.remove(['connectionDraftURL','connectionDraftToken']);
  urlEl.value=u.origin;tokenEl.value=token;
  await chrome.runtime.sendMessage({type:'CONNECT'});status.textContent='연결 중…';setTimeout(refresh,1500);
 }catch(e){status.textContent=e.message;}
};
document.querySelector('#disconnect').onclick=async()=>{
 try{
  await Promise.all(writes);
  await chrome.storage.local.remove(['token','connectionDraftToken']);tokenEl.value='';
  await chrome.runtime.sendMessage({type:'CONNECT'});await refresh();
 }catch(e){status.textContent=e.message;}
};

async function refreshSiteCount(){
 try{const permissions=await chrome.permissions.getAll();document.querySelector('#siteCount').textContent=`허용된 사이트 패턴 ${(permissions.origins||[]).length}개`;}catch{document.querySelector('#siteCount').textContent='사이트 권한을 확인할 수 없습니다.';}
}
document.querySelector('#manageSites').onclick=()=>chrome.runtime.openOptionsPage();
chrome.permissions.onAdded.addListener(refreshSiteCount);
chrome.permissions.onRemoved.addListener(refreshSiteCount);
refreshSiteCount();

document.querySelector('#releaseTabs').onclick=async()=>{
 try{const result=await chrome.runtime.sendMessage({type:'RELEASE_TABS'});status.textContent=result.ok?'작업을 중지하고 탭 연결을 해제했습니다. 탭은 닫지 않았습니다.':result.error;}catch(e){status.textContent=e.message;}
};
