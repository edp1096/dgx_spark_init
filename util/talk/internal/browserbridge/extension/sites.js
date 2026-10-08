const list=document.querySelector('#allowedSites'),status=document.querySelector('#siteStatus'),filter=document.querySelector('#filter');
const required=new Set(chrome.runtime.getManifest().host_permissions||[]);
function isRequired(origin){
 if(required.has(origin))return true;
 try{const granted=new URL(origin.replace('*.', ''));return [...required].some(pattern=>{const rule=new URL(pattern.replace('*.', ''));return rule.protocol===granted.protocol&&(pattern.includes('*.')?(granted.hostname===rule.hostname||granted.hostname.endsWith('.'+rule.hostname)):!origin.includes('*.')&&granted.hostname===rule.hostname);});}catch{return false;}
}
let origins=[],server='',revision=0;
function isServer(pattern){
 try{const site=new URL(server),permission=new URL(pattern.replace('*.', ''));return site.protocol===permission.protocol&&(site.hostname===permission.hostname||(pattern.includes('*.')&&site.hostname.endsWith('.'+permission.hostname)));}catch{return false;}
}
function render(){
 const query=filter.value.trim().toLowerCase(),shown=origins.filter(origin=>origin.toLowerCase().includes(query));
 list.replaceChildren();document.querySelector('#siteCount').textContent=`${origins.length}개`;
 const empty=document.querySelector('#empty');empty.hidden=shown.length>0;empty.textContent=origins.length?'검색 결과가 없습니다.':'허용된 사이트가 없습니다.';
 for(const origin of shown){
  const item=document.createElement('li'),info=document.createElement('div'),address=document.createElement('span'),kind=document.createElement('span');
  info.className='site-info';address.className='site-address';address.textContent=origin;kind.className='site-kind';
  kind.textContent=isRequired(origin)?'확장 기본 권한':isServer(origin)?'Talk 연결 서버 · 해제하면 서버 연결에 영향을 줄 수 있습니다.':'사용자가 허용한 사이트';
  info.append(address,kind);item.append(info);
  if(!isRequired(origin)){
   const button=document.createElement('button');button.className='remove';button.textContent='해제';button.setAttribute('aria-label',origin+' 권한 해제');
   button.onclick=async()=>{
    button.disabled=true;
    try{const removed=await chrome.permissions.remove({origins:[origin]});status.textContent=removed?`${origin} 권한을 해제했습니다.`:'권한을 해제하지 못했습니다.';await refresh();filter.focus();}
    catch(error){status.textContent=error.message;button.disabled=false;}
   };item.append(button);
  }
  list.append(item);
 }
}
async function refresh(){
 const current=++revision;
 try{const [permissions,settings]=await Promise.all([chrome.permissions.getAll(),chrome.storage.local.get('server')]);if(current!==revision)return;origins=[...new Set(permissions.origins||[])].sort();server=settings.server||'';render();}
 catch(error){status.textContent='사이트 권한 목록을 불러오지 못했습니다: '+error.message;}
}
document.querySelector('#addSite').onsubmit=async event=>{
 event.preventDefault();
 try{const u=new URL(document.querySelector('#site').value);if(!['http:','https:'].includes(u.protocol)||u.username||u.password)throw Error('http/https 사이트 주소를 입력하세요.');
 const allowed=await chrome.permissions.request({origins:[u.origin+'/*']});status.textContent=allowed?`${u.origin} 접근을 허용했습니다.`:'사이트 접근을 허용하지 않았습니다.';
 if(allowed){document.querySelector('#site').value='';filter.value='';}await refresh();
 }catch(error){status.textContent=error.message;}
};
filter.oninput=render;
chrome.permissions.onAdded.addListener(refresh);chrome.permissions.onRemoved.addListener(refresh);
chrome.storage.onChanged.addListener((changes,area)=>{if(area==='local'&&changes.server)refresh();});
refresh();
