// Adapted from the user-supplied naver-review-extension-mvp (Talk attachment).
// Mutations address exactly one inspected frame, never broadcast across frames.
let socket,connecting=false,heartbeat,authenticated=false;
const cancelled=new Set();let activeID;
const allowed=url=>{try{const u=new URL(url);return u.protocol==='https:'&&['naver.com','naverpay.com'].some(h=>u.hostname===h||u.hostname.endsWith('.'+h));}catch{return false;}};
const pause=ms=>new Promise(r=>setTimeout(r,ms));
async function connect(){
 if(connecting||socket?.readyState===WebSocket.OPEN)return;
 connecting=true;
 try{
  const {server,token}=await chrome.storage.local.get(['server','token']);if(!server||!token)return;
  const u=new URL('/api/browser/connect',server);u.protocol=u.protocol==='https:'?'wss:':'ws:';
  const ws=new WebSocket(u);socket=ws;
  ws.onopen=()=>{ws.send(JSON.stringify({token,protocol:8}));clearInterval(heartbeat);heartbeat=setInterval(()=>{if(ws.readyState===WebSocket.OPEN)ws.send('{}');},20000);};
  let serial=Promise.resolve();
  ws.onmessage=e=>{const cmd=JSON.parse(e.data);if(cmd.ready){authenticated=true;return;}if(cmd.cancel){cancelled.add(cmd.cancel);return;}serial=serial.then(async()=>{activeID=cmd.id;let result;try{result=await run(cmd);}catch(err){result={ok:false,error:err.message,observation:err.observation};}if(ws.readyState===WebSocket.OPEN)ws.send(JSON.stringify({id:cmd.id,result}));}).catch(()=>{});};
  ws.onclose=()=>{if(socket!==ws)return;clearInterval(heartbeat);if(activeID)cancelled.add(activeID);authenticated=false;socket=null;};
 }finally{connecting=false;}
}
chrome.alarms.create('bridge',{periodInMinutes:1});
chrome.alarms.onAlarm.addListener(a=>{if(a.name==='bridge')connect();});
chrome.runtime.onStartup.addListener(connect);
chrome.runtime.onMessage.addListener((m,s,reply)=>{
 if(s.id!==chrome.runtime.id||!s.url?.startsWith(chrome.runtime.getURL('')))return;
 if(m.type==='STATUS'){reply({connected:authenticated&&socket?.readyState===WebSocket.OPEN});return;}
 if(m.type==='CONNECT'){socket?.close();socket=null;authenticated=false;clearInterval(heartbeat);connect().then(()=>reply({ok:true}));return true;}
});
connect();
async function frames(tabId){
 const tab=await chrome.tabs.get(tabId);if(!allowed(tab.url))throw Error('네이버 탭만 제어할 수 있습니다.');
 await chrome.scripting.executeScript({target:{tabId,allFrames:true},files:['naver-content.js']}).catch(()=>{});
 return (await chrome.webNavigation.getAllFrames({tabId})).filter(f=>allowed(f.url));
}
async function send(tabId,frameId,msg){return chrome.tabs.sendMessage(tabId,{...msg,type:'TALK_REVIEW_V8'},{frameId});}
async function inspect(tabId,product=''){const out=[];for(const f of await frames(tabId)){try{const r=await send(tabId,f.frameId,{action:'inspect',product});if(r?.ok)out.push({frame_id:f.frameId,...r});}catch(e){out.push({frame_id:f.frameId,url:f.url,ok:false,error:e.message});}}return out;}
async function remember(items){const {targets={}}=await chrome.storage.session.get('targets');for(const v of items)targets[v.id]=v;const values=Object.values(targets).sort((a,b)=>b.at-a.at).slice(0,500);await chrome.storage.session.set({targets:Object.fromEntries(values.map(v=>[v.id,v]))});}
async function getTarget(id){const {targets={}}=await chrome.storage.session.get('targets');const t=targets[id];if(!t||Date.now()-t.at>30*60*1000)throw Error('상품 대상이 만료됐습니다. 구매상품 목록을 다시 조회하세요.');return t;}
async function realReviewClick(t,check,request={action:'prepare_open',id:t.id,url:t.url}){
 const debuggee={tabId:t.tab_id};let attached=false,pressed=false;
 try{
  await chrome.debugger.attach(debuggee,'1.3');attached=true;check();
  const prepared=await send(t.tab_id,t.frame_id,request);
  if(!prepared?.ok)throw Error(prepared?.error||'클릭 대상을 확인하지 못했습니다.');
  let point=prepared.point;
  const allFrames=await chrome.webNavigation.getAllFrames({tabId:t.tab_id});
  let frame=allFrames.find(f=>f.frameId===t.frame_id);
  if(!frame)throw Error('대상 프레임이 변경됐습니다.');
  while(frame.parentFrameId>=0){
   const results=await chrome.scripting.executeScript({target:{tabId:t.tab_id,frameIds:[frame.parentFrameId]},func:(url,p)=>{
    const candidates=[...document.querySelectorAll('iframe')].filter(el=>el.src===url);
    if(candidates.length!==1)throw Error('iframe 위치를 하나로 확인할 수 없습니다.');
    const el=candidates[0],r=el.getBoundingClientRect(),sx=r.width/el.offsetWidth,sy=r.height/el.offsetHeight;
    const result={x:r.left+(el.clientLeft+p.x)*sx,y:r.top+(el.clientTop+p.y)*sy};
    if(document.elementFromPoint(result.x,result.y)!==el)throw Error('iframe 클릭 위치가 가려져 있습니다.');
    return result;
   },args:[frame.url,point]});
   point=results[0]?.result;if(!point)throw Error('iframe 좌표 확인 실패');
   frame=allFrames.find(f=>f.frameId===frame.parentFrameId);if(!frame)throw Error('상위 프레임 확인 실패');
  }
  check();
  if(prepared.receipt){const verified=await send(t.tab_id,t.frame_id,{...request,action:'verify_snapshot'});if(!verified?.ok)throw Error(verified?.error||'등록 전 내용 확인 실패');}
  for(const type of ['mouseMoved','mousePressed','mouseReleased']){
   if(type==='mousePressed')pressed=true;
   await chrome.debugger.sendCommand(debuggee,'Input.dispatchMouseEvent',{type,x:point.x,y:point.y,button:type==='mouseMoved'?'none':'left',buttons:type==='mousePressed'?1:0,clickCount:type==='mouseMoved'?0:1});
  }
  let report={};try{report=await send(t.tab_id,t.frame_id,{action:'click_report',id:prepared.click_id||t.id});}catch{}
  return {ok:true,click_dispatched:true,method:'chrome_mouse_input',label:prepared.label,trusted_event:report.trusted_event??null,user_activation:report.user_activation??null,receipt:prepared.receipt};
 }catch(error){error.pointerPressed=pressed;throw error;}finally{if(attached)await chrome.debugger.detach(debuggee).catch(()=>{});}
}
async function fillPopup(form,draft,check){
 const target={tab_id:form.tabId,frame_id:form.frameId};
 const fields={form_id:form.id,product:draft.product,text:draft.text,rating:draft.rating};
 const selected=await realReviewClick(target,check,{...fields,action:'prepare_rating'});
 const debuggee={tabId:form.tabId};let attached=false;
 try{
  await chrome.debugger.attach(debuggee,'1.3');attached=true;check();
  const prepared=await send(form.tabId,form.frameId,{...fields,action:'prepare_text'});
  if(!prepared?.ok)throw Error(prepared?.error||'입력칸 선택 실패');
  await chrome.debugger.sendCommand(debuggee,'Input.insertText',{text:draft.text});
 }finally{if(attached)await chrome.debugger.detach(debuggee).catch(()=>{});}
 await pause(200);check();
 const actual=await send(form.tabId,form.frameId,{...fields,action:'verify_snapshot'});
 if(!actual?.ok)throw Error(actual?.error||'입력 결과 불일치');
 return {...actual,status:'filled',submitted:false,rating_input:selected.method};
}
async function submitPopup(form,snapshot,check){
 const target={tab_id:form.tabId,frame_id:form.frameId};let pressed=false;
 try{
  const clicked=await realReviewClick(target,check,{action:'prepare_submit',form_id:form.id,product:snapshot.product,text:snapshot.text,rating:snapshot.rating});
  pressed=true;
  return await send(form.tabId,form.frameId,{action:'observe_submit',receipt:clicked.receipt});
 }catch(error){
  const attempted=pressed||!!error.pointerPressed;
  return {ok:false,status:attempted?'uncertain':'blocked',attempted_submit:attempted,error:error.message};
 }
}

const openedForms=new Map();
async function openForm(t,check){
 const cached=openedForms.get(t.id);
 if(cached){
  try{const detail=await inspect(cached.tabId,t.product);
   if(detail.some(f=>f.frame_id===cached.frameId&&(f.forms||[]).some(x=>x.id===cached.id&&x.product_matched)))return cached;
  }catch{}
  openedForms.delete(t.id);
 }
 const windowsBefore=await chrome.windows.getAll({windowTypes:['normal','popup']});
 const windowTypes=new Map(windowsBefore.map(w=>[w.id,w.type]));
 const tabView=tab=>({id:tab.id,window_id:tab.windowId,window_type:windowTypes.get(tab.windowId),url:tab.url||tab.pendingUrl,pending_url:tab.pendingUrl,status:tab.status,title:tab.title,opener_tab_id:tab.openerTabId});
 const before=(await chrome.tabs.query({})).filter(tab=>allowed(tab.url)).map(tabView);
 const beforeByID=new Map(before.map(tab=>[tab.id,tab]));
 const beforePopupForms=new Map();
 for(const tab of before.filter(tab=>tab.window_type==='popup')){try{const detail=await inspect(tab.id,t.product);beforePopupForms.set(tab.id,JSON.stringify(detail.map(f=>f.forms||[])));}catch{}}
 check();await frames(t.tab_id);
 let opened;
 try{opened=await realReviewClick(t,check);}
 catch(e){opened={ok:false,click_dispatched:null,error:e.message};}
 const observation={before_windows:windowsBefore.map(w=>({id:w.id,type:w.type})),before_tabs:before,open_result:opened,new_tabs:[],changed_tabs:[],new_windows:[],reused_windows:[],observed_frames:[]};
 let form;
 for(let n=0;n<20&&!form;n++){
  await pause(400);check();
  const windowsAfter=await chrome.windows.getAll({windowTypes:['normal','popup']});for(const w of windowsAfter)windowTypes.set(w.id,w.type);
  observation.new_windows=windowsAfter.filter(w=>!windowsBefore.some(old=>old.id===w.id)).map(w=>({id:w.id,type:w.type}));
  const after=(await chrome.tabs.query({})).filter(tab=>allowed(tab.url||tab.pendingUrl)).map(tabView);
  observation.new_tabs=after.filter(tab=>!beforeByID.has(tab.id));
  observation.changed_tabs=after.filter(tab=>beforeByID.has(tab.id)&&beforeByID.get(tab.id).url!==tab.url);
  // Pre-existing unrelated review-detail tabs are not evidence of this click.
  // Also inspect new noopener windows: they need not have openerTabId set.
  const candidates=after.filter(tab=>tab.id===t.tab_id||!beforeByID.has(tab.id)||observation.changed_tabs.some(v=>v.id===tab.id)||tab.window_type==='popup');
  const matches=[],observed=[];
  for(const tab of candidates){
   try{
    const detail=await inspect(tab.id,t.product);
    const reused=beforePopupForms.has(tab.id)&&JSON.stringify(detail.map(f=>f.forms||[]))!==beforePopupForms.get(tab.id);
    if(reused&&!observation.reused_windows.some(w=>w.id===tab.window_id))observation.reused_windows.push({id:tab.window_id,type:tab.window_type});
    const eligible=tab.id===t.tab_id||!beforeByID.has(tab.id)||observation.changed_tabs.some(v=>v.id===tab.id)||reused;
    for(const f of detail){
     observed.push({window_id:tab.window_id,window_type:tab.window_type,tab_id:tab.id,frame_id:f.frame_id,url:f.url,forms:f.forms||[],editors:f.editors||[],error:f.error});
     for(const x of f.forms||[])if(eligible&&x.product_matched)matches.push({windowId:tab.window_id,windowType:tab.window_type,tabId:tab.id,frameId:f.frame_id,...x});
    }
   }catch(e){observed.push({tab_id:tab.id,url:tab.url,error:e.message});}
  }
  observation.observed_frames=observed;
  if(matches.length>1){observation.failure='ambiguous_editors';break;}
  if(matches.length===1)form=matches[0];
 }
 if(!form){
  observation.failure ||= observation.observed_frames.some(f=>f.editors?.length)?'editor_detected_but_unmatched':observation.new_tabs.length||observation.changed_tabs.length?'navigation_observed_without_editor':'no_editor_or_navigation_observed';
  const err=Error('상품명이 일치하는 리뷰 입력창을 확인하지 못했습니다. 클릭 전후 관찰 결과를 확인하세요.');
  err.observation=observation;throw err;
 }
 form.observation=observation;openedForms.set(t.id,form);return form;
}
async function run(cmd){
 const a=cmd.args||{};
 const check=()=>{if(cancelled.has(cmd.id)||!authenticated||(cmd.expires&&Date.now()>cmd.expires))throw Error('브라우저 작업이 취소되거나 연결이 끊겼습니다.');};check();
 if(cmd.action==='tabs')return {ok:true,tabs:(await chrome.tabs.query({})).filter(t=>allowed(t.url)).map(t=>({tab_id:t.id,title:t.title,url:t.url}))};
 if(['inspect','navigate','scroll'].includes(cmd.action)){
  let tabId=a.tab_id,details,actionResult;
  if(cmd.action==='navigate'){
   const target=await getTarget(a.target_id);tabId=target.tab_id;
   if(target.kind!=='navigation')throw Error('조회 결과의 navigation 대상만 이동할 수 있습니다.');
   check();await frames(tabId);
   try{actionResult=await send(tabId,target.frame_id,{action:'navigate',id:target.id});}catch(e){actionResult={ok:false,status:'navigation_unconfirmed',error:e.message};}
   if(actionResult?.ok===false&&actionResult.status!=='navigation_unconfirmed')return actionResult;
   await pause(1800);check();details=await inspect(tabId);
  }else if(cmd.action==='scroll'){
   const available=await frames(tabId);const selected=available.find(f=>f.frameId===(a.frame_id||0));
   if(!selected)throw Error('스크롤할 프레임을 찾지 못했습니다.');
   const result=await send(tabId,selected.frameId,{action:'scroll',steps:a.steps});
   if(!result?.ok)return result||{ok:false,error:'스크롤 응답 없음'};
   details=[{frame_id:selected.frameId,...result}];
  }else details=await inspect(tabId);
  const items=[],navigation=[];
  for(const f of details){
   for(const item of f.items||[])items.push({...item,kind:'item',tab_id:tabId,frame_id:f.frame_id,url:f.url,at:Date.now()});
   for(const item of f.navigation||[])navigation.push({...item,kind:'navigation',tab_id:tabId,frame_id:f.frame_id,url:f.url,at:Date.now()});
  }
  await remember([...items,...navigation]);
  const extractionFailed=items.length===0&&details.some(f=>f.status==='extraction_failed');
  return {ok:details.length>0&&!extractionFailed,error:extractionFailed?'리뷰 버튼을 상품 ID와 연결하지 못했습니다. 상품 목록으로 진단 텍스트를 대신 사용하거나 미로딩이라고 단정하지 마세요.':undefined,actionable_item_count:items.length,items,navigation,action_result:actionResult,frames:details.map(({items,navigation,...f})=>f),
   scope:'현재 Chrome의 렌더링된 DOM을 읽었습니다. 빈 items는 미로딩의 증거가 아닙니다. diagnostics를 확인하고 navigation 또는 scroll을 사용하세요. 전체 구매내역 수집 완료를 의미하지 않습니다.'};
 }
 if(cmd.action==='resolve')return {ok:true,items:await Promise.all(a.ids.map(getTarget))};
 if(cmd.action==='open'){
  const t=await getTarget(a.target_id);if(t.kind!=='item')throw Error('상품 id를 지정하세요.');
  const form=await openForm(t,check);
  return {ok:true,status:'editor_open',product:t.product,window_id:form.windowId,window_type:form.windowType,tab_id:form.tabId,frame_id:form.frameId,form_id:form.id,context:form.context,observation:form.observation};
 }
 if(cmd.action==='fill'){
  if(a.reviews?.length!==1)throw Error('입력은 상품 한 개씩 처리합니다.');
  const draft=a.reviews[0],t=await getTarget(draft.id);
  if(t.kind!=='item'||t.product!==draft.product)throw Error('상품 대상 불일치');
  const form=await openForm(t,check);check();
  return fillPopup(form,draft,check);
 }
 if(['read_current','submit_current'].includes(cmd.action)){
  const t=await getTarget(a.target_id),form=openedForms.get(t.id);
  if(t.kind!=='item'||!form)throw Error('먼저 이 상품의 리뷰창을 open 또는 fill로 열어야 합니다.');
  const message={form_id:form.id,product:t.product};
  if(cmd.action==='read_current')return send(form.tabId,form.frameId,{...message,action:'read_current'});
  if(!a.snapshot||a.snapshot.product!==t.product||a.snapshot.form_id!==form.id)throw Error('확인한 입력창과 현재 대상이 다릅니다.');
  const {receipts={}}=await chrome.storage.local.get('receipts');
  const digest=await crypto.subtle.digest('SHA-256',new TextEncoder().encode(t.url+'|'+t.summary));
  const key='item:'+Array.from(new Uint8Array(digest),v=>v.toString(16).padStart(2,'0')).join('');
  if(receipts[key])return {ok:false,status:'already_attempted',error:'이미 등록을 시도한 상품입니다. 구매내역에서 결과를 확인하세요.'};
  check();receipts[key]={status:'uncertain',product:t.product};await chrome.storage.local.set({receipts});
  const result=await submitPopup(form,a.snapshot,check);
  if(result?.attempted_submit===false)delete receipts[key];else receipts[key]=result||{status:'uncertain'};
  await chrome.storage.local.set({receipts});return result;
 }
 if(cmd.action!=='submit')throw Error('지원하지 않는 작업');
 const {receipts={}}=await chrome.storage.local.get('receipts');
 // Persist before any mutation. A retry after interruption must not submit twice.
 if(receipts[cmd.id])return receipts[cmd.id];
 receipts[cmd.id]={ok:false,status:'uncertain',error:'이 요청은 이미 시작됐습니다. 구매후기 등록 여부를 직접 확인하세요.'};
 await chrome.storage.local.set({receipts});
 const results=[];
 for(const draft of a.reviews){
  let result;
  try{
   const t=await getTarget(draft.id);
   if(t.product!==draft.product)throw Error('승인한 상품명과 현재 상품이 다릅니다.');
   const digest=await crypto.subtle.digest('SHA-256',new TextEncoder().encode(t.url+'|'+t.summary));
   const key='item:'+Array.from(new Uint8Array(digest),v=>v.toString(16).padStart(2,'0')).join('');
   if(receipts[key])throw Error('이미 등록을 시도한 상품입니다. 중복 등록 방지를 위해 구매내역에서 확인하세요.');
   const form=await openForm(t,check),tabId=form.tabId;
   await fillPopup(form,draft,check);
   check();receipts[key]={status:'uncertain',product:t.product,at:Date.now()};await chrome.storage.local.set({receipts});
   result=await submitPopup(form,draft,check);
   if(!result)result={ok:false,status:'uncertain',error:'등록 응답을 확인하지 못했습니다.'};
   if(result.attempted_submit===false)delete receipts[key];else receipts[key]=result;
  }catch(e){result={ok:false,error:e.message,observation:e.observation};}
  results.push({id:draft.id,...result});
  // Stop on an error instead of continuing with potentially changed navigation.
  if(!result.ok)break;
 }
 const result={ok:results.length===a.reviews.length&&results.every(r=>r.ok),results,not_attempted:a.reviews.slice(results.length).map(v=>v.id)};
 receipts[cmd.id]=result;await chrome.storage.local.set({receipts});return result;
}
