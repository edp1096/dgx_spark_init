// DOM input handling adapted from the user-uploaded naver-review-extension-mvp.
(() => {
 if(globalThis.__TALK_REVIEW_BRIDGE_V8__)return;
 globalThis.__TALK_REVIEW_BRIDGE_V8__=true;
 const clean=s=>String(s||'').replace(/\s+/g,' ').trim();
 const visible=el=>{const r=el.getBoundingClientRect(),s=getComputedStyle(el);return r.width>0&&r.height>0&&s.display!=='none'&&s.visibility!=='hidden';};
 const text=el=>clean(el.innerText||el.getAttribute('aria-label')||el.value||el.textContent);
 const clicks=root=>[...root.querySelectorAll('button,a,[role="button"],[role="tab"],input[type="submit"]')].filter(e=>visible(e)&&!e.disabled&&e.getAttribute('aria-disabled')!=='true');
 // Identify the review action, independently of the reward copy after it.
 const reviewButton=el=>{
  if(el.closest('nav,[role="navigation"],[class*="MyLNB_"]'))return false;
  const label=text(el).replace(/\s+/g,'');
  return label.length<=100 && /^(?:한달사용리뷰(?:쓰기|쓰고)?|(?:상품)?(?:리뷰|후기)(?:작성(?:하기)?|쓰기|쓰고))/.test(label)
   && !/삭제|수정|신고|취소|등록완료/.test(label);
 };
 const navButton=el=>/^(?:상품리뷰|작성가능한리뷰|내가작성한리뷰|리뷰작성|주문내역|구매내역)(?:[\d,]+|\([\d,]+\))?$/.test(text(el).replace(/\s+/g,''));
 const navigation=new Map();
 const clickReports=new Map();
 const items=new Map(),forms=new Map();
 const elementIDs=new WeakMap();
 const idFor=el=>{if(!elementIDs.has(el))elementIDs.set(el,crypto.randomUUID());return elementIDs.get(el);};
 function productIn(root){
  // Prefer product-name elements and actual product links over card-wide text.
  const selectors=[
   '[data-product-name],[class*="product" i][class*="name" i],[class*="product" i][class*="title" i],[class*="product_name" i],[class*="productName" i],[class*="product_title" i],[class*="productTitle" i]',
   'a[href*="/products/"],a[href*="/product/"],a[href*="/window-products/"]',
   '[class*="name" i],[class*="title" i],h3,h4'
  ];
  for(const selector of selectors){
   const nodes=[...root.querySelectorAll(selector)].filter(visible);
   const names=[...new Set(nodes.map(el=>clean(el.getAttribute('data-product-name')||text(el))).filter(t=>t.length>=2&&t.length<=300&&!/리뷰|후기|배송|구매확정|판매자|문의|^[\d,.\s]+원$/.test(t)))];
   if(names.length===1)return names[0];
   if(names.length>1)return '';
  }
  // Image alt is used only when the card supplies a single unambiguous name.
  const names=[...new Set([...root.querySelectorAll('img[alt]')].filter(visible).map(el=>clean(el.alt)).filter(t=>t.length>3&&!/^(상품이미지|상품사진|썸네일|아이콘|로고)$/.test(t)))];
  return names.length===1?names[0]:'';
 }
 function findItem(button){
  const month=/한달사용/.test(text(button));
  for(let root=button.parentElement,depth=0;root&&root!==document.body&&depth<12;root=root.parentElement,depth++){
   // One normal and one month-later review button may belong to the same item.
   if(clicks(root).filter(el=>reviewButton(el)&&/한달사용/.test(text(el))===month).length>1)break;
   const product=productIn(root);if(product)return {root,product};
  }
  return null;
 }
 function editorDiagnostics(){
  return [...document.querySelectorAll('textarea,[contenteditable="true"]')].filter(visible).slice(0,12).map(editor=>{
   const ancestors=[];for(let el=editor.parentElement,n=0;el&&n<5;el=el.parentElement,n++){
    ancestors.push({tag:el.tagName,role:el.getAttribute('role'),classes:String(el.className).slice(0,160),text:text(el).slice(0,600)});
    if(el===document.body)break;
   }
   return {tag:editor.tagName,id:editor.id,placeholder:editor.getAttribute('placeholder'),aria:editor.getAttribute('aria-label'),classes:String(editor.className).slice(0,160),disabled:!!editor.disabled,readonly:!!editor.readOnly,ancestors};
  });
 }
 function getForms(product=''){
  const out=[];
  for(const editor of document.querySelectorAll('textarea,[contenteditable="true"]')){
   if(!visible(editor)||editor.disabled||editor.readOnly)continue;
   // A nested textarea wrapper is not necessarily the review form. Walk up to
   // the region containing the editor, review controls and the target product.
   for(let root=editor.parentElement;root;root=root.parentElement){
    if(!visible(root))continue;
    if(clicks(root).some(reviewButton))break;
    if([...root.querySelectorAll('textarea,[contenteditable="true"]')].filter(visible).length!==1)break;
    const context=text(root);
    const semantic=root.matches('form,[role="dialog"]');
    const submitControl=clicks(root).some(e=>/^(리뷰|후기|상품평)\s*등록$|^작성\s*완료$|^등록$/.test(text(e)));
    const ratingControl=root.querySelector('input[type="radio"],[role="radio"],[aria-label*="점"],[aria-label*="별"]');
    if(/리뷰|후기|상품평/.test(context)&&(semantic||(submitControl&&ratingControl))&&(!product||context.includes(product))){
     const id=idFor(editor);forms.set(id,{root,editor});
     out.push({id,context:context.slice(0,2500),product_matched:!!product&&context.includes(product)});break;
    }
    if(root===document.body)break;
   }
  }
  return out;
 }
 function inspect(product=''){
  const found=[],unmatched=[],nav=[];
  const buttons=clicks(document);
  const reviewButtons=buttons.filter(el=>reviewButton(el)&&(!navButton(el)||findItem(el)));
  for(const button of reviewButtons){
   const v=findItem(button);
   if(!v){
    const ancestors=[];for(let el=button.parentElement,n=0;el&&el!==document.body&&n<6;el=el.parentElement,n++){if(clicks(el).filter(reviewButton).length>2)break;ancestors.push({tag:el.tagName,classes:String(el.className).slice(0,200),text:text(el).slice(0,1000),elements:[...el.querySelectorAll('a,span,strong,p,img')].slice(0,30).map(x=>({tag:x.tagName,classes:String(x.className).slice(0,160),text:text(x).slice(0,250),alt:x.getAttribute('alt')}))});}
    unmatched.push({label:text(button),ancestors});continue;
   }
   const id=idFor(button);items.set(id,{...v,button,url:location.href});
   found.push({id,product:v.product,review_kind:/한달사용/.test(text(button))?'month':'initial',summary:text(v.root).slice(0,500)});
  }
  for(const el of buttons.filter(navButton)){
   // A card's review button opens an editor, not the review-list navigation.
   if(reviewButton(el)&&findItem(el))continue;
   const id=idFor(el);navigation.set(id,{el,label:text(el),url:location.href});
   nav.push({id,label:text(el)});
  }
  const loading=[...document.querySelectorAll('[aria-busy="true"],[role="progressbar"]')].some(visible);
  return {ok:true,version:8,url:location.href,title:document.title,items:found,navigation:nav,forms:getForms(product),editors:editorDiagnostics(),buttons:buttons.filter(e=>/리뷰|후기/.test(text(e))).slice(0,50).map(text),
   status:found.length?'items_found':reviewButtons.length?'extraction_failed':loading?'loading_observed':'no_review_items_detected',
   diagnostics:{review_button_count:reviewButtons.length,unmatched:unmatched.slice(0,8),loading_observed:loading,ready_state:document.readyState},
   guidance:'Only items with id are actionable review candidates. Diagnostic fragments are not a complete product list; never claim that only the first few products mentioned there have loaded. This is the live rendered DOM in the user Chrome, not an HTML fetch. Empty items do NOT prove lazy loading. Use navigation targets or scroll; extraction_failed means visible review buttons were not matched to product names.'};
 }
 async function scrollPage(steps){
  const snapshots=[inspect()],movement=[];
  for(let i=0;i<steps;i++){
   const anchor=[...items.values()].find(v=>v.button.isConnected)?.button||clicks(document).find(reviewButton);
   let scroller=document.scrollingElement;
   for(let el=anchor?.parentElement;el&&el!==document.body;el=el.parentElement){if(/auto|scroll/.test(getComputedStyle(el).overflowY)&&el.scrollHeight>el.clientHeight+10){scroller=el;break;}}
   const before=scroller.scrollTop;
   let lastChange=Date.now(),changes=0;
   const observer=new MutationObserver(()=>{lastChange=Date.now();changes++;});
   observer.observe(document.body,{childList:true,subtree:true,characterData:true});
   try{
    scroller.scrollBy({top:Math.max(300,scroller.clientHeight*0.8),behavior:'instant'});
    const started=Date.now();
    while(Date.now()-started<4000){await new Promise(r=>setTimeout(r,150));if(Date.now()-started>=1800&&Date.now()-lastChange>=600&&!document.querySelector('[aria-busy="true"]'))break;}
   }finally{observer.disconnect();}
   movement.push({before,after:scroller.scrollTop,dom_changes:changes});snapshots.push(inspect());
   if(scroller.scrollTop===before&&changes===0)break;
  }
  // Retain items that a virtualized list removed from the DOM; they are data,
  // not an assertion that their old element remains actionable.
  const all=new Map();for(const snapshot of snapshots)for(const item of snapshot.items)all.set(item.id,item);
  return {...snapshots.at(-1),items:[...all.values()],scroll:{steps:movement,added_items:[...all.keys()].filter(id=>!snapshots[0].items.some(v=>v.id===id)).length,complete:false},
   guidance:'Scrolled the existing client DOM and waited for mutations. complete=false: reaching a quiet viewport does not prove that the whole purchase history was loaded. A virtualized item may need to be brought back into view before submitting.'};
 }
  function nativeSetValue(el, value) {
    if (el instanceof HTMLTextAreaElement || el instanceof HTMLInputElement) {
      const proto = el instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
      const setter = Object.getOwnPropertyDescriptor(proto, "value")?.set;
      if (setter) setter.call(el, value);
      else el.value = value;
      el.dispatchEvent(new InputEvent("input", { bubbles: true, inputType: "insertText", data: value }));
      el.dispatchEvent(new Event("change", { bubbles: true }));
      return;
    }
    if (el.isContentEditable) {
      el.focus();
      el.textContent = value;
      el.dispatchEvent(new InputEvent("input", { bubbles: true, inputType: "insertText", data: value }));
      el.dispatchEvent(new Event("change", { bubbles: true }));
    }
  }


 const wait=ms=>new Promise(r=>setTimeout(r,ms));
 function readCurrent(a){
  const f=forms.get(a.form_id);
  if(!f||!f.editor.isConnected||!visible(f.editor)||!text(f.root).includes(a.product))throw Error('현재 리뷰 입력창을 확인할 수 없습니다.');
  const selected=[...f.root.querySelectorAll('input[type="radio"],button,[role="radio"]')].filter(el=>el.checked||el.getAttribute('aria-checked')==='true'||el.getAttribute('aria-pressed')==='true');
  const values=selected.map(el=>clean(el.getAttribute('aria-label')||el.labels?.[0]?.textContent||text(el)).match(/^(?:(?:별점|평점)\s*)?([1-5])\s*점(?:\s*선택)?$/)).filter(Boolean);
  if(values.length!==1)throw Error('현재 선택한 별점을 하나로 확인할 수 없습니다.');
  return {ok:true,status:'current_form',product:a.product,text:f.editor.value??f.editor.textContent,rating:Number(values[0][1]),form_id:a.form_id};
 }
 const pendingSubmissions=new Map();
 function currentForm(a){
  const f=forms.get(a.form_id);
  if(!f||!f.editor.isConnected||!visible(f.editor)||!text(f.root).includes(a.product))throw Error('대상 팝업의 리뷰 입력창이 변경됐습니다.');
  return f;
 }
 function inputPoint(el){
  el.scrollIntoView({block:'center',behavior:'instant'});
  const r=el.getBoundingClientRect(),left=Math.max(0,r.left),right=Math.min(innerWidth,r.right),top=Math.max(0,r.top),bottom=Math.min(innerHeight,r.bottom);
  if(right<=left||bottom<=top)throw Error('클릭 요소가 화면 안에 없습니다.');
  const x=(left+right)/2,y=(top+bottom)/2,hit=document.elementFromPoint(x,y);
  if(hit!==el&&!el.contains(hit))throw Error('다른 요소가 클릭 대상을 가리고 있습니다.');
  const id=idFor(el);clickReports.set(id,{trusted_event:null});
  el.addEventListener('click',event=>clickReports.set(id,{trusted_event:event.isTrusted,user_activation:navigator.userActivation.isActive}),{once:true,capture:true});
  return {ok:true,point:{x,y},click_id:id,label:text(el)};
 }
 function prepareRating(a){
  const {root}=currentForm(a);
  if(!Number.isInteger(a.rating)||a.rating<1||a.rating>5)throw Error('별점은 1..5 정수여야 합니다.');
  const candidates=[...root.querySelectorAll('input[type="radio"],button,[role="radio"]')].filter(el=>{
   const label=clean(el.getAttribute('aria-label')||el.labels?.[0]?.textContent||text(el));
   return new RegExp('^(?:별점|평점)?\\s*'+a.rating+'\\s*점(?:\\s*선택)?$').test(label)&&(visible(el)||[...(el.labels||[])].some(visible));
  });
  if(candidates.length!==1)throw Error('팝업의 별점 대상을 하나로 확인하지 못했습니다.');
  const candidate=candidates[0],target=visible(candidate)?candidate:[...candidate.labels].find(visible);
  return inputPoint(target);
 }
 function prepareText(a){
  const {editor}=currentForm(a);editor.scrollIntoView({block:'center',behavior:'instant'});editor.focus();
  if(editor instanceof HTMLTextAreaElement)editor.select();
  else {const range=document.createRange();range.selectNodeContents(editor);const selection=getSelection();selection.removeAllRanges();selection.addRange(range);}
  if(document.activeElement!==editor&&!editor.contains(document.activeElement))throw Error('입력칸 포커스를 확인하지 못했습니다.');
  return {ok:true};
 }
 function verifySnapshot(a){
  const current=readCurrent(a);
  if(current.text!==a.text||current.rating!==a.rating)throw Error('확인 후 내용 또는 별점이 달라졌습니다. 덮어쓰거나 등록하지 않았습니다.');
  return current;
 }
 function prepareSubmission(a){
  verifySnapshot(a);const {root}=currentForm(a);
  const buttons=clicks(root).filter(e=>/^(리뷰|후기|상품평)\s*등록$|^작성\s*완료$|^등록$/.test(text(e)));
  if(buttons.length!==1)throw Error('팝업의 등록 버튼을 하나로 확인하지 못했습니다.');
  const point=inputPoint(buttons[0]),receipt=crypto.randomUUID();
  let finish;
  const promise=new Promise(resolve=>{finish=resolve;});
  const state={promise,observer:null,timer:null};
  const complete=result=>{clearTimeout(state.timer);state.observer?.disconnect();finish(result);};
  const selector='[role="alert"],[role="status"],[role="dialog"],.toast';
  state.observer=new MutationObserver(records=>{
   const changed=new Set();
   for(const r of records){const el=r.target.nodeType===1?r.target:r.target.parentElement;const parent=el?.closest(selector);if(parent)changed.add(parent);for(const n of r.addedNodes||[])if(n.nodeType===1){if(n.matches(selector))changed.add(n);for(const v of n.querySelectorAll(selector))changed.add(v);}}
   const notice=[...changed].find(el=>visible(el)&&/(리뷰|후기|상품평).{0,20}(등록|작성).{0,10}(완료|되었습니다|됐습니다)/.test(text(el)));
   if(notice)complete({ok:true,status:'submitted',evidence:text(notice).slice(0,500)});
  });
  state.observer.observe(document.body,{childList:true,subtree:true,characterData:true});
  state.timer=setTimeout(()=>complete({ok:false,status:'uncertain',attempted_submit:true,error:'등록 입력 후 완료 표시를 확인하지 못했습니다. 자동 재시도하지 마세요.'}),10000);
  pendingSubmissions.set(receipt,state);setTimeout(()=>pendingSubmissions.delete(receipt),60000);
  return {...point,receipt};
 }
 async function submit(a){
  let clicked=false,observer;
  try {
  const f=forms.get(a.form_id);
  if(!f||!f.editor.isConnected||!visible(f.editor)||!text(f.root).includes(a.product))throw Error('상품 입력창이 변경됐습니다.');
  if(typeof a.text!=='string'||a.text.trim().length<1||a.text.length>10000||!Number.isInteger(a.rating)||a.rating<1||a.rating>5)throw Error('리뷰 내용 또는 별점이 올바르지 않습니다.');
  const {root,editor}=f;
  if(a.action==='submit_current'){
   const current=readCurrent(a);
   if(current.text!==a.text||current.rating!==a.rating)throw Error('확인한 뒤 입력 내용이나 별점이 바뀌었습니다. 덮어쓰거나 등록하지 않았습니다.');
  }else{
  // Match an explicit rating label within this review form; never arbitrary radios.
  const ratingCandidates=[...root.querySelectorAll('input[type="radio"],button,[role="radio"]')].filter(el=>{
   const label=clean(el.getAttribute('aria-label')||el.labels?.[0]?.textContent||text(el));
   return new RegExp('^(별점\\s*)?'+a.rating+'\\s*점(\\s*선택)?$').test(label)&&(visible(el)||[...(el.labels||[])].some(visible));
  });
  if(ratingCandidates.length!==1)throw Error('별점 대상을 하나로 확인하지 못했습니다.');
  const rating=ratingCandidates[0];rating.click();
  editor.focus();nativeSetValue(editor,a.text);await wait(300);
  if(clean(editor.value??editor.textContent)!==clean(a.text))throw Error('입력한 리뷰 내용 검증 실패');
  if(!(rating.checked||rating.getAttribute('aria-checked')==='true'||rating.getAttribute('aria-pressed')==='true'))throw Error('선택된 별점을 확인하지 못했습니다.');
  }
  if(a.action==='fill')return {ok:true,status:'filled',submitted:false,product:a.product,text:editor.value??editor.textContent,rating:a.rating};
  const submitters=clicks(root).filter(e=>/^(리뷰|후기|상품평)\s*등록$|^작성\s*완료$|^등록$/.test(text(e)));
  if(submitters.length!==1)throw Error('리뷰 등록 버튼을 하나로 확인하지 못했습니다.');
  const noticeSelector='[role="alert"],[role="status"],[role="dialog"],.toast';
  const changed=new Set();
  observer=new MutationObserver(records=>{
   for(const r of records){
    const el=r.target.nodeType===1?r.target:r.target.parentElement;
    const owner=el?.closest(noticeSelector);if(owner)changed.add(owner);
    for(const n of r.addedNodes||[])if(n.nodeType===1){if(n.matches(noticeSelector))changed.add(n);for(const v of n.querySelectorAll(noticeSelector))changed.add(v);}
   }
  });
  observer.observe(document.body,{childList:true,subtree:true,characterData:true});
  clicked=true;submitters[0].click();
  for(let i=0;i<30;i++){
   await wait(300);
   const notice=[...changed].find(el=>el.isConnected&&visible(el)&&/(리뷰|후기|상품평).{0,20}(등록|작성).{0,10}(완료|되었습니다|됐습니다)/.test(text(el)));
   if(notice)return {ok:true,status:'submitted',evidence:text(notice).slice(0,500)};
  }
  return {ok:false,status:'uncertain',attempted_submit:true,error:'등록 버튼은 눌렀지만 완료 메시지를 확인하지 못했습니다. 구매내역에서 확인하고 자동 재시도하지 마세요.'};
  } catch(e) {return {ok:false,status:clicked?'uncertain':'blocked',attempted_submit:clicked,error:e.message};} finally {observer?.disconnect();}
 }
 chrome.runtime.onMessage.addListener((m,s,reply)=>{
  if(s.id!==chrome.runtime.id||m?.type!=='TALK_REVIEW_V8')return;
  (async()=>{
   if(m.action==='inspect')return inspect(m.product||'');
   if(m.action==='scroll')return scrollPage(Math.max(1,Math.min(4,Number(m.steps)||2)));
   if(m.action==='navigate'){
    const v=navigation.get(m.id);
    if(!v||v.url!==location.href||!v.el.isConnected||!visible(v.el)||text(v.el)!==v.label||!navButton(v.el))throw Error('메뉴 대상이 변경됐습니다. 다시 조회하세요.');
    if(v.el.tagName==='A'){
     const u=new URL(v.el.href,location.href);
     if(u.protocol!=='https:'||!['naver.com','naverpay.com'].some(h=>u.hostname===h||u.hostname.endsWith('.'+h)))throw Error('허용되지 않은 이동 주소');
    }
    v.el.click();return {ok:true,clicked:v.label};
   }
   if(m.action==='click_report')return {ok:true,...clickReports.get(m.id)};
   if(m.action==='open'||m.action==='prepare_open'){
    const v=items.get(m.id);
    if(!v||v.url!==location.href||m.url!==location.href||!v.button.isConnected||!visible(v.button)||!reviewButton(v.button)||productIn(v.root)!==v.product)throw Error('구매상품 대상이 바뀌었습니다. 다시 조회하세요.');
    v.button.scrollIntoView({block:'center',behavior:'instant'});
    if(m.action==='prepare_open'){
     const box=v.button.getBoundingClientRect();
     const left=Math.max(0,box.left),right=Math.min(innerWidth,box.right),top=Math.max(0,box.top),bottom=Math.min(innerHeight,box.bottom);
     if(right<=left||bottom<=top)throw Error('리뷰 버튼이 화면 안에 보이지 않습니다.');
     const x=(left+right)/2,y=(top+bottom)/2,hit=document.elementFromPoint(x,y);
     if(hit!==v.button&&!v.button.contains(hit))throw Error('다른 요소가 리뷰 버튼을 가리고 있습니다.');
     clickReports.set(m.id,{trusted_event:null});
     v.button.addEventListener('click',event=>clickReports.set(m.id,{trusted_event:event.isTrusted,user_activation:navigator.userActivation.isActive}),{once:true,capture:true});
     return {ok:true,label:text(v.button),point:{x,y}};
    }
    const href=v.button.tagName==='A'?v.button.href:'';
    if(href&&v.button.target==='_blank'){
     const u=new URL(href,location.href);
     if(u.protocol==='https:'&&['naver.com','naverpay.com'].some(h=>u.hostname===h||u.hostname.endsWith('.'+h)))return {ok:true,open_url:u.href,label:text(v.button),click_dispatched:false};
    }
    v.button.click();return {ok:true,click_dispatched:true,label:text(v.button),note:'DOM click dispatched; this does not prove that a popup opened.'};
   }
   if(m.action==='prepare_rating')return prepareRating(m);
   if(m.action==='prepare_text')return prepareText(m);
   if(m.action==='verify_snapshot')return verifySnapshot(m);
   if(m.action==='prepare_submit')return prepareSubmission(m);
   if(m.action==='observe_submit'){
    const state=pendingSubmissions.get(m.receipt);if(!state)throw Error('등록 관찰 기록이 없습니다. 자동 재시도하지 마세요.');
    const result=await state.promise;pendingSubmissions.delete(m.receipt);return result;
   }
   if(m.action==='read_current')return readCurrent(m);
   if(['submit','fill','submit_current'].includes(m.action))return submit(m);
   throw Error('지원하지 않는 작업');
  })().then(reply,e=>reply({ok:false,error:e.message}));return true;
 });
})();
