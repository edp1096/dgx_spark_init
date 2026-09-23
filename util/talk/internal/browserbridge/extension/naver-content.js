// DOM input handling adapted from the user-uploaded naver-review-extension-mvp.
(() => {
 if(globalThis.__TALK_REVIEW_BRIDGE_V14__)return;
 globalThis.__TALK_REVIEW_BRIDGE_V14__=true;
 const clean=s=>String(s||'').replace(/\s+/g,' ').trim();
 const visible=el=>{const r=el.getBoundingClientRect(),s=getComputedStyle(el);return r.width>0&&r.height>0&&s.display!=='none'&&s.visibility!=='hidden';};
 const operable=el=>visible(el)||[...(el.labels||[])].some(visible);
 const text=el=>clean(el.innerText||el.getAttribute('aria-label')||el.value||el.textContent);
 const clicks=root=>[...root.querySelectorAll('button,a,[role="button"],[role="tab"],input[type="submit"]')].filter(e=>visible(e)&&!e.disabled&&e.getAttribute('aria-disabled')!=='true');
 // Identify the review action, independently of the reward copy after it.
 const reviewButton=el=>{
  if(el.closest('nav,[role="navigation"],[class*="MyLNB_"]'))return false;
  const area=el.getAttribute('data-shp-area')||el.getAttribute('data-nlog-area');
  if(['rvw.pntop','rvw.ntcop'].includes(area)||el.closest('[class*="productBox_guide_point"],[class*="reviewPointGuide_"]'))return false;
  const label=text(el).replace(/\s+/g,'');
  if(/^(?:한달사용)?(?:상품)?(?:리뷰|후기)작성시/.test(label))return false;
  return label.length<=100 && /^(?:한달사용리뷰(?:쓰기|쓰고)?|(?:상품)?(?:리뷰|후기)(?:작성(?:하기)?|쓰기|쓰고))/.test(label)
   && !/삭제|수정|신고|취소|등록완료/.test(label);
 };
 const navButton=el=>/^(?:상품리뷰|작성가능한리뷰|내가작성한리뷰|리뷰작성|주문내역|구매내역)(?:[\d,]+|\([\d,]+\))?$/.test(text(el).replace(/\s+/g,''));
 const navigation=new Map();
 const clickReports=new Map();
 const items=new Map(),forms=new Map();
 const elementIDs=new WeakMap();
 const idFor=el=>{if(!elementIDs.has(el))elementIDs.set(el,crypto.randomUUID());return elementIDs.get(el);};
 const norm=s=>clean(s).toLowerCase().replace(/[\s\[\](){}<>|·ㆍ,:;_'\"`~!@#$%^&*+=?\\/.-]+/g,'');
 const productMatches=(context,product)=>{
  if(!product)return true;
  const a=norm(context),b=norm(product);if(!a||!b)return false;
  if(a.includes(b)||b.includes(a))return true;
  const tokens=clean(product).split(/[\s,()[\]{}|/]+/).map(norm).filter(x=>x.length>=2);
  if(!tokens.length)return false;
  const hits=tokens.filter(x=>a.includes(x)).length;
  return hits>=Math.min(3,tokens.length)&&hits/tokens.length>=0.6;
 };
 function ratingValue(el){
  const direct=el.getAttribute('data-value')||el.value;
  if(/^[1-5]$/.test(String(direct||'')))return Number(direct);
  const label=clean(el.getAttribute('aria-label')||el.labels?.[0]?.textContent||text(el));
  const m=label.match(/(?:별점|평점)?\s*([1-5])\s*점?/);return m?Number(m[1]):null;
 }
 // Numeric data-value is NOT enough to identify the 1..5 star rating.
 // Naver's product-specific follow-up questions may also use numeric values
 // (for example 1/2/3) on <a role="radio"> controls.  Identify the star
 // rating by its telemetry/group context instead of by each option alone.
 function isStarRatingControl(el){
  if(!el)return false;
  if(/^(?:별점|평점)?\s*[1-5]\s*점(?:\s*선택)?$/.test(clean(el.getAttribute("aria-label")||el.labels?.[0]?.textContent)))return true;
  const area=el.getAttribute('data-shp-area')||el.getAttribute('data-nlog-area')||'';
  const kind=el.getAttribute('data-shp-contents-type')||'';
  if(area==='rvw.rate'||/리뷰별점|별점 클릭/.test(kind))return true;
  const group=el.closest('[role="radiogroup"]');
  if(!group)return false;
  const controls=[...group.querySelectorAll('input[type="radio"],[role="radio"]')].filter(operable);
  if(controls.length!==5)return false;
  const values=controls.map(ratingValue);
  return values.every(Number.isInteger)&&new Set(values).size===5&&values.every(v=>v>=1&&v<=5);
 }
 function selectedRating(root){
  const selected=[...root.querySelectorAll('input[type="radio"],[role="radio"]')]
   .filter(isStarRatingControl)
   .filter(el=>el.checked||el.getAttribute('aria-checked')==='true'||el.getAttribute('aria-pressed')==='true')
   .map(ratingValue).filter(Number.isInteger);
  return selected.length===1?selected[0]:null;
 }
 function questionTitle(container){
  const candidates=[...container.querySelectorAll('legend,strong,h3,h4,[class*="title" i],[class*="question" i]')]
   .filter(visible).map(text).filter(t=>t&&t.length<=300&&!/상품 리뷰 작성|리뷰쓰기/.test(t));
  return candidates[0]||'';
 }
 function questionSchema(root){
  const seen=new Set(),out=[];
  const groups=[...root.querySelectorAll('fieldset,[role="radiogroup"],[role="group"],[class*="wrapBox_wrap_box"], [class*="question" i]')];
  for(const group of groups){
   if(!visible(group)||seen.has(group))continue;
   const controls=[...group.querySelectorAll('input[type="radio"],input[type="checkbox"],button[role="radio"],button[role="checkbox"],[role="radio"],[role="checkbox"],select')].filter(operable);
   if(!controls.length)continue;
   const ratingLike=controls.length>=5&&controls.every(isStarRatingControl);
   if(ratingLike)continue;
   let owner=group;
   if(group.getAttribute('role')==='radiogroup'||group.getAttribute('role')==='group'){
    for(let el=group.parentElement,n=0;el&&n<4;el=el.parentElement,n++){
     if(questionTitle(el)){owner=el;break;}
    }
   }
   if(seen.has(owner))continue;seen.add(owner);
   const prompt=questionTitle(owner)||questionTitle(group);if(!prompt)continue;
   const qid='q:'+idFor(owner),options=[];
   for(const el of [...owner.querySelectorAll('input[type="radio"],input[type="checkbox"],button[role="radio"],button[role="checkbox"],[role="radio"],[role="checkbox"]')].filter(operable)){
    if(isStarRatingControl(el))continue;
    const label=clean(el.getAttribute('aria-label')||el.labels?.[0]?.textContent||text(el)||el.value);
    if(!label)continue;
    const optionId='o:'+idFor(el);options.push({id:optionId,label,value:el.getAttribute('data-value')||el.value||label,selected:!!el.checked||el.getAttribute('aria-checked')==='true'||el.getAttribute('aria-pressed')==='true'});
   }
   const select=owner.querySelector('select');
   if(select){for(const o of [...select.options])if(o.value)options.push({id:'o:'+idFor(o),label:clean(o.textContent),value:o.value,selected:o.selected});}
   if(options.length){const required=!!owner.querySelector('[required],[aria-required="true"]')||/필수/.test(prompt);out.push({id:qid,prompt,type:owner.querySelector('input[type="checkbox"],[role="checkbox"]')?'multi':'single',required,options});}
  }
  return out;
 }
 function resolveQuestion(root,qid){
  const raw=String(qid||'').replace(/^q:/,'');
  for(const el of root.querySelectorAll('*'))if(idFor(el)===raw)return el;
  return null;
 }
 function resolveOption(root,optionId){
  const raw=String(optionId||'').replace(/^o:/,'');
  for(const el of root.querySelectorAll('input[type="radio"],input[type="checkbox"],button,[role="radio"],[role="checkbox"],option'))if(idFor(el)===raw)return el;
  return null;
 }
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
   for(let root=editor.parentElement;root;root=root.parentElement){
    if(!visible(root))continue;
    if(clicks(root).some(button=>{if(!reviewButton(button))return false;const item=findItem(button);return item&&!item.root.contains(editor);}))break;
    if([...root.querySelectorAll('textarea,[contenteditable="true"]')].filter(visible).length!==1)break;
    const context=text(root), semantic=root.matches('form,[role="dialog"]');
    // IMPORTANT: Naver's review submit button is disabled until rating/text are valid.
    // Do not use clicks(root) here because clicks() intentionally filters disabled controls.
    // A disabled "등록" button is still strong evidence that this DOM subtree is the review form.
    const submitControl=[...root.querySelectorAll('button,[role="button"],input[type="submit"]')]
      .some(e=>visible(e)&&/^(리뷰|후기|상품평)\s*등록$|^작성\s*완료$|^등록$/.test(text(e)));
    const ratingControl=[...root.querySelectorAll('input[type="radio"],[role="radio"]')].find(isStarRatingControl)||root.querySelector('[aria-label*="별점"],[aria-label*="평점"]');
    if(/리뷰|후기|상품평/.test(context)&&(semantic||(submitControl&&ratingControl))){
     const matched=productMatches(context,product);
     if(product&&!matched&&root.parentElement&&productMatches(text(root.parentElement),product)&&!clicks(root.parentElement).some(reviewButton))continue;
     const id=idFor(editor),questions=questionSchema(root);
     forms.set(id,{root,editor,questions});
     const submit=[...root.querySelectorAll('button,[role="button"],input[type="submit"]')].find(e=>visible(e)&&/^(리뷰|후기|상품평)\s*등록$|^작성\s*완료$|^등록$/.test(text(e)));
     out.push({id,draft_resumed:document.documentElement.hasAttribute("data-sparktalk-resumed-draft"),context:context.slice(0,2500),product_matched:!!product&&matched,questions,rating_values:[...root.querySelectorAll('[role="radio"],input[type="radio"]')].filter(isStarRatingControl).map(ratingValue).filter(Number.isInteger),submit_disabled:!!submit?.disabled});break;
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
  return {ok:true,version:14,url:location.href,title:document.title,items:found,navigation:nav,forms:getForms(product),editors:editorDiagnostics(),buttons:buttons.filter(e=>/리뷰|후기/.test(text(e))).slice(0,50).map(text),
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
    globalThis.__talkDOM.scroll(scroller,Math.max(300,scroller.clientHeight*0.8));
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



 function readCurrent(a){
  const f=forms.get(a.form_id);
  if(!f||!f.editor.isConnected||!visible(f.editor)||!productMatches(text(f.root),a.product))throw Error('현재 리뷰 입력창을 확인할 수 없습니다.');
  const rating=selectedRating(f.root);if(!Number.isInteger(rating))throw Error('현재 선택한 별점을 하나로 확인할 수 없습니다.');
  const answers=questionSchema(f.root).map(q=>({question_id:q.id,selected:q.options.filter(o=>o.selected).map(o=>o.id)}));
  return {ok:true,status:'current_form',product:a.product,text:f.editor.value??f.editor.textContent,rating,answers,form_id:a.form_id};
 }

 const pendingSubmissions=new Map();
 function currentForm(a){
  const f=forms.get(a.form_id);
  if(!f||!f.editor.isConnected||!visible(f.editor)||!productMatches(text(f.root),a.product))throw Error('대상 팝업의 리뷰 입력창이 변경됐습니다.');
  return f;
 }
 function inputPoint(el){
  const {x,y}=globalThis.__talkDOM.point(el);
  const id=idFor(el);clickReports.set(id,{trusted_event:null});
  el.addEventListener('click',event=>clickReports.set(id,{trusted_event:event.isTrusted,user_activation:navigator.userActivation.isActive}),{once:true,capture:true});
  return {ok:true,point:{x,y},click_id:id,label:text(el)};
 }
 function prepareResumeDraft(){
  const candidates=[...document.querySelectorAll('[role="dialog"],[role="alertdialog"],[class*="confirm"][class*="box"]')].filter(visible).filter(el=>{
   const message=text(el);return /리뷰|작성\s*중|임시\s*저장/.test(message)&&/이어서\s*(?:작성|쓰)|계속\s*작성|작성.*계속/.test(message);
  });
  const dialogs=candidates.filter(el=>!candidates.some(other=>other!==el&&el.contains(other)));
  if(!dialogs.length)return {ok:true,none:true};
  if(dialogs.length!==1)throw Error('이어쓰기 확인창을 하나로 확인하지 못했습니다.');
  const buttons=clicks(dialogs[0]).filter(el=>/^(확인|예|이어서\s*작성(?:하기)?|계속\s*작성(?:하기)?)$/.test(text(el)));
  if(buttons.length!==1)throw Error('이어쓰기 확인 버튼을 하나로 확인하지 못했습니다.');
  buttons[0].addEventListener('click',()=>document.documentElement.setAttribute('data-sparktalk-resumed-draft','true'),{once:true});
  return inputPoint(buttons[0]);
 }
 function prepareRating(a){
  const {root}=currentForm(a);
  if(!Number.isInteger(a.rating)||a.rating<1||a.rating>5)throw Error('별점은 1..5 정수여야 합니다.');
  const candidates=[...root.querySelectorAll('input[type="radio"],[role="radio"]')].filter(isStarRatingControl).filter(el=>ratingValue(el)===a.rating&&(visible(el)||[...(el.labels||[])].some(visible)));
  if(candidates.length!==1)throw Error('팝업의 별점 '+a.rating+'점 대상을 하나로 확인하지 못했습니다.');
  const candidate=candidates[0],target=visible(candidate)?candidate:[...candidate.labels].find(visible);
  return inputPoint(target);
 }
 function prepareAnswer(a){
  const {root}=currentForm(a),question=resolveQuestion(root,a.question_id);if(!question)throw Error('추가 질문을 찾지 못했습니다.');
  const option=resolveOption(question,a.option_id);if(!option)throw Error('추가 질문 선택지를 찾지 못했습니다.');
  if(option.checked||option.selected||option.getAttribute('aria-checked')==='true')return {ok:true,already_selected:true};
  const target=option.tagName==='OPTION'?option.parentElement:(visible(option)?option:[...(option.labels||[])].find(visible));
  if(!target)throw Error('추가 질문 선택지가 보이지 않습니다.');
  if(target.tagName==='SELECT')return {ok:true,select:true,question_id:a.question_id,option_id:a.option_id,value:option.value};
  return {...inputPoint(target),question_id:a.question_id,option_id:a.option_id};
 }
 function setSelectAnswer(a){
  const {root}=currentForm(a),question=resolveQuestion(root,a.question_id);if(!question)throw Error('추가 질문을 찾지 못했습니다.');
  const option=resolveOption(question,a.option_id);if(!option||option.tagName!=='OPTION')throw Error('선택 옵션을 찾지 못했습니다.');
  return {ok:globalThis.__talkDOM.select(option.parentElement,option.value)};
 }

 function prepareText(a){
  const {editor}=currentForm(a);globalThis.__talkDOM.focus(editor,true);
  return {ok:true};
 }
 function verifySnapshot(a){
  const current=readCurrent(a);
  if(current.text!==a.text||current.rating!==a.rating)throw Error('확인 후 내용 또는 별점이 달라졌습니다. 덮어쓰거나 등록하지 않았습니다.');
  if(Array.isArray(a.answers)){for(const answer of a.answers){
   const q=current.answers.find(v=>v.question_id===answer.question_id);
   const expected=answer.selected||answer.option_ids||(answer.option_id?[answer.option_id]:[]);
   if(!q||expected.some(id=>!q.selected.includes(id))||(Array.isArray(answer.selected)&&q.selected.length!==expected.length))throw Error('추가 질문 응답이 변경되었거나 적용되지 않았습니다.');
  }}
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
 chrome.runtime.onMessage.addListener((m,s,reply)=>{
  if(s.id!==chrome.runtime.id||m?.type!=='TALK_REVIEW_V14')return;
  (async()=>{
   if(m.action==='inspect')return inspect(m.product||'');
   if(m.action==='scroll')return scrollPage(Math.max(1,Math.min(4,Number(m.steps)||2)));
   if(m.action==='prepare_navigation'){
    const v=navigation.get(m.id);
    if(!v||v.url!==location.href||!v.el.isConnected||!visible(v.el)||text(v.el)!==v.label||!navButton(v.el))throw Error('메뉴 대상이 변경됐습니다. 다시 조회하세요.');
    if(v.el.tagName==='A'){
     const u=new URL(v.el.href,location.href);
     if(u.protocol!=='https:'||!['naver.com','naverpay.com'].some(h=>u.hostname===h||u.hostname.endsWith('.'+h)))throw Error('허용되지 않은 이동 주소');
    }
    return inputPoint(v.el);
   }
   if(m.action==='click_report')return {ok:true,...clickReports.get(m.id)};
   if(m.action==='prepare_open'){
    const v=items.get(m.id);
    if(!v||v.url!==location.href||m.url!==location.href||!v.button.isConnected||!visible(v.button)||!reviewButton(v.button)||productIn(v.root)!==v.product)throw Error('구매상품 대상이 바뀌었습니다. 다시 조회하세요.');
    return inputPoint(v.button);
   }
   if(m.action==='close_state')return {ok:true,dirty:[...document.querySelectorAll('textarea,[contenteditable="true"]')].filter(visible).some(el=>String(el.value??el.textContent).trim())};
   if(m.action==='prepare_resume')return prepareResumeDraft();
   if(m.action==='prepare_rating')return prepareRating(m);
   if(m.action==='prepare_answer')return prepareAnswer(m);
   if(m.action==='set_select_answer')return setSelectAnswer(m);
   if(m.action==='prepare_text')return prepareText(m);
   if(m.action==='verify_snapshot')return verifySnapshot(m);
   if(m.action==='prepare_submit')return prepareSubmission(m);
   if(m.action==='observe_submit'){
    const state=pendingSubmissions.get(m.receipt);if(!state)throw Error('등록 관찰 기록이 없습니다. 자동 재시도하지 마세요.');
    const result=await state.promise;pendingSubmissions.delete(m.receipt);return result;
   }
   if(m.action==='read_current')return readCurrent(m);
   throw Error('지원하지 않는 작업');
  })().then(reply,e=>reply({ok:false,error:e.message}));return true;
 });
})();
