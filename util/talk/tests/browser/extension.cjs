const {chromium}=require('../../web/node_modules/playwright');
const fs=require('node:fs'),os=require('node:os'),path=require('node:path');
(async()=>{
 const mode=process.env.BROWSER_CASE||'inline';
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'talk-extension-'));
 fs.cpSync(path.resolve(__dirname,'../../internal/browserbridge/extension'),path.join(dir,'extension'),{recursive:true});
 const manifestPath=path.join(dir,'extension/manifest.json'),manifest=JSON.parse(fs.readFileSync(manifestPath));
 manifest.host_permissions.push('http://127.0.0.1/*');fs.writeFileSync(manifestPath,JSON.stringify(manifest));
 const ext=path.join(dir,'extension');
 const context=await chromium.launchPersistentContext(path.join(dir,'profile'),{executablePath:'/home/edp1096/.cache/ms-playwright/chromium-1243/chrome-linux-arm64/chrome',headless:true,args:['--no-sandbox','--ignore-certificate-errors','--host-resolver-rules=MAP shopping.naver.com 127.0.0.1',`--disable-extensions-except=${ext}`,`--load-extension=${ext}`]});
 try{
 context.on('dialog',()=>{});
 context.on('console',m=>console.error('chrome:',m.text()));
 const worker=context.serviceWorkers()[0]||await context.waitForEvent('serviceworker');
 await worker.evaluate(async ({server,token})=>{await chrome.storage.local.set({server,token});},{server:process.env.BRIDGE_URL,token:process.env.BRIDGE_TOKEN});
 const popup=await context.newPage();await popup.goto(new URL('connect.html',worker.url()).href);await popup.evaluate(()=>chrome.runtime.sendMessage({type:'CONNECT'}));console.error('extension configured');
 await context.route('https://shopping.naver.com/**',route=>route.fulfill({contentType:'text/html',body:`<!doctype html><meta charset="utf-8"><article><a href="https://smartstore.naver.com/example/products/123">테스트 머그컵</a><button id="open">리뷰쓰고 최대 150원 받기</button></article><article><a href="https://smartstore.naver.com/example/products/456">테스트 수건</a><button id="open2">한달사용리뷰 쓰고 최대 10원 받기</button></article><form id="review" hidden><h2>테스트 머그컵 리뷰 작성</h2><input type="radio" aria-label="5점"><textarea></textarea><button type="button" id="submit">리뷰 등록</button></form><div role="status" id="notice"></div><script>window.submissions=0;document.querySelector('#open').onclick=()=>{document.querySelector('h2').textContent='테스트 머그컵 리뷰 작성';document.querySelector('#review').hidden=false;};document.querySelector('#open2').onclick=()=>{document.querySelector('h2').textContent='테스트 수건 리뷰 작성';document.querySelector('#review').hidden=false;};document.querySelector('#submit').onclick=()=>{window.submissions++;document.querySelector('#notice').textContent='리뷰 등록이 완료되었습니다';};</script>`}));
 const page=await context.newPage();await page.goto('https://shopping.naver.com/test-purchases');
 if(mode==='trusted'){
  await page.evaluate(()=>{const button=document.querySelector('#open'),original=button.onclick;button.onclick=event=>{if(event.isTrusted&&navigator.userActivation.isActive)original(event);};button.click();if(!document.querySelector('#review').hidden)throw Error('Synthetic click unexpectedly opened editor');});
 }
 if(mode==='popup'){
  await page.evaluate(url=>{const a=document.createElement('a');a.id='open';a.href=url;a.target='_blank';a.rel='noopener';a.textContent='리뷰쓰고 최대 150원 받기';document.querySelector('#open').replaceWith(a);},'https://shopping.naver.com:'+new URL(process.env.FIXTURE_URL).port+'/popup/reviews/form');
 }
 if(['popup_window','popup_reuse','resume_source','resume_popup','resume_dom'].includes(mode)){
  const writer='https://shopping.naver.com:'+new URL(process.env.FIXTURE_URL).port+(mode==='resume_popup'?'/popup/reviews/monthly-form':'/popup/reviews/form')+(mode==='resume_popup'?'?resume=1':mode==='resume_dom'?'?resume_dom=1':'');
  if(mode==='popup_reuse'){
   const popupEvent=context.waitForEvent('page');
   await page.evaluate(url=>{window.reviewPopup=window.open(url,'reviewEditor','popup=yes,width=520,height=680');},writer);
   const existing=await popupEvent;await existing.waitForLoadState();
   await existing.evaluate(()=>document.querySelector('h2').textContent='이전 상품 리뷰 작성');
  }
  await page.evaluate(({url,mode})=>{document.querySelector('#open').onclick=()=>{if(mode==='resume_source'&&!confirm('작성 중이던 리뷰가 있습니다. 이어서 작성하시겠습니까?'))return;window.reviewPopup=window.open(url,'reviewEditor','popup=yes,width=520,height=680');};},{url:writer,mode});
 }
 if(mode==='idle'||mode==='unmatched'){
  await page.evaluate(mode=>{document.querySelector('#open').onclick=()=>{if(mode==='unmatched'){document.querySelector('#review').hidden=false;document.querySelector('h2').textContent='다른 상품 리뷰 작성';}};},mode);
  const old=await context.newPage();await context.route('https://shopping.naver.com/old-detail',r=>r.fulfill({contentType:'text/html',body:'<title>Existing review detail</title>'}));await old.goto('https://shopping.naver.com/old-detail');
 }
 console.log('READY');console.error('fixture ready');
 let capturedSubmissions=null;
 const lines=require('node:readline').createInterface({input:process.stdin});
 for await(const line of lines){
  if(line==='close_ready'){
   capturedSubmissions=0;for(const p of context.pages().filter(p=>p.url().startsWith('https://shopping.naver.com')))capturedSubmissions+=await p.evaluate(()=>window.submissions||0);
   await page.locator('#review').evaluate(el=>el.hidden=true);console.log('CLOSE_READY');continue;
  }
  if(line==='edit'){
   const writer=context.pages().find(p=>p.url().includes('/popup/reviews/form'));
   if(!writer)throw Error('popup writer missing');
   await writer.locator('textarea').fill('사용자가 팝업에서 고친 최종 후기');
   await writer.locator('[aria-label="4점"]').check();
   console.log('EDITED');continue;
  }
  if(line==='finish')break;
 }
 lines.close();
 let submissions=0;for(const tab of context.pages().filter(p=>p.url().startsWith('https://shopping.naver.com')))submissions+=await tab.evaluate(()=>window.submissions||0);if(capturedSubmissions!==null)submissions=capturedSubmissions;const expected=['idle','unmatched'].includes(mode)?0:2;if(submissions!==expected)throw Error('Unexpected submissions '+submissions);
 console.log('PASS: Chrome extension, websocket, item discovery, sequential submission, persistent duplicate guard');
 }finally{await context.close();fs.rmSync(dir,{recursive:true,force:true});}
})().catch(e=>{console.error(e);process.exitCode=1;});
