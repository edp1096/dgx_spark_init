const { chromium } = require('../../web/node_modules/playwright');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({executablePath:'/usr/bin/google-chrome',headless:true,args:['--no-sandbox']});
 try{
 const page=await browser.newPage();
 await page.goto('http://localhost:8585');
 await page.setContent(`<main><article><a href="https://smartstore.naver.com/shop/products/123">테스트 머그컵</a><button id="open">리뷰 작성</button></article><form hidden id="review"><h2>테스트 머그컵 리뷰 작성</h2><label><input type="radio" name="rating" value="5" aria-label="5점">5점</label><label><input type="radio" name="unrelated" value="5">옵션</label><textarea placeholder="리뷰 내용"></textarea><button type="button" id="submit">리뷰 등록</button></form><div role="status" id="notice"></div></main>`);
 await page.evaluate(()=>{
  window.chrome={runtime:{id:'test',onMessage:{addListener:f=>window.listener=f}}};
  window.post=(m)=>new Promise(r=>window.listener({type:'TALK_REVIEW_V13',...m},{id:'test'},r));
  window.submissions=0;
  document.querySelector('#open').onclick=()=>document.querySelector('#review').hidden=false;
  document.querySelector('#submit').onclick=()=>{window.submissions++;document.querySelector('#notice').textContent='리뷰 등록이 완료되었습니다';};
 });
 await page.addScriptTag({content:fs.readFileSync(path.join(__dirname,'../../internal/browserbridge/extension/naver-content.js'),'utf8')});
 const list=await page.evaluate(()=>window.post({action:'inspect'}));assert.equal(list.items.length,1);assert.equal(list.items[0].product,'테스트 머그컵');
 const open=await page.evaluate(({id,url})=>window.post({action:'open',id,url}),{id:list.items[0].id,url:list.url});assert.equal(open.ok,true);
 const after=await page.evaluate(()=>window.post({action:'inspect'}));assert.equal(after.forms.length,1);
 const bad=await page.evaluate(id=>window.post({action:'submit',form_id:id,product:'다른 상품',text:'사용 후기',rating:5}),after.forms[0].id);assert.equal(bad.ok,false);
 const filled=await page.evaluate(id=>window.post({action:'fill',form_id:id,product:'테스트 머그컵',text:'제출 전 실제 입력 확인',rating:5}),after.forms[0].id);
 assert.equal(filled.status,'filled');assert.equal(filled.submitted,false);
 assert.equal(await page.locator('textarea').inputValue(),'제출 전 실제 입력 확인');
 assert.equal(await page.evaluate(()=>window.submissions),0);
 await page.locator('textarea').fill('사용자가 직접 수정한 내용');
 const snapshot=await page.evaluate(id=>window.post({action:'read_current',form_id:id,product:'테스트 머그컵'}),after.forms[0].id);
 assert.equal(snapshot.text,'사용자가 직접 수정한 내용');assert.equal(snapshot.rating,5);
 await page.locator('textarea').fill('확인 후 다시 수정한 최종 내용');
 const changed=await page.evaluate(s=>window.post({...s,action:'submit_current'}),snapshot);
 assert.equal(changed.ok,false);assert.equal(changed.attempted_submit,false);
 assert.equal(await page.locator('textarea').inputValue(),'확인 후 다시 수정한 최종 내용');
 assert.equal(await page.evaluate(()=>window.submissions),0);
 const latest=await page.evaluate(id=>window.post({action:'read_current',form_id:id,product:'테스트 머그컵'}),after.forms[0].id);
 const ok=await page.evaluate(s=>window.post({...s,action:'submit_current'}),latest);assert.equal(ok.status,'submitted',JSON.stringify(ok));
 assert.equal(await page.locator('textarea').inputValue(),'확인 후 다시 수정한 최종 내용');
 console.log('PASS: manual edits preserved, stale snapshot blocked, current form submitted without refilling');
 assert.equal(await page.evaluate(()=>window.submissions),1);
 assert.equal(await page.locator('input[name="unrelated"]').isChecked(),false);
 await page.locator('#review').evaluate(el=>el.remove());
 const stale=await page.evaluate(id=>window.post({action:'submit',form_id:id,product:'테스트 머그컵',text:'중복 시도',rating:5}),after.forms[0].id);assert.equal(stale.ok,false);
 // All action labels observed in the actual Talk trace, not just a synthetic exact label.
 const labels=['리뷰쓰고 최대 150원 받기','한달사용리뷰 쓰기','한달사용리뷰 쓰고 최대 10원 받기','리뷰쓰기 최대 150원','리뷰작성 680원','리뷰 작성'];
 await page.setContent('<nav><button>리뷰 작성</button><button>작성 가능한 리뷰</button></nav>'+labels.map((label,i)=>`<article><span class="ProductInfo_name__test">테스트 상품 ${i}</span><button>${label}</button></article>`).join(''));
 const variants=await page.evaluate(()=>window.post({action:'inspect'}));
 assert.equal(variants.items.length,labels.length,JSON.stringify(variants));
 assert.equal(variants.diagnostics.review_button_count,labels.length);assert.equal(variants.diagnostics.unmatched.length,0);
 console.log('PASS: all six observed label variants match products; navigation is excluded');
 // Real logged button labels must survive point-reward suffixes.
 await page.setContent(`<button id="nav">상품리뷰26</button><section id="list" style="height:180px;overflow-y:auto"><article><span class="Store_name__x">테스트 판매점</span><span class="ProductInfo_name__abc">이미 표시된 상품</span><button>리뷰쓰기 최대 150원</button></article><div style="height:800px"></div></section>`);
 await page.evaluate(()=>{
  document.querySelector('#nav').onclick=()=>{document.querySelector('#nav').textContent='작성 가능한 리뷰';};
  document.querySelector('#list').addEventListener('scroll',()=>setTimeout(()=>{
   if(document.querySelector('#late'))return;
   const a=document.createElement('article');a.id='late';a.innerHTML='<span class="ProductInfo_name__def">추가 로딩 상품</span><button>리뷰작성 680원</button>';document.querySelector('#list').append(a);
  },450),{once:true});
 });
 const ready=await page.evaluate(()=>window.post({action:'inspect'}));assert.equal(ready.items.length,1);assert.equal(ready.status,'items_found');assert.equal(ready.navigation.length,1);
 const nav=await page.evaluate(id=>window.post({action:'navigate',id}),ready.navigation[0].id);assert.equal(nav.ok,true);
 const scrolled=await page.evaluate(()=>window.post({action:'scroll',steps:1}));assert.equal(scrolled.items.length,2,JSON.stringify(scrolled));assert.equal(scrolled.scroll.added_items,1);assert.ok(scrolled.scroll.steps[0].after>0);assert.equal(scrolled.scroll.complete,false);
 await page.setContent('<article><span>추출할 이름 속성이 없는 상품</span><button>리뷰쓰기 최대 150원</button></article>');
 const unmatched=await page.evaluate(()=>window.post({action:'inspect'}));assert.equal(unmatched.status,'extraction_failed');assert.equal(unmatched.diagnostics.loading_observed,false);assert.equal(unmatched.diagnostics.review_button_count,1);assert.ok(unmatched.diagnostics.unmatched[0].ancestors.length);
 console.log('PASS: rendered items, reward suffixes, navigation, nested scroll, delayed items, extraction failure distinguished from loading');
 console.log('PASS: item binding, form identity, exact text, rating isolation, completion evidence, stale form rejection');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
