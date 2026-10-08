const {chromium}=require('../../web/node_modules/playwright');
const fs=require('node:fs'),os=require('node:os'),path=require('node:path'),assert=require('node:assert/strict');
(async()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'talk-popup-'));
 const ext=path.resolve(__dirname,'../../internal/browserbridge/extension');
 const context=await chromium.launchPersistentContext(dir,{executablePath:'/home/edp1096/.cache/ms-playwright/chromium-1243/chrome-linux-arm64/chrome',headless:true,args:['--no-sandbox',`--disable-extensions-except=${ext}`,`--load-extension=${ext}`]});
 try{
  const worker=context.serviceWorkers()[0]||await context.waitForEvent('serviceworker');
  const popupURL=new URL('connect.html',worker.url()).href;
  let popup=await context.newPage();await popup.goto(popupURL);
  await popup.locator('#url').fill('http://192.168.100.61:8585');await popup.close();
  popup=await context.newPage();await popup.goto(popupURL);
  await popup.waitForFunction(()=>document.querySelector('#url').value==='http://192.168.100.61:8585');
  await popup.locator('#token').fill('b'.repeat(64));await popup.close();
  popup=await context.newPage();await popup.goto(popupURL);
  await popup.waitForFunction(()=>document.querySelector('#token').value==='b'.repeat(64));
  assert.equal(await popup.locator('#url').inputValue(),'http://192.168.100.61:8585');
  const active=await popup.evaluate(()=>chrome.storage.local.get(['server','token']));assert.deepEqual(active,{});
  await popup.locator('#token').fill('');await popup.close();
  popup=await context.newPage();await popup.goto(popupURL);
  const stored=await popup.evaluate(()=>chrome.storage.local.get('connectionDraftToken'));assert.equal(stored.connectionDraftToken,'');
  console.log('PASS: address and key survive separate popup closes; drafts do not change active connection; clearing is persisted');
 }finally{await context.close();fs.rmSync(dir,{recursive:true,force:true});}
})().catch(e=>{console.error(e);process.exitCode=1;});
