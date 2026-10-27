// Inspect the model's real browser layout, including captions and asset loading.
import puppeteer from 'puppeteer-core';
import sharp from 'sharp';
import {createServer} from 'node:http';
import {readFile,writeFile} from 'node:fs/promises';
import {resolve,extname,sep} from 'node:path';

const folder=resolve(process.argv[2]);
const spec=JSON.parse(await readFile(resolve(folder,'composition-spec.json'),'utf8'));
const server=createServer(async(req,res)=>{
  try{
    const pathname=decodeURIComponent(new URL(req.url,'http://localhost').pathname);
    if(pathname==='/favicon.ico'){res.writeHead(204).end();return;}
    const path=resolve(folder,'.'+(pathname==='/'?'/index.html':pathname));
    if(!path.startsWith(folder+sep)){res.writeHead(403).end();return;}
    const mime={'.html':'text/html','.js':'text/javascript','.json':'application/json','.png':'image/png','.jpg':'image/jpeg','.svg':'image/svg+xml','.mp4':'video/mp4','.wav':'audio/wav','.woff2':'font/woff2'}[extname(path)]||'application/octet-stream';
    res.setHeader('Content-Type',mime);res.end(await readFile(path));
  }catch{res.writeHead(404).end();}
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
const origin=`http://127.0.0.1:${server.address().port}`;
let browser;
const issues=[],samples=[],previews=[],assetPixels=[];
try{
  browser=await puppeteer.launch({executablePath:process.env.HYPERFRAMES_BROWSER_PATH||'/usr/bin/chromium',headless:true,args:['--no-sandbox','--disable-gpu','--disable-dev-shm-usage']});
  const page=await browser.newPage();await page.setViewport({width:720,height:1280,deviceScaleFactor:1});
  await page.setRequestInterception(true);
  page.on('request',r=>{if(r.url().startsWith(origin+'/')||/^(data|blob):/.test(r.url()))r.continue();else{issues.push({code:'external_request',url:r.url()});r.abort();}});
  page.on('pageerror',e=>issues.push({code:'javascript_error',message:e.message}));
  page.on('response',r=>{if(r.status()>=400)issues.push({code:'missing_asset',url:r.url(),status:r.status()});});
  await page.goto(origin,{waitUntil:'networkidle0',timeout:30000});
  await page.evaluate(async()=>{await document.fonts.ready;await Promise.all(window.__lottieReady||[]);});
  const contract=await page.evaluate(()=>{
    const spec=window.__shorts_spec;
    return {duration:window.__timelines?.shorts?.duration(),voices:spec.scenes.map((s,i)=>{const e=document.getElementById('voice-'+i);return {index:i,exists:!!e,src:e?.getAttribute('src'),start:Number(e?.dataset.start),duration:Number(e?.dataset.duration),volume:Number(e?.dataset.volume)}})};
  });
  if(!Number.isFinite(contract.duration)||Math.abs(contract.duration-spec.duration)>.05)issues.push({code:'timeline_duration',actual:contract.duration,expected:spec.duration});
  for(const voice of contract.voices){const scene=spec.scenes[voice.index];if(scene.voice_duration>0&&(!voice.exists||voice.src!==`voice-${voice.index}.wav`||Math.abs(voice.start-scene.voice_start)>.01||Math.abs(voice.duration-scene.voice_duration)>.01||voice.volume!==1))issues.push({code:'voice_contract',voice});}
  const headlineEnd=spec.headline_rules?.mode==='opening'?Math.min(spec.duration,spec.headline_rules.duration_seconds):spec.duration;
  const openingMoment=spec.headline_rules&&spec.headline_rules.mode!=='none'?Number(Math.min(.8,headlineEnd*.4).toFixed(3)):null;
  const headlineTimes=openingMoment===null?[]:[openingMoment,Number((headlineEnd*.6).toFixed(3)),Number(Math.max(.05,headlineEnd-Math.min(.5,headlineEnd*.2)-.05).toFixed(3)),Number(Math.min(spec.duration-.05,headlineEnd+.2).toFixed(3))];
  const sampleTimes=[...new Set([...headlineTimes,...spec.scenes.map(s=>Number((s.start+s.duration*.5).toFixed(3))),...spec.captions.map(c=>Number((c.start+c.duration*.5).toFixed(3)))])].sort((a,b)=>a-b);
  let previewIndex=0;
  for(const time of sampleTimes){
    const result=await page.evaluate(async(time)=>{
      const spec=window.__shorts_spec,tl=window.__timelines.shorts;
      for(const el of document.querySelectorAll('[data-start]')){
        const start=Number(el.dataset.start),length=Number(el.dataset.duration);el.style.visibility=time>=start&&time<start+length?'visible':'hidden';
        if(el.tagName==='VIDEO'&&time>=start&&time<start+length&&Number.isFinite(el.duration))el.currentTime=Math.min(el.duration-.04,Math.max(0,time-start));
      }
      tl.seek(time,false);
      for(const animation of window.__hfLottie||[]){if(animation.isLoaded)animation.goToAndStop((time*animation.frameRate)%animation.totalFrames,true);}
      await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));
      const findings=[],styles=[];let headline=null;
      for(const caption of spec.captions){
        if(time<caption.start||time>=caption.start+caption.duration)continue;
        const el=document.getElementById(caption.id);
        if(!el){findings.push({code:'caption_missing',id:caption.id});continue;}
        const box=el.getBoundingClientRect(),css=getComputedStyle(el),rules=spec.caption_rules;
        const rows=[...el.querySelectorAll(':scope > .line')],textCSS=getComputedStyle(rows[0]||el);
        const fontSize=Math.min(...(rows.length?rows:[el]).map(row=>parseFloat(getComputedStyle(row).fontSize)));
        const rowCount=rows.reduce((sum,row)=>{const style=getComputedStyle(row);return sum+Math.max(1,Math.round(row.scrollHeight/(parseFloat(style.lineHeight)||parseFloat(style.fontSize)*1.2)));},0);
        if(css.display==='none'||css.visibility==='hidden'||Number(css.opacity)<.05||box.width<20||box.height<20)findings.push({code:'caption_hidden',id:caption.id,opacity:css.opacity,visibility:css.visibility});
        if(Math.abs(time-(caption.start+caption.duration*.5))<.01&&Number(css.opacity)<.8)findings.push({code:'caption_faded_at_midpoint',id:caption.id,opacity:css.opacity});
        if(box.left<720*rules.safe_x-3||box.right>720*(1-rules.safe_x)+3||box.top<1280*rules.safe_top-3||box.bottom>1280*rules.safe_bottom+3)findings.push({code:'caption_outside_safe_area',id:caption.id,bounds:{x:box.x,y:box.y,w:box.width,h:box.height}});
        if(fontSize<rules.min_font_px)findings.push({code:'caption_too_small',id:caption.id,font:fontSize});
        if(rowCount>rules.max_lines||rows.some(row=>row.scrollWidth>row.clientWidth+2))findings.push({code:'caption_overflow',id:caption.id,lines:rowCount});
        if(el.textContent.replace(/\s/g,'')!==caption.text.replace(/\s/g,''))findings.push({code:'caption_text_changed',id:caption.id});
        styles.push({id:caption.id,font_family:textCSS.fontFamily,font_size:fontSize,color:textCSS.color,background:css.backgroundColor,background_before:getComputedStyle(el,':before').backgroundColor,emphasis_styles:[...el.querySelectorAll('em')].map(e=>({text:e.textContent,color:getComputedStyle(e).color,font_weight:getComputedStyle(e).fontWeight})),x:box.x,y:box.y,width:box.width,height:box.height,lines:rowCount,highlights:el.querySelectorAll('em').length});
      }
      if(spec.headline_rules){
        const rules=spec.headline_rules,el=document.getElementById('project-title'),css=el&&getComputedStyle(el),box=el?.getBoundingClientRect();
        const end=rules.mode==='opening'?Math.min(spec.duration,rules.duration_seconds):spec.duration;
        const expected=rules.mode!=='none'&&time<end;
        const visible=el&&css.display!=='none'&&css.visibility!=='hidden'&&Number(css.opacity)>.05&&box.width>0&&box.height>0;
        // A short opacity-only exit is valid inside an opening title's final
        // 20% (at most .5 s). Dedicated steady-state samples still require it.
        const exitFade=expected&&rules.mode==='opening'&&time>=end-Math.min(.5,end*.2)&&el&&css.display!=='none'&&css.visibility!=='hidden'&&Number(css.opacity)<=.05&&box.width>0&&box.height>0;
        if(expected&&!visible&&!exitFade)findings.push({code:'headline_hidden',time});
        if(!expected&&visible)findings.push({code:'headline_outside_interval',time,end});
        if(expected&&visible){
          const rows=[...el.querySelectorAll('.headline-line')],elements=rows.length?rows:[el];
          const textCSS=getComputedStyle(elements[0]);
          const font=Math.min(...elements.map(e=>parseFloat(getComputedStyle(e).fontSize)));
          const lines=elements.reduce((n,e)=>{const s=getComputedStyle(e);return n+Math.max(1,Math.round(e.scrollHeight/(parseFloat(s.lineHeight)||parseFloat(s.fontSize)*1.2)));},0);
          if(font<rules.min_font_px)findings.push({code:'headline_too_small',font,minimum:rules.min_font_px});
          if(lines>rules.max_lines||elements.some(e=>e.scrollWidth>e.clientWidth+2))findings.push({code:'headline_overflow',lines});
          if(box.left<720*rules.safe_left-3||box.right>720*rules.safe_right+3||box.top<1280*rules.safe_top-3||box.bottom>1280*rules.safe_bottom+3)findings.push({code:'headline_outside_safe_area',bounds:{x:box.x,y:box.y,w:box.width,h:box.height}});
          if(el.textContent.replace(/\s/g,'')!==(rules.text||spec.title).replace(/\s/g,''))findings.push({code:'headline_text_changed'});
          for(const c of styles){if(Math.min(box.right,c.x+c.width)-Math.max(box.left,c.x)>4&&Math.min(box.bottom,c.y+c.height)-Math.max(box.top,c.y)>4)findings.push({code:'headline_caption_overlap',caption:c.id,time});}
          headline={text:el.textContent,font_family:textCSS.fontFamily,font_size:font,font_weight:textCSS.fontWeight,color:textCSS.color,background:css.backgroundColor,lines,x:box.x,y:box.y,width:box.width,height:box.height,opacity:css.opacity};
        }
      }
      for(let i=0;i<spec.scenes.length;i++){
        const scene=spec.scenes[i];if(time<scene.start||time>=scene.start+scene.duration)continue;
        const el=document.getElementById('visual-'+i),box=el?.getBoundingClientRect(),css=el&&getComputedStyle(el);
        if(scene.kind!=='typography'&&(!el||css.display==='none'||Number(css.opacity)<.1||box.width<120||box.height<120))findings.push({code:'visual_not_visible',scene:i});
        if(el?.tagName==='IMG'&&(!el.complete||el.naturalWidth===0))findings.push({code:'image_not_loaded',scene:i});
      }
      for(let i=0;i<spec.assets.length;i++){
        const asset=spec.assets[i];if(asset.scene_asset)continue;
        const el=document.getElementById('decoration-'+i),box=el?.getBoundingClientRect(),css=el&&getComputedStyle(el);
        if(!el)findings.push({code:'decoration_missing',asset_id:asset.id});
        else if(asset.kind==='lottie'&&!el.querySelector('svg path,svg rect,svg ellipse,svg image'))findings.push({code:'lottie_empty',asset_id:asset.id});
      }
      return {time,findings,styles,headline};
    },time);
    issues.push(...result.findings);samples.push(result);
    const sceneMoment=time===openingMoment||spec.scenes.some(s=>Math.abs(time-(s.start+s.duration*.5))<.002);
    if(sceneMoment){
      const name=`preview-${previewIndex++}.png`;
      const screenshot=await page.screenshot({path:resolve(folder,name)});previews.push({file:name,time});
      const baseline=await sharp(screenshot).removeAlpha().raw().toBuffer();
      for(let i=0;i<spec.assets.length;i++){
        const asset=spec.assets[i];
        const ids=asset.scene_asset?spec.scenes.flatMap((s,j)=>s.asset_id===asset.id&&time>=s.start&&time<s.start+s.duration?['visual-'+j]:[]):['decoration-'+i];
        for(const id of ids){
          const saved=await page.evaluate(id=>{const el=document.getElementById(id);if(!el)return null;const previous=el.style.getPropertyValue('display'),priority=el.style.getPropertyPriority('display');el.style.setProperty('display','none','important');return {previous,priority};},id);
          if(saved===null)continue;
          const hidden=await sharp(await page.screenshot()).removeAlpha().raw().toBuffer();
          await page.evaluate(({id,saved})=>{const el=document.getElementById(id);if(saved.previous)el.style.setProperty('display',saved.previous,saved.priority);else el.style.removeProperty('display');},{id,saved});
          let changed=0;for(let p=0;p<baseline.length;p+=3){if(Math.abs(baseline[p]-hidden[p])+Math.abs(baseline[p+1]-hidden[p+1])+Math.abs(baseline[p+2]-hidden[p+2])>30)changed++;}
          assetPixels.push({asset_id:asset.id,time,changed_pixels:changed});

        }
      }
    }
  }
  for(const asset of spec.assets){if(!assetPixels.some(sample=>sample.asset_id===asset.id&&sample.changed_pixels>=24))issues.push({code:'decoration_not_visible',asset_id:asset.id,message:'Selected asset has no visible pixels in any scene preview. Ensure it appears on screen with contrast; dark Lottie lines with screen blend can disappear.'});}
  if(previews.length)await writeFile(resolve(folder,'preview.png'),await readFile(resolve(folder,previews[0].file)));
  const report={passed:issues.length===0,issues,samples,previews,asset_pixels:assetPixels,external_network_allowed:false};
  await writeFile(resolve(folder,'audit.json'),JSON.stringify(report,null,2));console.log(JSON.stringify({passed:report.passed,issues,previews}));
  if(issues.length)process.exitCode=2;
}catch(error){
  const report={passed:false,issues:[...issues,{code:'preview_failed',message:error.message}],samples,previews};await writeFile(resolve(folder,'audit.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report));process.exitCode=2;
}finally{if(browser)await browser.close();server.closeAllConnections();await new Promise(resolve=>server.close(resolve));}
