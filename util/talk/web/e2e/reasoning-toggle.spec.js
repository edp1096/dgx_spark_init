import { test, expect } from '@playwright/test';
import { createServer } from 'node:http';

test('reasoning toggle survives streamed chunks and repeated clicks', async ({page,request}) => {
 let send,finish;const gate=new Promise(resolve=>finish=resolve);
 const backend=createServer(async(req,res)=>{
  if(req.method!=='POST'){res.end(JSON.stringify({data:[{id:'test-model'}]}));return;}
  let raw='';for await(const chunk of req)raw+=chunk;
  if(!JSON.parse(raw).stream){res.end(JSON.stringify({choices:[{message:{content:'Reasoning test'}}]}));return;}
  res.setHeader('Content-Type','text/event-stream');
  send=text=>res.write('data: '+JSON.stringify({choices:[{delta:{reasoning_content:text}}]})+'\n\n');
  send('첫 번째 생각입니다.\n');await gate;
  res.end('data: '+JSON.stringify({choices:[{delta:{content:'완료'}}]})+'\n\ndata: [DONE]\n\n');
 });
 await new Promise(resolve=>backend.listen(0,'127.0.0.1',resolve));
 const original=await(await request.get('/api/config')).json();let session;
 try {
  const config=structuredClone(original);config.model.endpoint=`http://127.0.0.1:${backend.address().port}`;
  await request.put('/api/config',{data:config});session=await(await request.post('/api/sessions',{data:{title:'Reasoning toggle'}})).json();
  await page.goto('/');await page.locator('.composer textarea').fill('생각과정 테스트');await page.locator('.composer textarea').press('Enter');
  const toggle=page.locator('.reasoning-toggle');await expect(toggle).toBeVisible();
  for(let i=0;i<4;i++){
   await toggle.hover();await page.mouse.down();send(`추가 생각 ${i}.\n`);await page.waitForTimeout(80);await page.mouse.up();
   await expect(toggle).toHaveAttribute('aria-expanded',i%2?'false':'true');
  }
  await toggle.click();send('마지막 생각.');await expect(page.locator('.reasoning-text')).toBeVisible();
  finish();await expect(page.locator('.bubble')).toContainText(['생각과정 테스트','완료']);
  await expect(toggle).toHaveAttribute('aria-expanded','true');
  await page.getByRole('button',{name:'↑ 생각 과정 접기',exact:true}).click();await expect(toggle).toHaveAttribute('aria-expanded','false');
  await toggle.click();await expect(page.locator('.reasoning-text')).toBeVisible();
 }finally{
  finish();if(session)await request.delete(`/api/sessions/${session.id}`);await request.put('/api/config',{data:original});backend.closeAllConnections();await new Promise(resolve=>backend.close(resolve));
 }
});
