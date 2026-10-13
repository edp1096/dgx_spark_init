import {test,expect} from '@playwright/test';
import {createServer} from 'node:http';

test('prompt searches and reads other chats then moves the selected chats',async({page,request})=>{
  let target,ids=[];
  const backend=createServer(async(req,res)=>{
    if(req.method!=='POST'){res.end(JSON.stringify({data:[{id:'test-model'}]}));return;}
    let raw='';for await(const part of req)raw+=part;const body=JSON.parse(raw);
    if(!body.stream){res.end(JSON.stringify({choices:[{message:{content:'정리'}}]}));return;}
    const last=body.messages.findLastIndex(item=>item.role==='user');const text=body.messages[last].content;
    const results=body.messages.slice(last+1).filter(item=>item.role==='tool').map(item=>JSON.parse(item.content));
    let name,args;
    if(results.length===0){name='conversation_search';args={query:'번역'};}
    if(results.length===1){ids=results[0].conversations.map(item=>item.session_id);name='conversation_read';args={session_ids:ids};}
    if(results.length===2){name='session_folder';args={action:'create',name:'번역'};}
    if(results.length===3){target=results[2].folder.id;name='session_folder';args={action:'move_selected',group_id:target,session_ids:ids};}
    res.setHeader('Content-Type','text/event-stream');
    const delta=args?{tool_calls:[{index:0,id:'organize-'+results.length,type:'function',function:{name,arguments:JSON.stringify({...args,user_request:text})}}]}:{content:'다른 번역 대화 2개를 옮겼습니다.'};
    res.end('data: '+JSON.stringify({choices:[{delta,finish_reason:args?'tool_calls':'stop'}]})+'\n\ndata: [DONE]\n\n');
  });
  await new Promise(resolve=>backend.listen(0,'127.0.0.1',resolve));
  const original=await(await request.get('/api/config')).json();const sessions=[];
  try{
    const cfg=structuredClone(original);cfg.model.endpoint=`http://127.0.0.1:${backend.address().port}`;cfg.context.enabled=false;cfg.tools.enabled=false;
    expect((await request.put('/api/config',{data:cfg})).ok()).toBeTruthy();
    for(const title of ['번역 A','번역 B','뉴스','정리 요청'])sessions.push(await(await request.post('/api/sessions',{data:{title}})).json());
    const current=sessions.at(-1);await request.patch(`/api/sessions/${current.id}`,{data:{title:current.title}});
    await page.goto('/');await page.locator('.composer textarea').fill('번역 폴더 만들고 다른 모든 번역 대화방을 번역 폴더로 옮겨라.');
    await page.getByRole('button',{name:'메시지 전송',exact:true}).click();
    await expect(page.getByText('다른 번역 대화 2개를 옮겼습니다.',{exact:true})).toBeVisible();
    const folder=page.locator('.chat-group').filter({has:page.locator('.group-toggle').filter({hasText:'번역'})});
    await expect(folder.locator('.session-select')).toHaveCount(2);
    const fresh=await(await request.get('/api/sessions')).json();
    for(const session of sessions.slice(0,2))expect(fresh.find(s=>s.id===session.id).group_id).toBe(target);
    for(const session of sessions.slice(2))expect(fresh.find(s=>s.id===session.id).group_id).toBe('');
    expect(ids.sort()).toEqual(sessions.slice(0,2).map(s=>s.id).sort());
    await page.reload();await expect(folder.locator('.session-select')).toHaveCount(2);
  }finally{
    backend.closeAllConnections();await new Promise(resolve=>backend.close(resolve));
    for(const session of sessions)await request.delete(`/api/sessions/${session.id}`);
    if(target)await request.delete(`/api/groups/${target}`);await request.put('/api/config',{data:original});
  }
});
