import http from 'node:http';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {spawn} from 'node:child_process';
let busy=false;
function reply(res,status,body){res.writeHead(status,{'Content-Type':'application/json'});res.end(JSON.stringify(body));}
const server=http.createServer(async(req,res)=>{
 if(req.method==='GET'&&req.url==='/health'){reply(res,200,{status:'ok',formats:['docx','pptx','xlsx','pdf','hwp','hwpx'],busy});return;}
 if(req.method!=='POST'||req.url!=='/v1/documents'){reply(res,404,{error:'Not found'});return;}
 if(busy){reply(res,429,{error:'Another document is being generated. Retry after it finishes.'});return;}
 busy=true;let dir,child,timer;let aborted=false;let result,status=200;
 const stop=()=>{aborted=true;if(child?.pid){try{process.kill(-child.pid,'SIGKILL');}catch{}}};
 res.on('close',()=>{if(!res.writableEnded)stop();});
 try{
  let size=0;const chunks=[];for await(const chunk of req){size+=chunk.length;if(size>16<<20)throw Error('Request exceeds 16 MiB');chunks.push(chunk);}
  const data=Buffer.concat(chunks);JSON.parse(data.toString());if(aborted)throw Error('Request cancelled');
  dir=await fs.mkdtemp(path.join(os.tmpdir(),'sparktalk-document-'));await fs.writeFile(path.join(dir,'input.json'),data,{mode:0o600});
  child=spawn(process.execPath,[new URL('./worker.mjs',import.meta.url).pathname,dir],{detached:true,stdio:['ignore','ignore','pipe']});
  let detail='';child.stderr.on('data',b=>{if(detail.length<4096)detail+=b.toString();});
  timer=setTimeout(stop,90000);
  const code=await new Promise((resolve,reject)=>{child.once('error',reject);child.once('exit',resolve);});
  if(code!==0||aborted)throw Error(aborted?'Document generation timed out or was cancelled':detail.trim()||'Document generation failed');
  result=JSON.parse(await fs.readFile(path.join(dir,'output.json'),'utf8'));
 }catch(err){status=422;result={error:err.message};}
 finally{clearTimeout(timer);try{if(dir)await fs.rm(dir,{recursive:true,force:true});}catch{status=500;result={error:'Temporary file cleanup failed'};}busy=false;if(!res.destroyed)reply(res,status,result);}
});
server.requestTimeout=100000;server.headersTimeout=15000;
server.listen(Number(process.env.PORT||8696),process.env.BIND_ADDRESS||'127.0.0.1');
