import fs from 'node:fs';
import path from 'node:path';
const out=[];
function visit(dir){for(const e of fs.readdirSync(dir,{withFileTypes:true})){if(!e.isDirectory())continue;const p=path.join(dir,e.name);if(e.name.startsWith('@')){visit(p);continue;}const file=path.join(p,'package.json');if(fs.existsSync(file)){const v=JSON.parse(fs.readFileSync(file));let license=v.license||v.licenses;if(!license&&v.name==='png-js'&&fs.readFileSync(path.join(p,'LICENSE'),'utf8').startsWith('MIT License'))license='MIT';if(!license)throw Error('Missing dependency license: '+v.name);out.push({name:v.name,version:v.version,license});}if(fs.existsSync(path.join(p,'node_modules')))visit(path.join(p,'node_modules'));}}
visit('/app/node_modules');fs.mkdirSync('/licenses',{recursive:true});fs.writeFileSync('/licenses/npm.json',JSON.stringify(out,null,2)+'\n');
