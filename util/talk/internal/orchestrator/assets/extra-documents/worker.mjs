import fs from 'node:fs/promises';
import {render} from './renderer.mjs';
import {renderTextDocument} from './text-document.mjs';
try{const dir=process.argv[2];const input=JSON.parse(await fs.readFile(`${dir}/input.json`,'utf8'));const renderer=process.argv[3]==='text'?renderTextDocument:render;await fs.writeFile(`${dir}/output.json`,JSON.stringify(await renderer(input,dir)));}catch(err){console.error(err.message);process.exitCode=1;}
