import fs from 'node:fs/promises';
import {render} from './renderer.mjs';
try{const dir=process.argv[2];const input=JSON.parse(await fs.readFile(`${dir}/input.json`,'utf8'));await fs.writeFile(`${dir}/output.json`,JSON.stringify(await render(input,dir)));}catch(err){console.error(err.message);process.exitCode=1;}
