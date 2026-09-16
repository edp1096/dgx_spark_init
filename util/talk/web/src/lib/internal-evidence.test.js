import test from 'node:test';
import assert from 'node:assert/strict';
import { stripInternalEvidence } from './internal-evidence.js';
test('hide reference records, retain answer JSON and code', () => {
 const line = '[Historical tool evidence: web_fetch; archive_id=524; use context_read for original] {"content":"internal"}';
 assert.equal(stripInternalEvidence(`Answer\n${line}\n{"normal":true}\n[normal]`),'Answer\n{"normal":true}\n[normal]');
 const code = '```text\n'+line+'\n```'; assert.equal(stripInternalEvidence(code),code);
 for(let n=12;n<line.indexOf(']');n++) assert.equal(stripInternalEvidence(line.slice(0,n)),'');
});
