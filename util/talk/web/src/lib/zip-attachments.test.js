import test from 'node:test';
import assert from 'node:assert/strict';
import { attachmentAccept, attachmentKind, isSupportedAttachmentFile } from './attachments.js';
test('ZIP supports picker, drag/drop and document presentation',()=>{
 assert.ok(attachmentAccept.includes('.zip'));
 for(const type of ['application/zip','application/x-zip-compressed','']){
  const file={name:'project.zip',type};assert.equal(isSupportedAttachmentFile(file),true);assert.equal(attachmentKind(file),'document');
 }
});
