import test from 'node:test';
import assert from 'node:assert/strict';
import { avatarImageID, avatarURL } from './avatars.js';

test('avatarURL resolves embedded presets and custom media', () => {
  assert.equal(avatarURL('preset:robot'), '/avatars/robot.png');
  assert.equal(avatarURL('preset:earth'), '/avatars/earth.png');
  assert.equal(avatarURL('preset:saturn'), '/avatars/saturn.png');
  assert.equal(avatarURL('preset:dog'), '/avatars/dog.png');
  assert.equal(avatarURL('preset:quantum-computer'), '/avatars/quantum-computer.png');
  assert.equal(avatarURL('preset:computer'), '/avatars/quantum-computer.png');
  assert.equal(avatarURL('preset:bear'), '/avatars/bear.png');
  assert.equal(avatarURL('preset:rabbit'), '/avatars/rabbit.png');
  assert.equal(avatarURL('/api/images/abc'), '/api/images/abc');
  assert.equal(avatarURL('invalid', 'person-blue'), '/avatars/person-blue.png');
});

test('avatarImageID returns only custom media ids', () => {
  assert.equal(avatarImageID('/api/images/abc'), 'abc');
  assert.equal(avatarImageID('preset:spark'), '');
});
