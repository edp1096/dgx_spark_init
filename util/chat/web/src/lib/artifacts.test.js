import test from 'node:test';
import assert from 'node:assert/strict';
import { artifactsFromMessage } from './artifacts.js';

test('combines html css and javascript fences into one sandbox document', () => {
  const [artifact] = artifactsFromMessage({
    id: 7,
    role: 'assistant',
    content: '```html\n<!doctype html><html><head><title>카운터</title></head><body><button id="go">0</button></body></html>\n```\n```css\nbutton { color: red; }\n```\n```js\ndocument.querySelector("#go").onclick = () => {};\n```',
  });
  assert.equal(artifact.title, '카운터');
  assert.match(artifact.document, /Content-Security-Policy/);
  assert.match(artifact.document, /button \{ color: red; \}/);
  assert.match(artifact.document, /querySelector/);
  assert.match(artifact.document, /connect-src 'none'/);
});

test('ignores ordinary code blocks and user messages', () => {
  assert.deepEqual(artifactsFromMessage({ role: 'assistant', content: '```go\npackage main\n```' }), []);
  assert.deepEqual(artifactsFromMessage({ role: 'user', content: '```html\n<p>no</p>\n```' }), []);
});
