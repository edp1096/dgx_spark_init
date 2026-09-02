import assert from 'node:assert/strict';
import test from 'node:test';
import { normalizeMarkdown, parseMarkdown } from './markdown.js';

test('renders long fenced code as a compact interactive card', () => {
  const source = Array.from({ length: 14 }, (_, index) => `const value${index} = ${index};`).join('\n');
  const html = parseMarkdown(`\`\`\`javascript\n${source}\n\`\`\``);
  assert.match(html, /code-card code-card-long/);
  assert.match(html, /data-code-copy/);
  assert.match(html, /data-code-toggle/);
  assert.match(html, /전체 보기/);
  assert.match(html, /language-javascript/);
});

test('unwraps a whole markdown fence emitted by a model', () => {
  assert.equal(normalizeMarkdown('```markdown\n# 제목\n\n**본문**\n```'), '# 제목\n\n**본문**');
});

test('keeps ordinary code fences intact', () => {
  const source = '설명\n\n```go\nfmt.Println("ok")\n```';
  assert.equal(normalizeMarkdown(source), source);
});

test('removes stray model control tokens', () => {
  assert.equal(normalizeMarkdown('<|turn>model\n# 제목'), 'model\n# 제목');
});

test('renders Gemma inline and block LaTeX with KaTeX', () => {
  const inline = parseMarkdown('이동 $\\rightarrow$ 오른쪽');
  assert.match(inline, /class="katex"/);
  assert.match(inline, /→/);

  const block = parseMarkdown('$$\n\\frac{35}{39}\n$$');
  assert.match(block, /class="katex-display"/);
});

test('renders bold text ending in punctuation before a Korean suffix', () => {
  const html = parseMarkdown('**한국경제신문(한경)**은 대한민국의 **종합 경제 일간지**야.');
  assert.match(html, /<strong>한국경제신문\(한경\)<\/strong><!-- -->은/u);
  assert.match(html, /<strong>종합 경제 일간지<\/strong><!-- -->야/u);
  assert.doesNotMatch(html, /\*\*/u);
});

test('does not rewrite strong-like text inside code', () => {
  const source = '`**한경(온라인)**은`\n\n```text\n**한경(온라인)**은\n```';
  assert.equal(normalizeMarkdown(source), source);
});
