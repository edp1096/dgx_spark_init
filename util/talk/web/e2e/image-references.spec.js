import { expect, test } from '@playwright/test';

test('shows ordered actual image inputs and their roles on a small screen', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  const sessionId = 'reference-session';
  const now = new Date().toISOString();
  const inputs = [
    { index: 1, id: 'scene', name: 'scene.png', url: '/api/media/scene', role: 'source', origin: 'assistant', description: '직전 생성 장면', crop_box: [16, 16, 96, 128] },
    { index: 2, id: 'original', name: 'character.png', url: '/api/media/original', role: 'reference', origin: 'user', description: '원본 캐릭터 얼굴' },
  ];
  await page.route('**/api/groups', route => route.fulfill({ json: [] }));
  await page.route('**/api/models', route => route.fulfill({ json: ['test-model'] }));
  await page.route('**/api/sessions', route => route.fulfill({ json: [{ id: sessionId, title: '참조 편집', model: 'test-model', reasoning_effort: 'medium', group_id: '', created_at: now, updated_at: now }] }));
  await page.route(`**/api/sessions/${sessionId}/messages`, route => route.fulfill({ json: [{
    id: 1, session_id: sessionId, role: 'assistant', status: 'completed', content: '참조 편집 결과', reasoning_content: '', created_at: now,
    tool_trace: [{ id: 'edit', name: 'image_generate', arguments: JSON.stringify({ operation: 'identity_edit', prompt: 'Refine the character face.' }), result: JSON.stringify({ input_images: inputs, reference_groups: [{subject:'인물 A',scene_image_id:'a1',head_image_id:'a2',selection_reason:'정면 얼굴이 선명한 후보를 보정에 선택',candidates:[1,2,3].map(n=>({id:'a'+n,name:'same.jpg',url:'/api/media/a'+n}))},{subject:'인물 B',scene_image_id:'b1',head_image_id:'b3',selection_reason:'각도에 맞는 후보 선택',candidates:[1,2,3].map(n=>({id:'b'+n,name:'same.jpg',url:'/api/media/b'+n}))}] }) }],
  }] }));
  await page.route(`**/api/sessions/${sessionId}/context`, route => route.fulfill({ json: { enabled: true, segments: [] } }));
  await page.route(`**/api/sessions/${sessionId}/ssh-grants`, route => route.fulfill({ json: [] }));
  await page.goto('/');
  await page.locator('.tool-trace > summary').click();
  const cards = page.locator('[aria-label="실제 사용한 입력 이미지"] .image-input-card');
  await expect(cards).toHaveCount(2);
  await expect(cards.nth(0)).toContainText('1 · 편집 대상 · 생성 결과');
  await expect(cards.nth(1)).toContainText('2 · 참조 · 원본 첨부');
  await expect(cards.nth(0)).toContainText('사용 영역: 16, 16, 96, 128');
  await expect(cards.nth(0).locator('img')).toHaveAttribute('src', '/api/media/scene');
  await expect(cards.nth(1).locator('img')).toHaveAttribute('src', '/api/media/original');
  await page.getByText('인물 A · 후보 3장',{exact:true}).click();
  const group=page.locator('details.image-input-evidence').first();
  await expect(group.locator('.image-input-card')).toHaveCount(3);
  await expect(group.locator('.image-input-card').nth(0)).toContainText('장면 생성 선택');
  await expect(group.locator('.image-input-card').nth(1)).toContainText('얼굴 보정 선택');
  await expect(group).toContainText('정면 얼굴이 선명한 후보');
  const viewport = page.viewportSize();
  for (const card of await page.locator('.image-input-card:visible').all()) {
    const box = await card.boundingBox();
    expect(box.x).toBeGreaterThanOrEqual(0);
    expect(box.x + box.width).toBeLessThanOrEqual(viewport.width);
  }
});
