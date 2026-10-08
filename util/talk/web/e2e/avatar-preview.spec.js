import { test, expect } from '@playwright/test';

test('enlarges both avatars, closes with Escape, and keeps profile settings accessible', async ({ page, request }) => {
  const session = await (await request.post('/api/sessions', { data: { title: 'Avatar preview' } })).json();
  try {
    await page.route('**/api/sessions/*/messages', route => route.fulfill({ json: [
      { id: 10001, role: 'user', content: '안녕', status: 'completed', attachments: [] },
      { id: 10002, role: 'assistant', content: '반갑습니다.', status: 'completed', attachments: [] },
    ] }));
    await page.goto('/');
    const user = page.getByRole('button', { name: '사용자 아바타 크게 보기', exact: true });
    await user.click();
    let dialog = page.getByRole('dialog', { name: /아바타 크게 보기/ });
    await expect(dialog.locator('img')).toHaveAttribute('src', '/avatars/person-blue.png');
    await page.keyboard.press('Escape');
    await expect(dialog).toHaveCount(0);
    await expect(user).toBeFocused();
    await page.getByRole('button', { name: 'AI 아바타 크게 보기', exact: true }).last().click();
    await expect(dialog.locator('img')).toHaveAttribute('src', '/avatars/spark.png');
    await page.setViewportSize({ width: 390, height: 760 });
    const box = await dialog.boundingBox();
    expect(box.width).toBeLessThanOrEqual(390);
    expect(box.height).toBeLessThanOrEqual(736);
    await dialog.getByRole('button', { name: '프로필 설정', exact: true }).click();
    await expect(page.getByRole('dialog', { name: '설정', exact: true })).toBeVisible();
    await expect(page.getByRole('dialog', { name: /아바타 크게 보기/ })).toHaveCount(0);
  } finally { await request.delete(`/api/sessions/${session.id}`); }
});
