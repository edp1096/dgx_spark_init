import { expect, test } from '@playwright/test';

test('does not offer a stale conversation model after a runtime switch', async ({ page }) => {
  const currentModel = 'current-runtime-model';
  await page.route('**/api/models', (route) => route.fulfill({ json: [currentModel] }));
  await page.route('**/api/sessions', (route) => route.fulfill({ json: [{
    id: 'stale-model-session',
    title: '기존 대화',
    model: 'previous-runtime-model',
    reasoning_effort: 'medium',
    group_id: '',
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  }] }));
  await page.route('**/api/sessions/stale-model-session/messages', (route) => route.fulfill({ json: [] }));
  await page.route('**/api/sessions/stale-model-session/context', (route) => route.fulfill({ json: { enabled: true, segments: [] } }));
  await page.route('**/api/sessions/stale-model-session/ssh-grants', (route) => route.fulfill({ json: [] }));

  await page.goto('/');

  const selector = page.locator('.model-controls select[aria-label="모델 선택"]');
  await expect(selector).toHaveValue(currentModel);
  await expect(selector.locator('option')).toHaveCount(1);
  await expect(selector.locator('option')).toHaveText(currentModel);
});
