import { expect, test } from '@playwright/test';

test('keeps long chat code compact until the user expands it', async ({ page }) => {
  await page.setViewportSize({ width: 1000, height: 800 });
  await page.goto('/');
  await page.locator('.messages').evaluate((messages) => {
    const card = document.createElement('div');
    card.className = 'prose';
    card.innerHTML = `<div class="code-card code-card-long" data-code-card>
      <div class="code-card-header"><span>javascript</span><div><button type="button" data-code-toggle aria-expanded="false">전체 보기</button></div></div>
      <pre><code>${Array.from({ length: 40 }, (_, index) => `const value${index} = ${index};`).join('\n')}</code></pre>
    </div>`;
    messages.append(card);
  });

  const card = page.locator('[data-code-card]');
  const code = card.locator('pre');
  expect((await code.boundingBox())?.height).toBeLessThanOrEqual(230);
  await card.getByRole('button', { name: '전체 보기' }).click();
  await expect(card).toHaveClass(/expanded/);
  expect((await code.boundingBox())?.height).toBeGreaterThan(500);
  await card.getByRole('button', { name: '접기' }).click();
  await expect(card).not.toHaveClass(/expanded/);
});
