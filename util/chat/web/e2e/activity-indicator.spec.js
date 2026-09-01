import { expect, test } from '@playwright/test';

test('uses the text brightness animation for active reasoning and tools', async ({ page }) => {
  await page.goto('/');
  await page.locator('body').evaluate((body) => {
    const probe = document.createElement('summary');
    probe.className = 'activity-pulse';
    probe.textContent = '생각 과정';
    body.append(probe);
  });

  const probe = page.locator('summary.activity-pulse');
  await expect(probe).toHaveCSS('animation-name', 'activity-text-pulse');
  await expect(probe).toHaveCSS('animation-duration', '1.45s');
});
