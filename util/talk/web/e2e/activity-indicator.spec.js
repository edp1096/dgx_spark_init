import { expect, test } from '@playwright/test';

test('uses a KITT-like text brightness scanner even with reduced motion enabled', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto('/');
  await page.locator('body').evaluate((body) => {
    const probe = document.createElement('span');
    probe.className = 'activity-label activity-scanner';
    probe.textContent = '생각 과정';
    body.append(probe);
  });

  const probe = page.locator('.activity-scanner');
  await expect(probe).toHaveCSS('animation-name', 'activity-text-scanner');
  await expect(probe).toHaveCSS('animation-duration', '1.15s');
  await expect(probe).toHaveCSS('animation-direction', 'alternate');
  await expect(probe).toHaveCSS('background-clip', 'text');
});
