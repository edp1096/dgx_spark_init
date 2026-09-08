import { expect, test } from '@playwright/test';

async function mountArtifactPanel(page) {
  await page.locator('.shell').evaluate((shell) => {
    shell.classList.add('artifact-open');
    shell.style.gridTemplateColumns = '260px minmax(420px, 1fr) 500px';
    const panel = document.createElement('aside');
    panel.className = 'artifact-panel';
    panel.innerHTML = `
      <header class="artifact-header"><div><strong>아티팩트</strong><small>격리된 웹 미리보기</small></div><div class="artifact-header-actions"><button aria-label="생성물 닫기">×</button></div></header>
      <div class="artifact-toolbar"><div><button class="active">미리보기</button></div></div>
      <div class="artifact-stage"><iframe title="테스트 생성물" srcdoc="<h1>preview</h1>"></iframe></div>
      <footer class="artifact-security">외부 통신이 차단됩니다.</footer>`;
    shell.append(panel);
  });
}

test('uses a side-by-side artifact workspace on desktop', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/');
  await mountArtifactPanel(page);

  const panel = await page.locator('.artifact-panel').boundingBox();
  const main = await page.locator('main').boundingBox();
  const stage = await page.locator('.artifact-stage').boundingBox();
  expect(panel?.width).toBe(500);
  expect(panel?.height).toBe(900);
  expect(stage?.height).toBeGreaterThan(760);
  expect((main?.x ?? 0) + (main?.width ?? 0)).toBeLessThanOrEqual(panel?.x ?? 0);
});

test('uses a full-screen artifact workspace on mobile', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 700 });
  await page.goto('/');
  await mountArtifactPanel(page);

  const panel = await page.locator('.artifact-panel').boundingBox();
  const stage = await page.locator('.artifact-stage').boundingBox();
  expect(panel?.x).toBe(0);
  expect(panel?.y).toBe(0);
  expect(panel?.width).toBe(390);
  expect(panel?.height).toBe(700);
  expect(stage?.height).toBeGreaterThan(580);
  await expect(page.getByRole('button', { name: '생성물 닫기' })).toBeVisible();
});
