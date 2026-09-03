import { expect, test } from '@playwright/test';

test('keeps settings values while navigating categorized tabs', async ({ page }) => {
  await page.goto('/');
  await page.locator('.settings-button').click();

  const tabs = page.getByRole('tab');
  await expect(tabs).toHaveCount(6);
  await expect(page.getByRole('tab', { name: '대화' })).toHaveAttribute('aria-selected', 'true');

  const effort = page.getByLabel('기본 reasoning effort', { exact: true });
  await effort.selectOption('xhigh');
  await page.getByRole('tab', { name: '음성' }).click();
  await expect(page.getByText('음성 인식', { exact: true })).toBeVisible();
  const omitParentheticals = page.getByLabel('괄호 속 부연설명 읽지 않기');
  await expect(omitParentheticals).toBeChecked();
  await omitParentheticals.uncheck();
  await page.getByRole('tab', { name: '대화' }).click();
  await expect(effort).toHaveValue('xhigh');
  await page.getByRole('tab', { name: '음성' }).click();
  await expect(omitParentheticals).not.toBeChecked();
  await page.getByRole('tab', { name: '대화' }).click();

  const initialModal = await page.locator('.settings-modal').boundingBox();
  const initialActions = await page.locator('.modal-actions').boundingBox();
  for (const name of ['기억', '음성', '기능', '외형', '시스템']) {
    await page.getByRole('tab', { name }).click();
    const modal = await page.locator('.settings-modal').boundingBox();
    const actions = await page.locator('.modal-actions').boundingBox();
    expect(modal?.height).toBe(initialModal?.height);
    expect(actions?.y).toBe(initialActions?.y);
  }
});

test('changes Qwen reasoning with the compact header slider', async ({ page }) => {
  await page.goto('/');
  const effort = page.getByRole('slider', { name: 'Reasoning effort' });
  await expect(effort).toHaveValue('2');
  await effort.fill('3');
  await expect(page.locator('.model-controls .qwen-effort-control output')).toHaveText('XHigh');
  await effort.fill('0');
  await expect(page.locator('.model-controls .qwen-effort-control output')).toHaveText('꺼짐');
});

test('keeps tab navigation and actions reachable on mobile', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 600 });
  await page.goto('/');
  await expect(page.locator('.sidebar')).toHaveCount(0);
  await page.getByRole('button', { name: '사이드바 열기 또는 닫기' }).click();
  await expect(page.locator('.sidebar')).toBeVisible();
  await page.locator('.sidebar').evaluate((element) => Promise.all(element.getAnimations().map((animation) => animation.finished)));
  await page.locator('.settings-button').click();

  const chatTab = page.getByRole('tab', { name: '대화' });
  await chatTab.focus();
  await chatTab.press('End');
  await expect(page.getByRole('tab', { name: '시스템' })).toHaveAttribute('aria-selected', 'true');
  await expect(page.getByRole('button', { name: '저장' })).toBeVisible();

  const modal = await page.locator('.settings-modal').boundingBox();
  expect((modal?.y ?? -1)).toBeGreaterThanOrEqual(0);
  expect((modal?.y ?? 601) + (modal?.height ?? 0)).toBeLessThanOrEqual(600);
});

test('manages persistent memories from the memory library', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: '▤ 기억·지식' }).click();

  await page.getByRole('button', { name: '＋ 새 기억' }).click();
  const create = page.locator('.memory-compose-card');
  await create.getByLabel('제목').fill('응답 형식');
  await create.getByLabel('내용').fill('답변은 간결하게 작성');
  await create.getByRole('button', { name: '추가', exact: true }).click();

  const item = page.locator('.memory-library-item').last();
  await expect(item).toBeVisible();
  await expect(item).toContainText('관련 있을 때 참고');
  await expect(item).toContainText('우선 적용');
  await item.getByRole('button', { name: '수정' }).click();
  await expect(item.getByLabel('기억 내용')).toHaveValue('답변은 간결하게 작성');
  await item.getByLabel('기억 내용').fill('답변은 아주 간결하게 작성');
  await item.getByLabel('신뢰 수준').selectOption('reference');
  await item.getByRole('button', { name: '저장' }).click();
  await expect(item).toContainText('답변은 아주 간결하게 작성');
  await expect(item).toContainText('참고');
  page.once('dialog', (dialog) => dialog.accept());
  await item.getByRole('button', { name: '삭제' }).click();
  await expect(item).toHaveCount(0);
});

test('uploads and searches a knowledge document from settings', async ({ page }) => {
	let savedCollection = null;
	await page.route('**/api/knowledge/collections/1', async (route) => {
		if (route.request().method() !== 'PUT') return route.continue();
		savedCollection = route.request().postDataJSON();
		await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ id: 1, ...savedCollection, documents: 0 }) });
	});
	await page.route('**/api/knowledge/sources', async (route) => {
    const input = route.request().postDataJSON();
    expect(input).toEqual({ collection_id: 1, url: 'https://example.com/guide', mode: 'auto' });
    await route.fulfill({ status: 201, contentType: 'application/json', body: JSON.stringify({ document: { title: '웹 장치 안내' }, links: [{ text: '자료 PDF', url: 'https://example.com/guide.pdf' }] }) });
  });
	await page.route('**/api/knowledge/documents/*/ocr', async (route) => {
		await route.fulfill({ status: 502, contentType: 'text/plain', body: 'OCR 상세 오류: 원본 형식을 확인하세요.' });
	});
  await page.goto('/');
  await page.getByRole('button', { name: '▤ 기억·지식' }).click();
  await page.getByRole('button', { name: '지식 자료실', exact: true }).click();

  await expect(page.getByLabel('현재 보관함')).toContainText('내 지식');
	await page.locator('.settings-form-row.two').first().getByLabel('설명').fill('검증용 보관함');
	await page.getByRole('button', { name: '보관함 저장' }).click();
	expect(savedCollection).toEqual({ name: '내 지식', description: '검증용 보관함', enabled: true });
  await page.getByLabel('웹 문서 주소').fill('https://example.com/guide');
  await page.getByRole('button', { name: '주소 가져오기' }).click();
  await expect(page.getByText('“웹 장치 안내” 자료를 가져왔습니다.')).toBeVisible();
  await expect(page.getByText('수집한 페이지에서 찾은 링크 1개')).toBeVisible();
  await page.locator('.knowledge-file-input').setInputFiles({
    name: 'guide.md',
    mimeType: 'text/markdown',
    buffer: Buffer.from('# 테스트 지식\n\n스파크톡 지식 검색 화면 통합 시험입니다.'),
  });
  const document = page.locator('.knowledge-document').filter({ hasText: 'guide' });
  await expect(document).toContainText('검색 가능');
	await page.getByRole('button', { name: '파일 목록', exact: false }).click();
	await expect(document).toHaveCount(0);
	await page.getByRole('button', { name: '파일 목록', exact: false }).click();
	await page.getByRole('button', { name: '오류만', exact: false }).click();
	await expect(document).toHaveCount(0);
	await expect(page.getByText('문제가 표시된 문서가 없습니다.')).toBeVisible();
	await page.getByRole('button', { name: '전체', exact: true }).click();
	await expect(document).toBeVisible();

  await page.getByPlaceholder('현재 보관함 검색').fill('통합 시험');
  await page.getByRole('button', { name: '검색', exact: true }).click();
  await expect(page.locator('.knowledge-results')).toContainText('스파크톡 지식 검색 화면');

	await page.locator('.knowledge-file-input').setInputFiles({
		name: 'scan.png',
		mimeType: 'image/png',
		buffer: Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=', 'base64'),
	});
	const scan = page.locator('.knowledge-document').filter({ hasText: 'scan' });
	await scan.getByRole('button', { name: 'OCR 실행' }).click();
	const errorToast = page.getByRole('alert');
	await expect(errorToast).toContainText('OCR 상세 오류');
	await errorToast.locator('span').click();
	await expect(errorToast).toBeVisible();
	await expect(errorToast.getByRole('button', { name: '복사' })).toBeVisible();
	await errorToast.getByRole('button', { name: '알림 닫기' }).click();
	await expect(errorToast).toHaveCount(0);
});

test('uses a full-screen memory library on mobile and returns to chat', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 700 });
  await page.goto('/');
  await page.getByRole('button', { name: '사이드바 열기 또는 닫기' }).click();
  await page.getByRole('button', { name: /기억·지식/ }).click();

  await expect(page.locator('.sidebar')).toHaveCount(0);
  await expect(page.locator('.knowledge-hub')).toBeVisible();
  const hub = await page.locator('.knowledge-hub').boundingBox();
  expect(hub?.x).toBe(0);
  expect(hub?.width).toBe(390);
  expect(hub?.height).toBe(700);

  await page.getByRole('button', { name: '대화로 돌아가기' }).click();
  await expect(page.locator('.composer')).toBeVisible();
});
