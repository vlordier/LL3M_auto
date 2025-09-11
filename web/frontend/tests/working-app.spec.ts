import { test, expect } from '@playwright/test';

test('app loads and shows welcome screen', async ({ page }) => {
  await page.goto('http://localhost:3000');

  // Wait for React to render
  await page.waitForTimeout(2000);

  // Take initial screenshot
  await page.screenshot({
    path: 'test-results/screenshots/working-app-loaded.png',
    fullPage: true
  });

  // Look for the welcome message
  await expect(page.getByText('Welcome to LL3M')).toBeVisible({ timeout: 10000 });

  // Look for the description
  await expect(page.getByText('Generate stunning 3D assets using natural language')).toBeVisible();

  // Look for example prompts
  await expect(page.getByText('Simple Objects')).toBeVisible();
  await expect(page.getByText('Fantasy & Sci-Fi')).toBeVisible();
  await expect(page.getByText('Architectural')).toBeVisible();

  // Look for the input field
  const input = page.getByPlaceholder('Describe the 3D asset you want to create...');
  await expect(input).toBeVisible();

  // Look for sidebar - use more specific selector
  await expect(page.getByRole('heading', { name: 'LL3M', exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: /browse sessions/i })).toBeVisible();

  // Take final screenshot showing successful load
  await page.screenshot({
    path: 'test-results/screenshots/working-app-ready.png',
    fullPage: true
  });

  console.log('✅ App loaded successfully with all expected elements!');
});
