import { test, expect } from '@playwright/test';

test('basic page load test', async ({ page }) => {
  // Navigate to the app
  await page.goto('http://localhost:3000');

  // Take a screenshot of what we see
  await page.screenshot({
    path: 'test-results/screenshots/basic-page-load.png',
    fullPage: true
  });

  // Check if the page has loaded
  const title = await page.title();
  expect(title).toContain('LL3M');

  // Wait a bit for React to render
  await page.waitForTimeout(3000);

  // Take another screenshot after waiting
  await page.screenshot({
    path: 'test-results/screenshots/after-wait.png',
    fullPage: true
  });

  // Look for any content that might be there
  const pageText = await page.textContent('body');
  console.log('Page text:', pageText?.substring(0, 200));

  // Check for basic elements
  const welcomeText = page.getByText('Welcome to LL3M');
  if (await welcomeText.isVisible({ timeout: 5000 })) {
    console.log('✅ Welcome text found');
  } else {
    console.log('⚠️ Welcome text not found');
  }

  // Check for any error messages
  const errorElements = await page.locator('[role="alert"], .error, .alert-error').all();
  if (errorElements.length > 0) {
    console.log('❌ Found error elements on page');
  }
});
