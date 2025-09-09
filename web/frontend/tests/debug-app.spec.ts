import { test, expect } from '@playwright/test';

test('debug app loading', async ({ page }) => {
  // Capture console logs
  const logs: string[] = [];
  const errors: string[] = [];

  page.on('console', msg => {
    if (msg.type() === 'error') {
      errors.push(`Console Error: ${msg.text()}`);
    } else {
      logs.push(`Console ${msg.type()}: ${msg.text()}`);
    }
  });

  page.on('pageerror', error => {
    errors.push(`Page Error: ${error.message}`);
  });

  // Navigate to the app
  await page.goto('http://localhost:3000');

  // Wait for potential errors to surface
  await page.waitForTimeout(5000);

  // Log what we found
  console.log('=== CONSOLE LOGS ===');
  logs.forEach(log => console.log(log));

  console.log('=== ERRORS ===');
  errors.forEach(error => console.log(error));

  // Take screenshot
  await page.screenshot({
    path: 'test-results/screenshots/debug-app.png',
    fullPage: true
  });

  // Check page source
  const content = await page.content();
  console.log('=== PAGE CONTENT ===');
  console.log(content.substring(0, 1000));
});
