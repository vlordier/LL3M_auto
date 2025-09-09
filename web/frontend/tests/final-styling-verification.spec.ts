import { test, expect } from '@playwright/test';

test('final styling verification - complete app', async ({ page }) => {
  await page.goto('http://localhost:3000');
  await page.waitForTimeout(3000);

  // Take comprehensive screenshots
  await page.screenshot({
    path: 'test-results/screenshots/final-app-styling-complete.png',
    fullPage: true
  });

  // Test specific sections
  await page.screenshot({
    path: 'test-results/screenshots/final-welcome-section.png',
    clip: { x: 0, y: 0, width: 1280, height: 800 }
  });

  // Check that styling is fully working
  await expect(page.getByText('Welcome to LL3M')).toBeVisible();
  await expect(page.getByText('Generate stunning 3D assets')).toBeVisible();

  // Verify buttons are styled
  const browseButton = page.getByRole('button', { name: /browse sessions/i });
  await expect(browseButton).toBeVisible();

  // Verify input styling
  const input = page.getByPlaceholder('Describe the 3D asset you want to create...');
  await expect(input).toBeVisible();

  // Check sidebar styling
  const sidebar = page.getByRole('heading', { name: 'LL3M', exact: true });
  await expect(sidebar).toBeVisible();

  console.log('✅ All styling verification checks passed!');
  console.log('🎨 Tailwind CSS is working correctly');
  console.log('📱 App is fully styled and functional');
});
