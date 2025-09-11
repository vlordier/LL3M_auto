import { test, expect } from '../fixtures/test-helpers';

test.describe('Application Startup', () => {
  test('should load the application successfully', async ({ appPage }) => {
    await appPage.goto();

    // Take screenshot of initial load
    await appPage.takeFullPageScreenshot('app-startup-initial');

    // Check for main application elements
    await expect(appPage.page.getByText('Welcome to LL3M')).toBeVisible();
    await expect(appPage.page.getByText('Generate stunning 3D assets using natural language')).toBeVisible();

    // Check for example prompts
    await expect(appPage.page.getByText('Simple Objects')).toBeVisible();
    await expect(appPage.page.getByText('Fantasy & Sci-Fi')).toBeVisible();
    await expect(appPage.page.getByText('Architectural')).toBeVisible();

    // Check for input field
    await expect(appPage.chatInput).toBeVisible();
    await expect(appPage.sendButton).toBeVisible();

    // Take screenshot after validation
    await appPage.takeFullPageScreenshot('app-startup-validated');
  });

  test('should show sidebar with branding', async ({ appPage }) => {
    await appPage.goto();

    // Check sidebar elements
    await expect(appPage.page.getByRole('heading', { name: 'LL3M', exact: true })).toBeVisible();
    await expect(appPage.page.getByRole('button', { name: /browse sessions/i })).toBeVisible();
    await expect(appPage.page.getByRole('button', { name: /settings/i })).toBeVisible();

    // Check version and status info
    await expect(appPage.page.getByText('Version')).toBeVisible();
    await expect(appPage.page.getByText('Status')).toBeVisible();
    await expect(appPage.page.getByText('Connected')).toBeVisible();

    await appPage.takeFullPageScreenshot('sidebar-elements');
  });

  test('should toggle sidebar visibility', async ({ appPage }) => {
    await appPage.goto();

    // Initial state - sidebar should be visible
    await expect(appPage.page.getByRole('heading', { name: 'LL3M', exact: true })).toBeVisible();
    await appPage.takeFullPageScreenshot('sidebar-open');

    // Click collapse button
    await appPage.page.getByRole('button', { name: /chevron.*left/i }).click();
    await appPage.page.waitForTimeout(500); // Allow animation

    // Sidebar should be collapsed
    await expect(appPage.page.getByText('LL3M')).toBeHidden();
    await appPage.takeFullPageScreenshot('sidebar-collapsed');

    // Click expand button
    await appPage.page.getByRole('button', { name: /chevron.*right/i }).click();
    await appPage.page.waitForTimeout(500); // Allow animation

    // Sidebar should be visible again
    await expect(appPage.page.getByText('LL3M')).toBeVisible();
    await appPage.takeFullPageScreenshot('sidebar-reopened');
  });

  test('should handle example prompt clicks', async ({ appPage }) => {
    await appPage.goto();

    // Click on first example
    const firstExample = appPage.page.getByText('Create a wooden chair with curved armrests');
    await expect(firstExample).toBeVisible();

    await firstExample.click();

    // Check that the text appears in input field
    await expect(appPage.chatInput).toHaveValue('Create a wooden chair with curved armrests');

    await appPage.takeFullPageScreenshot('example-prompt-filled');
  });

  test('should show responsive layout on mobile', async ({ page }) => {
    // Set mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/');

    // Check mobile-specific layout
    await expect(page.getByText('Welcome to LL3M')).toBeVisible();

    // Check that examples are stacked vertically on mobile
    const exampleCards = page.locator('[data-testid="example-cards"]');

    await page.screenshot({
      path: 'test-results/screenshots/mobile-layout.png',
      fullPage: true
    });
  });
});
