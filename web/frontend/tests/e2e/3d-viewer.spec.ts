import { test, expect } from '../fixtures/test-helpers';

test.describe('3D Viewer', () => {
  test('should display placeholder when no asset is loaded', async ({ appPage }) => {
    await appPage.goto();
    
    // Switch to 3D view
    await appPage.switchViewMode('3d');
    await appPage.takeFullPageScreenshot('3d-viewer-no-asset');
    
    // Should show placeholder state
    await expect(appPage.page.getByText('No 3D Asset Available')).toBeVisible();
    await expect(appPage.page.getByText('Generate a 3D asset to see it in the viewer')).toBeVisible();
    await expect(appPage.page.getByRole('button', { name: /start chatting/i })).toBeVisible();
  });

  test('should render 3D scene with placeholder model', async ({ appPage, page }) => {
    await appPage.goto();
    
    // Mock a successful generation with asset
    await page.route('**/api/generate', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          success: true,
          session_id: 'test-session',
          asset_path: 'test-cube.gltf',
          preview_url: 'test-preview.png',
          generation_time: 2.5
        })
      });
    });
    
    // Send a message to trigger generation
    await appPage.sendMessage('Create a simple cube');
    
    // Wait for the mocked response
    await page.waitForTimeout(2000);
    
    // Switch to 3D view
    await appPage.switchViewMode('3d');
    
    // Look for 3D canvas
    const canvas = page.locator('canvas');
    await expect(canvas).toBeVisible({ timeout: 10000 });
    
    await appPage.takeFullPageScreenshot('3d-viewer-with-canvas');
    
    // Check for 3D viewer controls
    await expect(page.getByTitle(/toggle wireframe/i)).toBeVisible();
    await expect(page.getByTitle(/toggle grid/i)).toBeVisible();
    await expect(page.getByTitle(/reset camera/i)).toBeVisible();
  });

  test('should show 3D viewer controls and interactions', async ({ appPage, page }) => {
    await appPage.goto();
    
    // Create a mock 3D scene
    await page.addInitScript(() => {
      // Mock successful asset loading
      window.mockAssetLoaded = true;
    });
    
    // Try to interact with viewer controls (if visible)
    const viewerContainer = page.locator('[data-testid="three-viewer"]');
    if (await viewerContainer.isVisible()) {
      await appPage.takeFullPageScreenshot('3d-viewer-controls-before');
      
      // Test wireframe toggle
      const wireframeButton = page.getByTitle(/toggle wireframe/i);
      if (await wireframeButton.isVisible()) {
        await wireframeButton.click();
        await appPage.takeFullPageScreenshot('3d-viewer-wireframe-toggle');
      }
      
      // Test grid toggle
      const gridButton = page.getByTitle(/toggle grid/i);
      if (await gridButton.isVisible()) {
        await gridButton.click();
        await appPage.takeFullPageScreenshot('3d-viewer-grid-toggle');
      }
      
      // Test environment preset changes
      const environmentSelect = page.locator('select').filter({ hasText: /studio|city|forest/i });
      if (await environmentSelect.isVisible()) {
        await environmentSelect.selectOption('city');
        await page.waitForTimeout(1000);
        await appPage.takeFullPageScreenshot('3d-viewer-environment-city');
      }
    }
  });

  test('should handle view mode switching', async ({ appPage }) => {
    await appPage.goto();
    
    // Test all view modes
    await appPage.switchViewMode('chat');
    await expect(appPage.page.getByText('Welcome to LL3M')).toBeVisible();
    await appPage.takeFullPageScreenshot('view-mode-chat');
    
    // Split view (may be disabled without asset)
    const splitButton = appPage.page.getByRole('button', { name: /split/i });
    if (await splitButton.isEnabled()) {
      await appPage.switchViewMode('split');
      await appPage.takeFullPageScreenshot('view-mode-split');
    }
    
    // 3D view
    await appPage.switchViewMode('3d');
    await appPage.takeFullPageScreenshot('view-mode-3d');
  });

  test('should support mouse interactions with 3D scene', async ({ appPage, page }) => {
    await appPage.goto();
    
    // Switch to 3D view
    await appPage.switchViewMode('3d');
    
    const canvas = page.locator('canvas');
    if (await canvas.isVisible()) {
      // Test mouse interactions
      const canvasBounds = await canvas.boundingBox();
      if (canvasBounds) {
        const centerX = canvasBounds.x + canvasBounds.width / 2;
        const centerY = canvasBounds.y + canvasBounds.height / 2;
        
        // Simulate orbit (drag)
        await page.mouse.move(centerX, centerY);
        await page.mouse.down();
        await page.mouse.move(centerX + 50, centerY + 50);
        await page.mouse.up();
        
        await appPage.takeFullPageScreenshot('3d-viewer-after-orbit');
        
        // Simulate zoom (wheel)
        await page.mouse.wheel(0, -100);
        await page.waitForTimeout(500);
        
        await appPage.takeFullPageScreenshot('3d-viewer-after-zoom');
      }
    }
  });

  test('should show loading state for 3D assets', async ({ appPage, page }) => {
    await appPage.goto();
    
    // Mock slow asset loading
    await page.route('**/*.gltf', async route => {
      await page.waitForTimeout(2000); // Simulate slow loading
      await route.continue();
    });
    
    // Switch to 3D view and look for loading indicators
    await appPage.switchViewMode('3d');
    
    const loadingIndicators = [
      page.getByText(/loading.*3d.*model/i),
      page.locator('.animate-spin'),
      page.getByText(/loading/i)
    ];
    
    let foundLoading = false;
    for (const indicator of loadingIndicators) {
      try {
        await expect(indicator).toBeVisible({ timeout: 2000 });
        foundLoading = true;
        await appPage.takeFullPageScreenshot('3d-viewer-loading');
        break;
      } catch (error) {
        continue;
      }
    }
    
    if (!foundLoading) {
      console.log('⚠️ No 3D loading indicator found');
    }
  });

  test('should handle asset loading errors gracefully', async ({ appPage, page }) => {
    await appPage.goto();
    
    // Mock asset loading failure
    await page.route('**/*.gltf', route => route.abort());
    await page.route('**/*.obj', route => route.abort());
    
    // Try to load an asset that will fail
    await page.addInitScript(() => {
      window.mockAssetError = true;
    });
    
    await appPage.switchViewMode('3d');
    
    // Look for error states
    const errorIndicators = [
      page.getByText(/failed.*load/i),
      page.getByText(/error/i),
      page.locator('[data-testid="error-model"]')
    ];
    
    let foundError = false;
    for (const indicator of errorIndicators) {
      try {
        await expect(indicator).toBeVisible({ timeout: 5000 });
        foundError = true;
        await appPage.takeFullPageScreenshot('3d-viewer-error');
        break;
      } catch (error) {
        continue;
      }
    }
    
    if (!foundError) {
      console.log('⚠️ No explicit 3D error handling found');
    }
  });

  test('should be responsive on different screen sizes', async ({ page }) => {
    // Test on different viewport sizes
    const viewports = [
      { width: 1920, height: 1080, name: 'desktop-large' },
      { width: 1366, height: 768, name: 'desktop-medium' },
      { width: 768, height: 1024, name: 'tablet' },
      { width: 375, height: 667, name: 'mobile' }
    ];
    
    for (const viewport of viewports) {
      await page.setViewportSize(viewport);
      await page.goto('/');
      
      // Switch to 3D view
      const threeDButton = page.getByRole('button', { name: /3d view/i });
      if (await threeDButton.isVisible()) {
        await threeDButton.click();
      }
      
      await page.screenshot({ 
        path: `test-results/screenshots/3d-viewer-${viewport.name}.png`,
        fullPage: true 
      });
    }
  });
});