import { test, expect } from '../fixtures/test-helpers';

test.describe('Full User Journey', () => {
  test('complete asset generation workflow', async ({ appPage, workflowPage }) => {
    // Start recording for this critical test
    await test.step('Application startup', async () => {
      await appPage.goto();
      await appPage.takeFullPageScreenshot('journey-01-startup');

      // Verify initial state
      await expect(appPage.page.getByText('Welcome to LL3M')).toBeVisible();
      await expect(appPage.chatInput).toBeVisible();
    });

    await test.step('Send generation request', async () => {
      const prompt = 'Create a wooden coffee table with rounded edges and metal legs';
      await appPage.sendMessage(prompt);

      await appPage.takeFullPageScreenshot('journey-02-message-sent');

      // Verify message appears
      await expect(appPage.page.getByText(prompt)).toBeVisible();
    });

    await test.step('Monitor workflow progress', async () => {
      // Mock progressive workflow updates
      await workflowPage.page.route('**/api/generate', async route => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            success: true,
            session_id: 'journey-session',
            traces: [
              {
                trace_id: 'prompt-trace',
                phase: 'prompt_analysis',
                start_time: new Date().toISOString(),
                end_time: new Date().toISOString(),
                execution_time: 1.8,
                outputs: { complexity: 'medium', estimated_time: '45s' }
              },
              {
                trace_id: 'code-trace',
                phase: 'code_generation',
                start_time: new Date().toISOString(),
                end_time: null,
                execution_time: null,
                outputs: null
              }
            ]
          })
        });
      });

      // Wait for workflow to appear
      await expect(workflowPage.page.getByText('Generation Pipeline')).toBeVisible({ timeout: 15000 });
      await appPage.takeFullPageScreenshot('journey-03-workflow-active');

      // Check for step progress
      await expect(workflowPage.page.getByText('Prompt Analysis')).toBeVisible();
      await expect(workflowPage.page.getByText('Code Generation')).toBeVisible();
    });

    await test.step('View generated asset', async () => {
      // Mock completed generation
      await workflowPage.page.route('**/api/generate', async route => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            success: true,
            session_id: 'journey-session',
            asset_path: 'generated/coffee-table.gltf',
            preview_url: 'previews/coffee-table-preview.png',
            generation_time: 42.3,
            traces: [
              {
                phase: 'prompt_analysis',
                start_time: new Date(Date.now() - 45000).toISOString(),
                end_time: new Date(Date.now() - 43000).toISOString(),
                execution_time: 1.8
              },
              {
                phase: 'code_generation',
                start_time: new Date(Date.now() - 43000).toISOString(),
                end_time: new Date(Date.now() - 38000).toISOString(),
                execution_time: 4.2
              },
              {
                phase: 'asset_generation',
                start_time: new Date(Date.now() - 38000).toISOString(),
                end_time: new Date().toISOString(),
                execution_time: 36.3
              }
            ]
          })
        });
      });

      // Wait for generation completion
      await appPage.page.waitForTimeout(5000);

      // Look for asset preview
      await expect(appPage.page.getByText(/asset.*generated.*successfully/i)).toBeVisible({ timeout: 30000 });
      await appPage.takeFullPageScreenshot('journey-04-generation-complete');

      // Switch to 3D view
      await appPage.switchViewMode('3d');
      await appPage.takeFullPageScreenshot('journey-05-3d-view');
    });

    await test.step('Interact with 3D viewer', async () => {
      const canvas = appPage.page.locator('canvas');
      if (await canvas.isVisible()) {
        // Test 3D interactions
        const bounds = await canvas.boundingBox();
        if (bounds) {
          // Simulate orbit gesture
          await appPage.page.mouse.move(bounds.x + bounds.width/2, bounds.y + bounds.height/2);
          await appPage.page.mouse.down();
          await appPage.page.mouse.move(bounds.x + bounds.width/2 + 50, bounds.y + bounds.height/2 + 30);
          await appPage.page.mouse.up();

          await appPage.takeFullPageScreenshot('journey-06-3d-interaction');
        }

        // Test viewer controls
        const wireframeButton = appPage.page.getByTitle(/toggle wireframe/i);
        if (await wireframeButton.isVisible()) {
          await wireframeButton.click();
          await appPage.takeFullPageScreenshot('journey-07-wireframe-toggle');
        }
      }
    });

    await test.step('Test split view mode', async () => {
      await appPage.switchViewMode('split');
      await appPage.takeFullPageScreenshot('journey-08-split-view');

      // Verify both chat and 3D viewer are visible
      await expect(appPage.chatInput).toBeVisible();
      const canvas = appPage.page.locator('canvas');
      if (await canvas.isVisible()) {
        console.log('✅ Split view working - both chat and 3D viewer visible');
      }
    });

    await test.step('Browse session history', async () => {
      // Mock session history
      await appPage.page.route('**/api/sessions', async route => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify([
            {
              session_id: 'journey-session',
              prompt: 'Create a wooden coffee table with rounded edges and metal legs',
              current_phase: 'completed',
              created_at: new Date().toISOString(),
              traces: [
                { phase: 'completed', execution_time: 42.3 }
              ],
              artifacts: [
                {
                  artifact_type: 'model',
                  relative_path: 'coffee-table.gltf',
                  size_bytes: 1536000
                }
              ],
              total_duration: 67.8
            }
          ])
        });
      });

      await appPage.sessionBrowserButton.click();
      await expect(appPage.page.getByText('Session Browser')).toBeVisible();
      await appPage.takeFullPageScreenshot('journey-09-session-browser');

      // Verify session appears
      await expect(appPage.page.getByText('coffee table')).toBeVisible();
      await expect(appPage.page.getByText('completed')).toBeVisible();

      // Close browser
      await appPage.page.getByRole('button', { name: /close/i }).first().click();
    });

    await test.step('Test amendment/refinement', async () => {
      // Switch back to chat
      await appPage.switchViewMode('chat');

      // Send refinement request
      const refinement = 'Make the table legs taller and add a shelf underneath';
      await appPage.sendMessage(refinement);

      await appPage.takeFullPageScreenshot('journey-10-refinement-request');

      // Verify refinement message appears
      await expect(appPage.page.getByText(refinement)).toBeVisible();
    });

    await test.step('Final verification', async () => {
      await appPage.takeFullPageScreenshot('journey-11-final-state');

      // Verify application is in good state
      await expect(appPage.chatInput).toBeVisible();
      await expect(appPage.page.getByText('LL3M')).toBeVisible();

      // Check that we can still interact
      await appPage.chatInput.fill('Test final interaction');
      await expect(appPage.sendButton).toBeEnabled();

      console.log('🎉 Complete user journey test passed successfully!');
    });
  });

  test('error recovery workflow', async ({ appPage }) => {
    await appPage.goto();

    await test.step('Handle generation error', async () => {
      // Mock generation failure
      await appPage.page.route('**/api/generate', async route => {
        await route.fulfill({
          status: 500,
          contentType: 'application/json',
          body: JSON.stringify({
            success: false,
            error: 'Generation service temporarily unavailable',
            session_id: null
          })
        });
      });

      await appPage.sendMessage('This will fail');
      await appPage.takeFullPageScreenshot('error-01-generation-failed');

      // Look for error handling
      await appPage.page.waitForTimeout(3000);

      // Verify app is still functional
      await expect(appPage.chatInput).toBeVisible();
      await expect(appPage.sendButton).toBeEnabled();
    });

    await test.step('Recover and try again', async () => {
      // Mock successful retry
      await appPage.page.route('**/api/generate', async route => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            success: true,
            session_id: 'recovery-session',
            message: 'Generation completed successfully'
          })
        });
      });

      await appPage.sendMessage('Retry the generation');
      await appPage.takeFullPageScreenshot('error-02-recovery-attempt');

      // Verify recovery
      await expect(appPage.page.getByText('Retry the generation')).toBeVisible();
    });
  });

  test('mobile user journey', async ({ page }) => {
    // Set mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/');

    await test.step('Mobile app startup', async () => {
      await page.screenshot({
        path: 'test-results/screenshots/mobile-journey-01-startup.png',
        fullPage: true
      });

      // Verify mobile layout
      await expect(page.getByText('Welcome to LL3M')).toBeVisible();
    });

    await test.step('Mobile chat interaction', async () => {
      const input = page.getByPlaceholder(/describe.*3d.*asset/i);
      await input.fill('Create a small decorative vase');
      await page.getByRole('button', { name: /send/i }).click();

      await page.screenshot({
        path: 'test-results/screenshots/mobile-journey-02-message-sent.png',
        fullPage: true
      });
    });

    await test.step('Mobile 3D viewer', async () => {
      // Try to switch to 3D view on mobile
      const threeDButton = page.getByRole('button', { name: /3d.*view/i });
      if (await threeDButton.isVisible()) {
        await threeDButton.click();

        await page.screenshot({
          path: 'test-results/screenshots/mobile-journey-03-3d-view.png',
          fullPage: true
        });
      }
    });

    await test.step('Mobile session browser', async () => {
      const browserButton = page.getByRole('button', { name: /browse.*sessions/i });
      if (await browserButton.isVisible()) {
        await browserButton.click();

        await page.screenshot({
          path: 'test-results/screenshots/mobile-journey-04-session-browser.png',
          fullPage: true
        });
      }
    });
  });

  test('accessibility compliance', async ({ appPage }) => {
    await appPage.goto();

    await test.step('Check keyboard navigation', async () => {
      // Tab through interface
      await appPage.page.keyboard.press('Tab');
      await appPage.page.keyboard.press('Tab');
      await appPage.page.keyboard.press('Tab');

      await appPage.takeFullPageScreenshot('accessibility-01-keyboard-focus');

      // Verify focus is visible
      const focusedElement = await appPage.page.locator(':focus').first();
      expect(await focusedElement.count()).toBeGreaterThan(0);
    });

    await test.step('Check ARIA labels and roles', async () => {
      // Look for proper ARIA attributes
      const ariaElements = [
        appPage.page.locator('[role="button"]'),
        appPage.page.locator('[aria-label]'),
        appPage.page.locator('[aria-describedby]')
      ];

      for (const element of ariaElements) {
        const count = await element.count();
        console.log(`Found ${count} elements with ARIA attributes`);
      }

      await appPage.takeFullPageScreenshot('accessibility-02-aria-check');
    });

    await test.step('Check color contrast', async () => {
      // Take screenshot for manual color contrast verification
      await appPage.takeFullPageScreenshot('accessibility-03-color-contrast');
    });
  });
});
