import { test, expect } from '../fixtures/test-helpers';

// Tiny 1x1 red PNG (base64)
const RED_PX_BASE64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8Xw8AAscB9tZ+XGAAAAAASUVORK5CYII=';

test.describe('Strawberry Generation Flow', () => {
  test('generates "a strawberry" and shows renders', async ({ appPage }) => {
    await test.step('Open app', async () => {
      await appPage.goto();
      await appPage.takeFullPageScreenshot('strawberry-01-startup');
      await expect(appPage.page.getByText('Welcome to LL3M')).toBeVisible();
    });

    await test.step('Mock generation endpoints', async () => {
      // Force fallback to synchronous generation
      await appPage.page.route('**/api/generate-async', async route => {
        await route.fulfill({ status: 500, body: 'disabled in test' });
      });

      // Provide successful generation response
      await appPage.page.route('**/api/generate', async route => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            success: true,
            session_id: 'strawberry-session',
            asset_path: 'models/strawberry.gltf',
            screenshot_path: 'previews/strawberry.png',
            exported_files: ['models/strawberry.gltf', 'models/strawberry.obj'],
            generation_time: 12.3,
          })
        });
      });

      // Session details for viewer/artifacts
      await appPage.page.route('**/api/sessions/strawberry-session', async route => {
        const now = new Date().toISOString();
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            session_id: 'strawberry-session',
            prompt: 'a strawberry',
            asset_name: 'strawberry',
            current_phase: 'completed',
            completion_percentage: 100,
            success: true,
            created_at: now,
            updated_at: now,
            traces: [],
            code_versions: [],
            artifacts: [
              {
                artifact_id: 'a1',
                artifact_type: 'model',
                file_path: 'models/strawberry.gltf',
                relative_path: 'models/strawberry.gltf',
                created_at: now,
                phase: 'asset_generation',
                version: 1,
                size_bytes: 2048,
                tags: []
              }
            ],
            final_result: {
              success: true,
              cached: false,
              asset_path: 'models/strawberry.gltf',
              screenshot_path: 'previews/strawberry.png',
              generation_time: 12.3,
              exported_files: ['models/strawberry.gltf']
            }
          })
        });
      });

      // Serve preview and asset bytes via /api/assets/*
      await appPage.page.route('**/api/assets/**', async route => {
        const body = Buffer.from(RED_PX_BASE64, 'base64');
        await route.fulfill({
          status: 200,
          contentType: 'image/png',
          body,
        });
      });
    });

    await test.step('Enter prompt and submit', async () => {
      await appPage.sendMessage('a strawberry');
      await appPage.takeFullPageScreenshot('strawberry-02-submitted');
      await expect(appPage.page.locator('[data-testid="messages-container"]').getByText('a strawberry').first()).toBeVisible();
    });

    await test.step('See render preview', async () => {
      // Wait a bit for UI to update and fetch preview
      await appPage.page.waitForTimeout(1000);
      const preview = appPage.page.locator('[data-testid="asset-preview"]');
      await expect(preview).toBeVisible({ timeout: 10000 });
      await appPage.takeFullPageScreenshot('strawberry-03-preview');
    });

    await test.step('Optional: switch to 3D view and capture', async () => {
      const threeButton = appPage.page.getByRole('button', { name: /3d\s*view/i });
      if (await threeButton.isVisible()) {
        await threeButton.click();
        await appPage.page.waitForTimeout(500);
        await appPage.takeFullPageScreenshot('strawberry-04-3d-view');
      }
    });
  });
});
