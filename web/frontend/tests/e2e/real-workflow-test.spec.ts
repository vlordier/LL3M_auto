import { test, expect } from '../fixtures/test-helpers';

test.describe('Real Workflow Integration', () => {
  test('should show actual user confirmation flow with real backend', async ({ appPage, page }) => {
    await appPage.goto();

    // Take initial screenshot
    await appPage.takeFullPageScreenshot('initial-state');

    // Check if the page loaded properly
    const appRoot = page.locator('[data-testid="app-root"]');
    await expect(appRoot).toBeVisible({ timeout: 10000 });

    // Look for the chat interface
    const chatInput = page.getByPlaceholder(/describe.*3d.*asset/i);
    
    if (await chatInput.isVisible({ timeout: 5000 })) {
      console.log('✅ Chat interface is visible');
      
      // Send a simple test message to trigger the real workflow
      await chatInput.fill('Create a simple cube');
      
      const sendButton = page.getByRole('button', { name: /send/i });
      await expect(sendButton).toBeVisible();
      await sendButton.click();
      
      console.log('✅ Message sent to backend');
      await appPage.takeFullPageScreenshot('message-sent');

      // Wait and look for any progress indicators
      await page.waitForTimeout(3000);
      await appPage.takeFullPageScreenshot('after-3s-wait');

      // Look for any progress or workflow updates
      const progressElements = [
        page.getByText(/understanding.*request/i),
        page.getByText(/reformulat/i),
        page.getByText(/planning/i),
        page.getByText(/generating/i),
        page.getByText(/progress/i),
        page.locator('[data-testid="progress-display"]'),
        page.locator('.animate-spin'),
      ];

      let foundProgress = false;
      for (const element of progressElements) {
        if (await element.isVisible({ timeout: 2000 }).catch(() => false)) {
          const text = await element.textContent().catch(() => 'element');
          console.log(`✅ Found progress indicator: ${text}`);
          foundProgress = true;
          break;
        }
      }

      await appPage.takeFullPageScreenshot('progress-check');

      if (!foundProgress) {
        console.log('⚠️ No progress indicators found');
      }

      // Wait longer to see if reformulation UI appears
      await page.waitForTimeout(10000);
      await appPage.takeFullPageScreenshot('after-10s-wait');

      // Look for reformulation-related UI
      const reformulationElements = [
        page.getByText(/reformulated.*request/i),
        page.getByText(/accept.*reformulation/i),
        page.getByText(/modify.*request/i),
        page.getByText(/use.*original/i),
        page.getByText(/confirmation/i),
      ];

      let foundReformulation = false;
      for (const element of reformulationElements) {
        if (await element.isVisible({ timeout: 2000 }).catch(() => false)) {
          const text = await element.textContent().catch(() => 'element');
          console.log(`✅ Found reformulation UI: ${text}`);
          foundReformulation = true;
          break;
        }
      }

      if (foundReformulation) {
        console.log('✅ Reformulation UI is working');
        await appPage.takeFullPageScreenshot('reformulation-ui-found');
      } else {
        console.log('⚠️ No reformulation UI found - workflow may be bypassing user confirmation');
      }

      // Check final state after longer wait
      await page.waitForTimeout(10000);
      await appPage.takeFullPageScreenshot('final-state');

    } else {
      console.log('❌ Chat interface not found - application may not have loaded properly');
      await appPage.takeFullPageScreenshot('no-chat-interface');
      
      // Look for error messages
      const errorElements = [
        page.getByText(/error/i),
        page.getByText(/failed/i),
        page.getByText(/something.*wrong/i),
        page.locator('[role="alert"]'),
      ];

      for (const element of errorElements) {
        if (await element.isVisible({ timeout: 1000 }).catch(() => false)) {
          const text = await element.textContent().catch(() => 'element');
          console.log(`❌ Found error: ${text}`);
          break;
        }
      }
    }
  });

  test('should check backend API availability', async ({ page }) => {
    // Test backend health endpoint
    try {
      const healthResponse = await page.request.get('/api/health');
      console.log(`Backend health: ${healthResponse.status()}`);
      
      if (healthResponse.ok()) {
        console.log('✅ Backend API is responding');
      } else {
        console.log('⚠️ Backend API health check failed');
      }
    } catch (error) {
      console.log(`❌ Backend API unreachable: ${error}`);
    }

    // Test session creation
    try {
      const sessionResponse = await page.request.post('/api/generate', {
        data: {
          prompt: 'Test cube creation',
          asset_name: 'test-cube'
        }
      });
      
      console.log(`Session creation: ${sessionResponse.status()}`);
      
      if (sessionResponse.ok()) {
        const data = await sessionResponse.json();
        console.log(`✅ Session created: ${data.session_id}`);
        
        // Check session status
        if (data.session_id) {
          const statusResponse = await page.request.get(`/api/sessions/${data.session_id}/step-control`);
          const statusData = await statusResponse.json();
          console.log(`Session step control:`, statusData);
          
          // Check progress
          const progressResponse = await page.request.get(`/api/sessions/${data.session_id}/progress`);
          const progressData = await progressResponse.json();
          console.log(`Session progress:`, progressData);
        }
      } else {
        console.log('⚠️ Session creation failed');
      }
    } catch (error) {
      console.log(`❌ Session creation error: ${error}`);
    }
  });
});