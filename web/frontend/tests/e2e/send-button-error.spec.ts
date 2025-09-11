import { test, expect } from '../fixtures/test-helpers';

test.describe('Send Button Error Investigation', () => {
  test('should not show application error when sending message (Fixed: React Hooks Rule Violation)', async ({ appPage, page }) => {
    await appPage.goto();

    // Take screenshot of initial state
    await appPage.takeFullPageScreenshot('initial-state');

    // Check if the page loaded properly
    const appRoot = page.locator('[data-testid="app-root"]');
    await expect(appRoot).toBeVisible({ timeout: 10000 });

    // Look for the chat interface
    const chatInput = page.getByPlaceholder(/describe.*3d.*asset/i);
    
    if (await chatInput.isVisible({ timeout: 5000 })) {
      console.log('✅ Chat interface is visible');
      
      // Set up error capture BEFORE any interactions
      const consoleErrors = [];
      const pageErrors = [];
      
      page.on('console', msg => {
        if (msg.type() === 'error') {
          const errorText = msg.text();
          consoleErrors.push(errorText);
          console.log(`❌ Console error: ${errorText}`);
        }
        // Also capture other console messages for debugging
        if (msg.type() === 'warn') {
          console.log(`⚠️ Console warning: ${msg.text()}`);
        }
        if (msg.type() === 'log') {
          const text = msg.text();
          if (text.includes('error') || text.includes('Error')) {
            console.log(`📋 Console log: ${text}`);
          }
        }
      });
      
      page.on('pageerror', error => {
        pageErrors.push(error);
        console.log(`❌ Page error: ${error.message}`);
        console.log(`❌ Stack trace: ${error.stack}`);
      });

      // Listen for uncaught promise rejections
      await page.addInitScript(() => {
        window.addEventListener('unhandledrejection', event => {
          console.error('Unhandled promise rejection:', event.reason);
        });
      });

      // Fill in the input
      await chatInput.fill('Create a simple cube');
      await appPage.takeFullPageScreenshot('input-filled');

      // Look for the send button
      const sendButton = page.getByRole('button', { name: /send/i });
      await expect(sendButton).toBeVisible();
      console.log('✅ Send button is visible');
      
      // Click the send button and immediately check for errors
      await sendButton.click();
      console.log('✅ Send button clicked');
      
      await appPage.takeFullPageScreenshot('after-send-click');
      
      // Wait a moment to see if any error appears
      await page.waitForTimeout(2000);
      
      // Check for application error messages
      const errorElements = [
        page.getByText(/Application Error/i),
        page.getByText(/Something went wrong/i),
        page.getByText(/loading the application/i),
        page.locator('[class*="error"]'),
        page.locator('[role="alert"]'),
      ];
      
      let foundError = false;
      for (const element of errorElements) {
        if (await element.isVisible({ timeout: 1000 }).catch(() => false)) {
          const text = await element.textContent().catch(() => 'element');
          console.log(`❌ Found error: ${text}`);
          foundError = true;
          break;
        }
      }
      
      if (!foundError) {
        console.log('✅ No application errors found');
      }
      
      await appPage.takeFullPageScreenshot('final-check');
      
      // Wait a bit more to catch any delayed errors
      await page.waitForTimeout(3000);
      
      if (consoleErrors.length > 0) {
        console.log(`❌ Console errors found: ${consoleErrors.join(', ')}`);
      } else {
        console.log('✅ No console errors detected');
      }

      // Check if we see progress indicators
      const progressElements = [
        page.getByText(/Starting Workflow/i),
        page.getByText(/Understanding Request/i),
        page.getByText(/progress/i),
        page.locator('.animate-spin'),
        page.locator('[data-testid="progress-display"]'),
      ];

      let foundProgress = false;
      for (const element of progressElements) {
        if (await element.isVisible({ timeout: 2000 }).catch(() => false)) {
          const text = await element.textContent().catch(() => 'progress element');
          console.log(`✅ Found progress indicator: ${text}`);
          foundProgress = true;
          break;
        }
      }

      if (!foundProgress) {
        console.log('⚠️ No progress indicators found');
      }

    } else {
      console.log('❌ Chat interface not found');
      await appPage.takeFullPageScreenshot('no-chat-interface');
    }
  });

  test('should capture network errors when sending', async ({ appPage, page }) => {
    // Monitor network requests
    const requests = [];
    const responses = [];
    const failures = [];

    page.on('request', request => {
      requests.push({
        url: request.url(),
        method: request.method(),
        headers: request.headers(),
      });
    });

    page.on('response', response => {
      responses.push({
        url: response.url(),
        status: response.status(),
        ok: response.ok(),
      });
    });

    page.on('requestfailed', request => {
      failures.push({
        url: request.url(),
        failure: request.failure()?.errorText,
      });
    });

    await appPage.goto();

    const chatInput = page.getByPlaceholder(/describe.*3d.*asset/i);
    if (await chatInput.isVisible({ timeout: 5000 })) {
      await chatInput.fill('Test message');
      
      const sendButton = page.getByRole('button', { name: /send/i });
      await sendButton.click();
      
      // Wait for network activity
      await page.waitForTimeout(5000);

      console.log('📊 Network Analysis:');
      console.log(`Total requests: ${requests.length}`);
      console.log(`Total responses: ${responses.length}`);
      console.log(`Failed requests: ${failures.length}`);

      // Check for failed requests
      if (failures.length > 0) {
        console.log('❌ Failed requests:', failures);
      }

      // Check for error responses
      const errorResponses = responses.filter(r => !r.ok);
      if (errorResponses.length > 0) {
        console.log('❌ Error responses:', errorResponses);
      }

      // Check for generate requests
      const generateRequests = requests.filter(r => r.url.includes('/api/generate'));
      console.log(`🔍 Generate API requests: ${generateRequests.length}`);
      generateRequests.forEach(req => {
        console.log(`- ${req.method} ${req.url}`);
      });
    }
  });
});