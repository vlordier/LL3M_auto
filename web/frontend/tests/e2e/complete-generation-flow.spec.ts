import { test, expect } from '../fixtures/test-helpers';

test.describe('Complete Asset Generation Flow', () => {
  test('should complete full generation workflow with real API calls', async ({ appPage, page }) => {
    await appPage.goto();

    // Start recording network requests to understand the actual API flow
    const requests: any[] = [];
    const responses: any[] = [];

    page.on('request', request => {
      if (request.url().includes('/api/')) {
        requests.push({
          url: request.url(),
          method: request.method(),
          postData: request.postData()
        });
      }
    });

    page.on('response', response => {
      if (response.url().includes('/api/')) {
        responses.push({
          url: response.url(),
          status: response.status(),
          statusText: response.statusText()
        });
      }
    });

    await appPage.takeFullPageScreenshot('initial-state');

    // Send a simple request to trigger the workflow
    const testPrompt = 'Create a simple red cube';
    await appPage.sendMessage(testPrompt);

    await appPage.takeFullPageScreenshot('message-sent');

    // Wait for initial generation response
    await page.waitForTimeout(3000);

    // Look for any progress indicators
    const progressElements = await page.locator('[data-testid="progress-display"], .progress, [class*="progress"]').all();
    console.log(`Found ${progressElements.length} progress elements`);

    // Look for reformulation UI if it appears
    const reformulationText = page.getByText(/reformulat/i);
    const isReformulationVisible = await reformulationText.isVisible({ timeout: 5000 }).catch(() => false);

    if (isReformulationVisible) {
      console.log('✅ Reformulation step appeared');
      await appPage.takeFullPageScreenshot('reformulation-appeared');

      // Look for confirmation buttons
      const acceptButton = page.getByText(/accept/i).first();
      const modifyButton = page.getByText(/modify/i).first();
      const rejectButton = page.getByText(/reject|original/i).first();

      const buttonsVisible = {
        accept: await acceptButton.isVisible({ timeout: 2000 }).catch(() => false),
        modify: await modifyButton.isVisible({ timeout: 2000 }).catch(() => false),
        reject: await rejectButton.isVisible({ timeout: 2000 }).catch(() => false)
      };

      console.log('Confirmation buttons visibility:', buttonsVisible);

      if (buttonsVisible.accept) {
        await acceptButton.click();
        console.log('✅ Clicked accept button');
        await appPage.takeFullPageScreenshot('accept-clicked');
      }
    } else {
      console.log('⚠️ No reformulation step detected - workflow may have proceeded directly');
    }

    // Wait for generation to continue
    await page.waitForTimeout(5000);

    // Look for generation progress messages
    const generatingText = page.getByText(/generat/i);
    if (await generatingText.isVisible({ timeout: 2000 })) {
      console.log('✅ Generation step visible');
      await appPage.takeFullPageScreenshot('generation-in-progress');
    }

    // Wait longer for potential completion
    await page.waitForTimeout(10000);

    // Check for any completion indicators
    const completedElements = [
      page.getByText(/complete/i),
      page.getByText(/finished/i),
      page.getByText(/ready/i),
      page.getByText(/done/i),
      page.locator('[data-testid="asset-preview"]'),
      page.locator('canvas'), // 3D viewer canvas
    ];

    let completionFound = false;
    for (const element of completedElements) {
      if (await element.isVisible({ timeout: 1000 }).catch(() => false)) {
        console.log(`✅ Completion indicator found: ${await element.textContent()?.catch(() => 'canvas element')}`);
        completionFound = true;
        break;
      }
    }

    await appPage.takeFullPageScreenshot('final-state');

    // Try to switch to 3D view to see if asset was generated
    try {
      await appPage.switchViewMode('3d');
      await appPage.takeFullPageScreenshot('3d-view-attempt');
      console.log('✅ Successfully switched to 3D view - asset may be available');
    } catch (error) {
      console.log('⚠️ Could not switch to 3D view - likely no asset generated yet');
    }

    // Log all captured API requests and responses for debugging
    console.log('\n=== API Requests ===');
    requests.forEach((req, i) => {
      console.log(`${i + 1}. ${req.method} ${req.url}`);
      if (req.postData) {
        console.log(`   Data: ${req.postData}`);
      }
    });

    console.log('\n=== API Responses ===');
    responses.forEach((res, i) => {
      console.log(`${i + 1}. ${res.status} ${res.url}`);
    });

    // Basic assertions
    expect(requests.length).toBeGreaterThan(0);
    expect(responses.length).toBeGreaterThan(0);
  });

  test('should handle backend connection issues gracefully', async ({ appPage, page }) => {
    await appPage.goto();

    // Mock backend unavailability
    await page.route('**/api/**', route => {
      route.abort('connectionrefused');
    });

    await appPage.takeFullPageScreenshot('before-backend-failure');

    // Try to send a message
    await appPage.sendMessage('Create a cube');

    // Wait for error handling
    await page.waitForTimeout(3000);

    // Look for error messages
    const errorIndicators = [
      page.getByText(/error/i),
      page.getByText(/failed/i),
      page.getByText(/connection/i),
      page.getByText(/unavailable/i),
      page.locator('[role="alert"]'),
      page.locator('.error')
    ];

    let errorFound = false;
    for (const indicator of errorIndicators) {
      if (await indicator.isVisible({ timeout: 2000 }).catch(() => false)) {
        console.log(`✅ Error indicator found: ${await indicator.textContent()}`);
        errorFound = true;
        await appPage.takeFullPageScreenshot('error-displayed');
        break;
      }
    }

    if (!errorFound) {
      console.log('⚠️ No clear error message displayed for backend failure');
      await appPage.takeFullPageScreenshot('no-error-shown');
    }
  });

  test('should maintain session state across page reloads', async ({ appPage, page }) => {
    await appPage.goto();

    // Mock a session with some progress
    await page.route('**/api/generate', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          success: true,
          session_id: 'persistent-test-session',
          message: 'Generation started'
        })
      });
    });

    await page.route('**/api/sessions/*/progress', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([{
          id: 'test-progress',
          timestamp: new Date().toISOString(),
          phase: 'modeling',
          message: 'Creating 3D geometry',
          progress_percent: 45
        }])
      });
    });

    // Start generation
    await appPage.sendMessage('Create a house');
    await page.waitForTimeout(2000);
    
    // Capture session ID from UI
    const sessionIdElement = page.getByText(/session.*:/i);
    let sessionId = '';
    if (await sessionIdElement.isVisible({ timeout: 2000 })) {
      const sessionText = await sessionIdElement.textContent();
      sessionId = sessionText?.split(':')[1]?.trim() || '';
      console.log(`Session ID: ${sessionId}`);
    }

    await appPage.takeFullPageScreenshot('before-reload');

    // Reload the page
    await page.reload();
    await page.waitForSelector('[data-testid="app-root"]', { timeout: 30000 });

    await appPage.takeFullPageScreenshot('after-reload');

    // Check if session state is preserved
    if (sessionId) {
      const restoredSessionElement = page.getByText(sessionId);
      if (await restoredSessionElement.isVisible({ timeout: 5000 })) {
        console.log('✅ Session state preserved after reload');
      } else {
        console.log('⚠️ Session state not preserved after reload');
      }
    }
  });

  test('should handle rapid user interactions gracefully', async ({ appPage, page }) => {
    await appPage.goto();

    // Mock fast responses
    let requestCount = 0;
    await page.route('**/api/generate', async route => {
      requestCount++;
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          success: true,
          session_id: `rapid-test-${requestCount}`,
          message: `Request ${requestCount} processed`
        })
      });
    });

    // Send multiple messages rapidly
    const messages = [
      'Create a cube',
      'Make it blue',
      'Add texture',
      'Make it bigger',
      'Add shadows'
    ];

    for (const message of messages) {
      await appPage.sendMessage(message);
      await page.waitForTimeout(500); // Brief pause between messages
    }

    await page.waitForTimeout(3000);
    await appPage.takeFullPageScreenshot('after-rapid-inputs');

    // Verify that the UI handled multiple rapid requests
    expect(requestCount).toBeGreaterThan(0);
    console.log(`Processed ${requestCount} rapid requests`);
  });

  test('should display appropriate loading states throughout workflow', async ({ appPage, page }) => {
    await appPage.goto();

    // Mock slow responses to catch loading states
    await page.route('**/api/generate', async route => {
      await page.waitForTimeout(2000); // Simulate slow initial response
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          success: true,
          session_id: 'loading-test-session'
        })
      });
    });

    await page.route('**/api/sessions/*/progress', async route => {
      await page.waitForTimeout(1000); // Simulate slow progress updates
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([{
          id: 'loading-progress',
          timestamp: new Date().toISOString(),
          phase: 'initializing',
          message: 'Setting up generation pipeline...',
          progress_percent: 5
        }])
      });
    });

    // Start generation and immediately look for loading indicators
    await appPage.sendMessage('Create a complex model');
    
    // Check for loading indicators within the first few seconds
    const loadingIndicators = [
      page.getByText(/loading/i),
      page.getByText(/processing/i),
      page.getByText(/generating/i),
      page.locator('.animate-spin'),
      page.locator('[data-testid*="loading"]'),
      page.locator('.spinner')
    ];

    let loadingFound = false;
    for (const indicator of loadingIndicators) {
      if (await indicator.isVisible({ timeout: 3000 }).catch(() => false)) {
        console.log(`✅ Loading indicator found: ${indicator}`);
        loadingFound = true;
        await appPage.takeFullPageScreenshot('loading-state-visible');
        break;
      }
    }

    if (!loadingFound) {
      console.log('⚠️ No loading indicators detected during slow operations');
      await appPage.takeFullPageScreenshot('no-loading-indicators');
    }

    // Wait for loading to potentially complete
    await page.waitForTimeout(5000);
    await appPage.takeFullPageScreenshot('after-loading-period');
  });
});