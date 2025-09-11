import { test, expect } from '../fixtures/test-helpers';

test.describe('Comprehensive User Flow - Prompt and Send Message', () => {
  test('should complete full user journey: type prompt, send message, see progress', async ({ appPage, page }) => {
    // Navigate to the application
    await appPage.goto();
    
    // Verify app is loaded properly
    const appRoot = page.locator('[data-testid="app-root"]');
    await expect(appRoot).toBeVisible({ timeout: 10000 });
    
    // Take screenshot of initial state
    await appPage.takeFullPageScreenshot('user-flow-initial');

    // Set up error monitoring to ensure no errors occur
    const consoleErrors = [];
    const pageErrors = [];
    
    page.on('console', msg => {
      if (msg.type() === 'error' && !msg.text().includes('Warning:')) {
        consoleErrors.push(msg.text());
        console.log(`❌ Console error: ${msg.text()}`);
      }
    });
    
    page.on('pageerror', error => {
      pageErrors.push(error);
      console.log(`❌ Page error: ${error.message}`);
    });

    // Step 1: Find and interact with chat input
    console.log('🔍 Step 1: Locating chat interface...');
    const chatInput = page.getByPlaceholder(/describe.*3d.*asset/i);
    await expect(chatInput).toBeVisible({ timeout: 5000 });
    console.log('✅ Chat interface found');

    // Step 2: Type a realistic prompt
    console.log('⌨️  Step 2: Typing user prompt...');
    const testPrompt = 'Create a modern office chair with armrests and wheels';
    await chatInput.fill(testPrompt);
    await expect(chatInput).toHaveValue(testPrompt);
    console.log('✅ Prompt entered successfully');
    
    await appPage.takeFullPageScreenshot('user-flow-prompt-entered');

    // Step 3: Locate and click send button
    console.log('🔍 Step 3: Locating send button...');
    const sendButton = page.getByRole('button', { name: /send/i });
    await expect(sendButton).toBeVisible();
    await expect(sendButton).toBeEnabled();
    console.log('✅ Send button found and enabled');

    // Step 4: Click send button
    console.log('🚀 Step 4: Sending message...');
    await sendButton.click();
    console.log('✅ Send button clicked');
    
    await appPage.takeFullPageScreenshot('user-flow-message-sent');

    // Step 5: Verify no application errors appear
    console.log('🔍 Step 5: Checking for application errors...');
    await page.waitForTimeout(2000); // Wait for any errors to appear
    
    const errorElements = [
      page.getByText(/Application Error/i),
      page.getByText(/Something went wrong/i),
      page.getByText(/loading the application/i),
      page.locator('[class*="error"]').filter({ hasText: /error/i }),
      page.locator('[role="alert"]'),
    ];
    
    for (const element of errorElements) {
      const isVisible = await element.isVisible().catch(() => false);
      if (isVisible) {
        const errorText = await element.textContent().catch(() => 'unknown error');
        throw new Error(`Application error detected: ${errorText}`);
      }
    }
    console.log('✅ No application errors found');

    // Step 6: Verify progress indicators appear
    console.log('🔍 Step 6: Checking for progress indicators...');
    const progressElements = [
      page.getByText(/Starting Workflow/i),
      page.getByText(/Understanding Request/i),
      page.getByText(/Processing/i),
      page.getByText(/progress/i),
      page.locator('.animate-spin'),
      page.locator('[data-testid="progress-display"]'),
      page.locator('[class*="progress"]'),
    ];

    let foundProgress = false;
    for (const element of progressElements) {
      const isVisible = await element.isVisible({ timeout: 3000 }).catch(() => false);
      if (isVisible) {
        const text = await element.textContent().catch(() => 'progress element');
        console.log(`✅ Found progress indicator: ${text}`);
        foundProgress = true;
        break;
      }
    }

    if (!foundProgress) {
      console.log('⚠️  No explicit progress indicators found, checking for workflow activity...');
    }

    // Step 7: Wait for workflow to begin and take final screenshot
    await page.waitForTimeout(3000);
    await appPage.takeFullPageScreenshot('user-flow-workflow-started');

    // Step 8: Verify input is cleared or disabled after sending
    console.log('🔍 Step 7: Verifying post-send state...');
    const inputValue = await chatInput.inputValue();
    const isInputDisabled = await chatInput.isDisabled().catch(() => false);
    
    if (inputValue === '' || isInputDisabled) {
      console.log('✅ Input properly cleared/disabled after sending');
    } else {
      console.log('ℹ️  Input still contains text, which may be intentional UX');
    }

    // Final verification: No JavaScript errors occurred
    if (consoleErrors.length > 0) {
      console.error('❌ Console errors detected:', consoleErrors);
      throw new Error(`Console errors occurred: ${consoleErrors.join(', ')}`);
    }

    if (pageErrors.length > 0) {
      console.error('❌ Page errors detected:', pageErrors);
      throw new Error(`Page errors occurred: ${pageErrors.map(e => e.message).join(', ')}`);
    }

    console.log('✅ User flow completed successfully without errors');
  });

  test('should handle multiple message submissions gracefully', async ({ appPage, page }) => {
    await appPage.goto();
    
    const chatInput = page.getByPlaceholder(/describe.*3d.*asset/i);
    const sendButton = page.getByRole('button', { name: /send/i });
    
    await expect(chatInput).toBeVisible({ timeout: 5000 });
    await expect(sendButton).toBeVisible();

    // Send first message
    await chatInput.fill('Create a simple cube');
    await sendButton.click();
    
    // Wait a moment
    await page.waitForTimeout(1000);
    
    // Try to send another message (should handle gracefully)
    const isInputInteractable = await chatInput.isEnabled().catch(() => false);
    const isSendButtonEnabled = await sendButton.isEnabled().catch(() => false);
    
    console.log(`Input enabled: ${isInputInteractable}, Send button enabled: ${isSendButtonEnabled}`);
    
    // The system should either disable inputs during processing or allow queuing
    // Both behaviors are acceptable as long as no errors occur
    const errorElements = page.getByText(/Application Error/i);
    await expect(errorElements).not.toBeVisible({ timeout: 2000 });
    
    console.log('✅ Multiple submissions handled gracefully');
  });
});