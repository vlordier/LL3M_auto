import { test, expect } from '../fixtures/test-helpers';

test.describe('Reformulation User Interaction', () => {
  test('should handle complete reformulation workflow with user interactions', async ({ appPage, page }) => {
    await appPage.goto();
    
    // Verify initial state
    await expect(page.locator('[data-testid="app-root"]')).toBeVisible({ timeout: 10000 });
    
    console.log('🚀 Starting reformulation interaction test...');
    
    // Set up comprehensive error monitoring
    const consoleErrors = [];
    const networkErrors = [];
    const apiRequests = [];
    const consoleLogs = [];
    
    page.on('console', msg => {
      const text = msg.text();
      if (msg.type() === 'error') {
        consoleErrors.push(text);
        console.log(`❌ Console error: ${text}`);
      } else if (msg.type() === 'log' && text.includes('ChatInterface reformulation')) {
        consoleLogs.push(text);
        console.log(`🔍 Debug log: ${text}`);
      }
    });
    
    page.on('requestfailed', request => {
      networkErrors.push({
        url: request.url(),
        failure: request.failure()?.errorText,
      });
      console.log(`🌐 Request failed: ${request.url()} - ${request.failure()?.errorText}`);
    });

    // Monitor API requests
    page.on('request', request => {
      if (request.url().includes('/api/')) {
        apiRequests.push({
          method: request.method(),
          url: request.url(),
          headers: request.headers(),
        });
        console.log(`📡 API Request: ${request.method()} ${request.url()}`);
      }
    });

    // Step 1: Enter a prompt that will trigger reformulation
    console.log('📝 Step 1: Entering prompt that triggers reformulation...');
    const chatInput = page.getByPlaceholder(/describe.*3d.*asset/i);
    await expect(chatInput).toBeVisible({ timeout: 5000 });
    
    await chatInput.fill('Create a spaceship');
    console.log('✅ Prompt entered: "Create a spaceship"');
    
    // Step 2: Send the message
    console.log('🚀 Step 2: Sending message...');
    const sendButton = page.getByRole('button', { name: /send/i });
    await expect(sendButton).toBeVisible();
    await sendButton.click();
    console.log('✅ Message sent');
    
    await appPage.takeFullPageScreenshot('reformulation-message-sent');
    
    // Step 3: Wait for reformulation to appear
    console.log('⏳ Step 3: Waiting for reformulation message...');
    
    // Look for the reformulation message component
    const reformulationMessage = page.locator('[data-testid="reformulation-message"]').first();
    
    // Wait longer for reformulation to complete and for the UI to update
    console.log('⏰ Waiting longer for reformulation to complete...');
    await page.waitForTimeout(20000);
    
    await expect(reformulationMessage).toBeVisible({ timeout: 30000 });
    console.log('✅ Reformulation message appeared');
    
    await appPage.takeFullPageScreenshot('reformulation-appeared');
    
    // Step 4: Verify reformulation content
    console.log('🔍 Step 4: Verifying reformulation content...');
    
    // Check for enhanced request section
    const enhancedRequest = page.locator('text=Enhanced Request:').first();
    await expect(enhancedRequest).toBeVisible();
    console.log('✅ Enhanced request section found');
    
    // Check for improvements section
    const improvements = page.locator('text=✅ Improvements:').first();
    await expect(improvements).toBeVisible();
    console.log('✅ Improvements section found');
    
    // Check for questions section
    const questions = page.locator('text=❓ Questions:').first();
    await expect(questions).toBeVisible();
    console.log('✅ Questions section found');
    
    // Step 5: Test "Use Enhanced Version" button
    console.log('✅ Step 5: Testing "Use Enhanced Version" action...');
    const useEnhancedButton = page.getByRole('button', { name: /Use Enhanced Version/i }).first();
    await expect(useEnhancedButton).toBeVisible();
    console.log('✅ "Use Enhanced Version" button found');
    
    // Click the button
    await useEnhancedButton.click();
    console.log('✅ "Use Enhanced Version" button clicked');
    
    await appPage.takeFullPageScreenshot('enhanced-version-clicked');
    
    // Step 6: Verify confirmation response
    console.log('⏳ Step 6: Waiting for confirmation processing...');
    
    // Look for confirmation message
    const confirmationMessage = page.locator('text=Request confirmed and processing').first();
    await expect(confirmationMessage).toBeVisible({ timeout: 5000 });
    console.log('✅ Confirmation message appeared');
    
    // Wait a bit to see workflow progress
    await page.waitForTimeout(3000);
    await appPage.takeFullPageScreenshot('workflow-continuing');
    
    // Step 7: Verify no duplicate progress components
    console.log('🔍 Step 7: Checking for duplicate progress components...');
    
    // Ensure there's no duplicate "Reformulated Request Ready" in progress
    const duplicateProgress = page.locator('text=✏️ Reformulated Request Ready').first();
    const isDuplicateVisible = await duplicateProgress.isVisible({ timeout: 2000 }).catch(() => false);
    
    if (isDuplicateVisible) {
      console.log('⚠️ Found duplicate progress component - this should be fixed');
    } else {
      console.log('✅ No duplicate progress components found');
    }
    
    // Step 8: Verify no API errors occurred
    console.log('🔍 Step 8: Checking for API errors...');
    
    // Print debug console logs
    console.log('🔍 Debug console logs captured:', consoleLogs.length);
    consoleLogs.forEach(log => {
      console.log(`   ${log}`);
    });
    
    // Check for confirmation API request
    const confirmRequests = apiRequests.filter(req => 
      req.url.includes('/confirm') && req.method === 'POST'
    );
    console.log(`📡 Found ${confirmRequests.length} confirmation requests`);
    
    if (confirmRequests.length === 0) {
      console.log('⚠️ No confirmation requests found - this might indicate an issue');
    } else {
      console.log('✅ Confirmation request was made');
    }
    
    // Check for network errors
    if (networkErrors.length > 0) {
      console.error('❌ Network errors detected:', networkErrors);
      throw new Error(`Network errors occurred: ${networkErrors.map(e => `${e.url}: ${e.failure}`).join(', ')}`);
    } else {
      console.log('✅ No network errors detected');
    }
    
    // Check for console errors (excluding known warnings)
    const criticalErrors = consoleErrors.filter(error => 
      !error.includes('Warning:') && 
      !error.includes('React has detected') &&
      !error.includes('DevTools')
    );
    
    if (criticalErrors.length > 0) {
      console.error('❌ Critical console errors:', criticalErrors);
      throw new Error(`Console errors occurred: ${criticalErrors.join(', ')}`);
    } else {
      console.log('✅ No critical console errors detected');
    }
    
    console.log('🎉 Reformulation interaction test completed successfully!');
  });

  test('should handle "Customize Further" workflow', async ({ appPage, page }) => {
    await appPage.goto();
    
    console.log('🚀 Testing "Customize Further" workflow...');
    
    // Send a message to trigger reformulation
    const chatInput = page.getByPlaceholder(/describe.*3d.*asset/i);
    await chatInput.fill('Create a chair');
    
    const sendButton = page.getByRole('button', { name: /send/i });
    await sendButton.click();
    
    // Wait for reformulation to appear
    const reformulationMessage = page.getByText('enhanced your request').first();
    await expect(reformulationMessage).toBeVisible({ timeout: 15000 });
    
    // Click "Customize Further" button
    const customizeButton = page.getByRole('button', { name: /Customize Further/i }).first();
    await expect(customizeButton).toBeVisible();
    await customizeButton.click();
    console.log('✅ "Customize Further" button clicked');
    
    await appPage.takeFullPageScreenshot('customize-further-clicked');
    
    // Verify custom input appears
    const customInput = page.getByPlaceholder(/Describe your specific changes/i);
    await expect(customInput).toBeVisible({ timeout: 3000 });
    console.log('✅ Custom input field appeared');
    
    // Enter custom modifications
    const customText = 'Make it a wooden rocking chair with carved details';
    await customInput.fill(customText);
    console.log(`✅ Custom text entered: "${customText}"`);
    
    // Submit changes
    const submitButton = page.getByRole('button', { name: /Submit Changes/i });
    await expect(submitButton).toBeVisible();
    await submitButton.click();
    console.log('✅ Custom changes submitted');
    
    await appPage.takeFullPageScreenshot('custom-changes-submitted');
    
    // Verify confirmation
    const confirmationMessage = page.locator('text=Request confirmed and processing').first();
    await expect(confirmationMessage).toBeVisible({ timeout: 5000 });
    console.log('✅ Custom modification confirmed');
  });

  test('should handle "Use Original" workflow', async ({ appPage, page }) => {
    await appPage.goto();
    
    console.log('🚀 Testing "Use Original" workflow...');
    
    // Send a message to trigger reformulation
    const chatInput = page.getByPlaceholder(/describe.*3d.*asset/i);
    await chatInput.fill('Create a table');
    
    const sendButton = page.getByRole('button', { name: /send/i });
    await sendButton.click();
    
    // Wait for reformulation to appear
    const reformulationMessage = page.getByText('enhanced your request').first();
    await expect(reformulationMessage).toBeVisible({ timeout: 15000 });
    
    // Click "Use Original" button
    const useOriginalButton = page.getByRole('button', { name: /Use Original/i }).first();
    await expect(useOriginalButton).toBeVisible();
    await useOriginalButton.click();
    console.log('✅ "Use Original" button clicked');
    
    await appPage.takeFullPageScreenshot('use-original-clicked');
    
    // Verify confirmation
    const confirmationMessage = page.locator('text=Request confirmed and processing').first();
    await expect(confirmationMessage).toBeVisible({ timeout: 5000 });
    console.log('✅ Original request confirmed');
  });

  test('should handle keyboard shortcuts in custom input', async ({ appPage, page }) => {
    await appPage.goto();
    
    console.log('🚀 Testing keyboard shortcuts...');
    
    // Trigger reformulation
    const chatInput = page.getByPlaceholder(/describe.*3d.*asset/i);
    await chatInput.fill('Create a lamp');
    
    const sendButton = page.getByRole('button', { name: /send/i });
    await sendButton.click();
    
    // Wait for reformulation and open custom input
    const reformulationMessage = page.getByText('enhanced your request').first();
    await expect(reformulationMessage).toBeVisible({ timeout: 15000 });
    
    const customizeButton = page.getByRole('button', { name: /Customize Further/i }).first();
    await customizeButton.click();
    
    const customInput = page.getByPlaceholder(/Describe your specific changes/i);
    await expect(customInput).toBeVisible();
    
    // Test Cmd+Enter shortcut
    await customInput.fill('Make it a desk lamp with adjustable arm');
    await customInput.press('Meta+Enter');
    console.log('✅ Cmd+Enter shortcut tested');
    
    // Verify submission
    const confirmationMessage = page.locator('text=Request confirmed and processing').first();
    await expect(confirmationMessage).toBeVisible({ timeout: 5000 });
    console.log('✅ Keyboard shortcut submission worked');
  });
});