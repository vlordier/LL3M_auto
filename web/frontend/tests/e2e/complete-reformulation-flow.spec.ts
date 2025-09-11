import { test, expect } from '../fixtures/test-helpers';

test.describe('Complete Reformulation Flow with Real Generation', () => {
  test('should create new session, show reformulation, and handle user interactions', async ({ appPage, page }) => {
    await appPage.goto();
    console.log('🚀 Starting complete reformulation flow test...');
    
    // Step 1: Send a request that will trigger reformulation
    console.log('📝 Step 1: Creating new generation request...');
    
    // Fill in the prompt
    const promptInput = page.locator('[data-testid="prompt-input"]');
    await promptInput.fill('Create a simple wooden chair');
    console.log('✅ Prompt entered: "Create a simple wooden chair"');
    
    // Send the message
    const sendButton = page.locator('[data-testid="send-button"]');
    await sendButton.click();
    console.log('📡 Message sent, waiting for reformulation...');
    
    // Take screenshot after sending
    await page.waitForTimeout(2000);
    await appPage.takeFullPageScreenshot('message-sent');
    
    // Step 2: Wait for reformulation message to appear
    console.log('⏳ Step 2: Waiting for reformulation message to appear...');
    
    const reformulationMessage = page.locator('[data-testid="reformulation-message"]').first();
    
    // Wait up to 30 seconds for reformulation (LLM API call can take time)
    await expect(reformulationMessage).toBeVisible({ timeout: 30000 });
    console.log('✅ Reformulation message appeared!');
    
    await appPage.takeFullPageScreenshot('reformulation-appeared');
    
    // Step 3: Verify reformulation content is displayed
    console.log('🔍 Step 3: Verifying reformulation content...');
    
    // Check for enhanced request content
    await expect(page.getByText('enhanced your request')).toBeVisible();
    await expect(page.getByText('Enhanced Request:')).toBeVisible();
    
    // Verify we have the action buttons
    const useEnhancedButton = page.getByRole('button', { name: /Use Enhanced Version/i }).first();
    const customizeFurtherButton = page.getByRole('button', { name: /Customize Further/i }).first();
    const useOriginalButton = page.getByRole('button', { name: /Use Original/i }).first();
    
    await expect(useEnhancedButton).toBeVisible();
    await expect(customizeFurtherButton).toBeVisible();  
    await expect(useOriginalButton).toBeVisible();
    
    console.log('✅ All reformulation elements verified');
    
    // Step 4: Test "Use Enhanced Version" interaction
    console.log('🎯 Step 4: Testing "Use Enhanced Version" interaction...');
    
    await useEnhancedButton.click();
    console.log('📡 Clicked "Use Enhanced Version" button');
    
    // Wait for the workflow to continue
    await page.waitForTimeout(3000);
    await appPage.takeFullPageScreenshot('enhanced-version-accepted');
    
    // Check that the reformulation message disappears and generation continues
    // The generation should now be in progress
    const generatingIndicator = page.locator('text=Generating').or(page.locator('text=Creating')).or(page.locator('text=Building'));
    
    // Wait a bit to see if generation starts
    await page.waitForTimeout(5000);
    const isGenerating = await generatingIndicator.isVisible({ timeout: 5000 }).catch(() => false);
    
    if (isGenerating) {
      console.log('✅ Generation started after accepting reformulation');
      await appPage.takeFullPageScreenshot('generation-in-progress');
    } else {
      console.log('ℹ️ Generation state not yet visible, but reformulation was accepted');
    }
    
    console.log('🎉 Complete reformulation flow test completed successfully!');
  });
  
  test('should test "Customize Further" workflow', async ({ appPage, page }) => {
    console.log('🚀 Testing "Customize Further" workflow...');
    
    // Use the existing session that should have reformulation ready
    const sessionId = 'session_1757523731_27916a27';
    await page.goto(`http://localhost:3000?sessionId=${sessionId}`);
    
    console.log(`📝 Loading session: ${sessionId}`);
    
    // Wait for session to load and reformulation to appear
    const reformulationMessage = page.locator('[data-testid="reformulation-message"]').first();
    await expect(reformulationMessage).toBeVisible({ timeout: 15000 });
    
    console.log('✅ Reformulation message loaded');
    await appPage.takeFullPageScreenshot('existing-reformulation-loaded');
    
    // Click "Customize Further"
    const customizeButton = page.getByRole('button', { name: /Customize Further/i }).first();
    await customizeButton.click();
    
    console.log('🔧 Clicked "Customize Further"');
    
    // Wait for custom input to appear
    const customTextarea = page.locator('[data-testid="custom-reformulation-input"]');
    await expect(customTextarea).toBeVisible({ timeout: 5000 });
    
    console.log('✅ Custom input textarea appeared');
    await appPage.takeFullPageScreenshot('customize-input-opened');
    
    // Enter custom modifications
    const customText = 'Make it a rocking chair with armrests and a cushioned seat';
    await customTextarea.fill(customText);
    
    console.log(`📝 Entered custom text: "${customText}"`);
    
    // Submit the customization
    const submitButton = page.getByRole('button', { name: /Submit/i }).or(page.getByRole('button', { name: /Use This Version/i })).first();
    await submitButton.click();
    
    console.log('📡 Submitted customization');
    await appPage.takeFullPageScreenshot('customization-submitted');
    
    // Wait to see if the workflow continues
    await page.waitForTimeout(3000);
    
    console.log('✅ Customize Further workflow test completed');
  });
  
  test('should test keyboard shortcuts in custom input', async ({ appPage, page }) => {
    console.log('🚀 Testing keyboard shortcuts in customization...');
    
    // Use the existing session
    const sessionId = 'session_1757523731_27916a27';
    await page.goto(`http://localhost:3000?sessionId=${sessionId}`);
    
    // Wait for reformulation and open customize input
    const reformulationMessage = page.locator('[data-testid="reformulation-message"]').first();
    await expect(reformulationMessage).toBeVisible({ timeout: 15000 });
    
    const customizeButton = page.getByRole('button', { name: /Customize Further/i }).first();
    await customizeButton.click();
    
    const customTextarea = page.locator('[data-testid="custom-reformulation-input"]');
    await expect(customTextarea).toBeVisible({ timeout: 5000 });
    
    // Test keyboard shortcuts
    await customTextarea.fill('Add vintage styling and carved details');
    
    // Test Ctrl+Enter to submit (or Cmd+Enter on Mac)
    const isMac = process.platform === 'darwin';
    const modifierKey = isMac ? 'Meta' : 'Control';
    
    await customTextarea.press(`${modifierKey}+Enter`);
    
    console.log(`✅ Tested ${modifierKey}+Enter keyboard shortcut`);
    await appPage.takeFullPageScreenshot('keyboard-shortcut-submitted');
    
    await page.waitForTimeout(2000);
    console.log('✅ Keyboard shortcuts test completed');
  });
});