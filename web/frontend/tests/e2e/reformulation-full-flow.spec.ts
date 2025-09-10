import { test, expect } from '../fixtures/test-helpers';

test.describe('Complete Reformulation Flow', () => {
  test('should complete full reformulation workflow from start to finish', async ({ appPage, page }) => {
    await appPage.goto();
    
    console.log('🚀 Starting complete reformulation workflow test...');
    
    // Step 1: Enter a prompt
    console.log('📝 Step 1: Entering prompt...');
    const chatInput = page.getByPlaceholder(/describe.*3d.*asset/i);
    await expect(chatInput).toBeVisible({ timeout: 10000 });
    await chatInput.fill('Create a spaceship');
    
    // Step 2: Send the message
    console.log('🚀 Step 2: Sending message...');
    const sendButton = page.getByRole('button', { name: /send/i });
    await sendButton.click();
    
    // Step 3: Wait for user and assistant messages to appear
    console.log('💬 Step 3: Waiting for messages...');
    await expect(page.locator('text=Create a spaceship')).toBeVisible({ timeout: 5000 });
    await expect(page.locator('text=Generating your 3D asset')).toBeVisible({ timeout: 5000 });
    
    // Step 4: Wait for reformulation to complete (this can take up to 45 seconds)
    console.log('⏰ Step 4: Waiting for reformulation to complete...');
    
    // Look for the reformulation message to appear - be patient
    const reformulationMessage = page.locator('[data-testid="reformulation-message"]');
    
    // Wait up to 60 seconds for reformulation
    let reformulationFound = false;
    let attempts = 0;
    const maxAttempts = 60; // 60 seconds
    
    while (!reformulationFound && attempts < maxAttempts) {
      await page.waitForTimeout(1000);
      reformulationFound = await reformulationMessage.isVisible().catch(() => false);
      attempts++;
      
      if (attempts % 10 === 0) {
        console.log(`⏳ Still waiting for reformulation... (${attempts}s)`);
      }
    }
    
    if (!reformulationFound) {
      console.log('❌ Reformulation message not found after 60 seconds');
      await appPage.takeFullPageScreenshot('reformulation-timeout');
      throw new Error('Reformulation did not appear within 60 seconds');
    }
    
    console.log('✅ Reformulation message appeared!');
    await appPage.takeFullPageScreenshot('reformulation-appeared');
    
    // Step 5: Verify reformulation content
    console.log('🔍 Step 5: Verifying reformulation content...');
    
    await expect(page.locator('text=enhanced your request')).toBeVisible();
    await expect(page.locator('text=Enhanced Request:')).toBeVisible();
    await expect(page.locator('text=✅ Improvements:')).toBeVisible();
    await expect(page.locator('text=❓ Questions:')).toBeVisible();
    
    // Step 6: Test "Use Enhanced Version" button
    console.log('✅ Step 6: Testing "Use Enhanced Version" button...');
    const useEnhancedButton = page.getByRole('button', { name: /Use Enhanced Version/i });
    await expect(useEnhancedButton).toBeVisible();
    await useEnhancedButton.click();
    
    await appPage.takeFullPageScreenshot('enhanced-version-clicked');
    
    // Step 7: Verify confirmation and continuation
    console.log('⏳ Step 7: Waiting for confirmation...');
    
    // Look for confirmation message
    await expect(page.locator('text=Request confirmed and processing')).toBeVisible({ timeout: 10000 });
    console.log('✅ Confirmation message appeared');
    
    // Step 8: Verify workflow continues (check for progress updates)
    console.log('🔄 Step 8: Verifying workflow continuation...');
    
    // Wait a bit to see if workflow progresses beyond reformulation
    await page.waitForTimeout(5000);
    await appPage.takeFullPageScreenshot('workflow-continuing');
    
    console.log('🎉 Complete reformulation workflow test completed successfully!');
  });

  test('should test "Customize Further" option', async ({ appPage, page }) => {
    await appPage.goto();
    
    console.log('🔧 Testing "Customize Further" workflow...');
    
    // Create session and wait for reformulation
    const chatInput = page.getByPlaceholder(/describe.*3d.*asset/i);
    await chatInput.fill('Create a chair');
    await page.getByRole('button', { name: /send/i }).click();
    
    // Wait for reformulation with extended timeout
    const reformulationMessage = page.locator('[data-testid="reformulation-message"]');
    await expect(reformulationMessage).toBeVisible({ timeout: 60000 });
    
    // Click "Customize Further"
    const customizeButton = page.getByRole('button', { name: /Customize Further/i });
    await customizeButton.click();
    
    // Verify custom input appears
    const customInput = page.getByPlaceholder(/Describe your specific changes/i);
    await expect(customInput).toBeVisible({ timeout: 5000 });
    
    // Enter custom modifications and submit
    await customInput.fill('Make it a wooden rocking chair with carved details');
    await page.getByRole('button', { name: /Submit Changes/i }).click();
    
    // Verify confirmation
    await expect(page.locator('text=Request confirmed and processing')).toBeVisible({ timeout: 10000 });
    
    console.log('✅ Customize Further workflow completed successfully!');
  });

  test('should test "Use Original" option', async ({ appPage, page }) => {
    await appPage.goto();
    
    console.log('📝 Testing "Use Original" workflow...');
    
    // Create session and wait for reformulation
    const chatInput = page.getByPlaceholder(/describe.*3d.*asset/i);
    await chatInput.fill('Create a table');
    await page.getByRole('button', { name: /send/i }).click();
    
    // Wait for reformulation
    const reformulationMessage = page.locator('[data-testid="reformulation-message"]');
    await expect(reformulationMessage).toBeVisible({ timeout: 60000 });
    
    // Click "Use Original"
    const useOriginalButton = page.getByRole('button', { name: /Use Original/i });
    await useOriginalButton.click();
    
    // Verify confirmation
    await expect(page.locator('text=Request confirmed and processing')).toBeVisible({ timeout: 10000 });
    
    console.log('✅ Use Original workflow completed successfully!');
  });
});