import { test, expect } from '../fixtures/test-helpers';

test.describe('Debug Session Loading', () => {
  test('should trace session loading step by step', async ({ appPage, page }) => {
    await appPage.goto();
    
    console.log('🔍 Step 1: Testing session loading with URL parameter...');
    
    // Enable console logging in the browser
    page.on('console', msg => {
      if (msg.text().includes('Reformulation') || msg.text().includes('Loading') || msg.text().includes('Session')) {
        console.log('📋 Browser Console:', msg.text());
      }
    });
    
    // Navigate to session URL
    const sessionId = 'session_1757519574_c1b7c8c7';
    await page.goto(`http://localhost:3000?sessionId=${sessionId}`);
    
    console.log('🔍 Step 2: Waiting for session to load...');
    
    // Wait for session to load - look for current session display
    await expect(page.locator('text=c1b7c8c7')).toBeVisible({ timeout: 10000 });
    console.log('✅ Session ID visible in UI');
    
    // Take screenshot after session loads
    await page.waitForTimeout(2000);
    await appPage.takeFullPageScreenshot('session-loaded-step2');
    
    console.log('🔍 Step 3: Checking if progress messages are loaded...');
    
    // Wait a bit more for progress messages to load
    await page.waitForTimeout(5000);
    await appPage.takeFullPageScreenshot('after-5s-wait');
    
    console.log('🔍 Step 4: Looking for reformulation message...');
    
    // Check if reformulation message appears
    const reformulationMessage = page.locator('[data-testid="reformulation-message"]');
    const isVisible = await reformulationMessage.isVisible({ timeout: 1000 }).catch(() => false);
    
    if (isVisible) {
      console.log('✅ Reformulation message found!');
      await appPage.takeFullPageScreenshot('reformulation-found');
    } else {
      console.log('❌ Reformulation message not found');
      await appPage.takeFullPageScreenshot('reformulation-not-found');
      
      // Check what we do have
      const messages = await page.locator('[data-testid="messages-container"] > div').count();
      console.log(`Messages in container: ${messages}`);
      
      // Check if progress display is shown instead
      const progressDisplay = await page.locator('[data-testid="progress-display"]').isVisible().catch(() => false);
      console.log(`Progress display visible: ${progressDisplay}`);
    }
  });
});