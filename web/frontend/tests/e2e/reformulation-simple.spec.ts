import { test, expect } from '../fixtures/test-helpers';

test.describe('Simple Reformulation Test', () => {
  test('should load a session with reformulation and show reformulation message', async ({ appPage, page }) => {
    await appPage.goto();
    
    // Verify initial state
    await expect(page.locator('[data-testid="app-root"]')).toBeVisible({ timeout: 10000 });
    
    console.log('🚀 Testing reformulation with known session...');
    
    // Navigate directly to a session we know has reformulation data
    const sessionId = 'session_1757519574_c1b7c8c7';
    await page.goto(`http://localhost:3000?sessionId=${sessionId}`);
    
    console.log(`📝 Loading session: ${sessionId}`);
    
    // Wait for session to be loaded (session ID visible in header)
    await expect(page.locator('text=Session:').locator('..').getByText('c1b7c8c7')).toBeVisible({ timeout: 10000 });
    
    // Take a screenshot to see what we have after session loads
    await appPage.takeFullPageScreenshot('session-loaded');
    
    // Look for the reformulation message with longer timeout
    console.log('🔍 Looking for reformulation message...');
    const reformulationMessage = page.locator('[data-testid="reformulation-message"]').first();
    
    // Wait for progress messages to load and reformulation to appear
    const isVisible = await reformulationMessage.isVisible({ timeout: 15000 }).catch(() => false);
    
    if (isVisible) {
      console.log('✅ Reformulation message found!');
      await appPage.takeFullPageScreenshot('reformulation-found');
      
      // Check for the key elements
      await expect(page.locator('text=enhanced your request')).toBeVisible();
      await expect(page.locator('text=Enhanced Request:')).toBeVisible();
      await expect(page.getByRole('button', { name: /Use Enhanced Version/i })).toBeVisible();
      
      console.log('✅ All reformulation elements found!');
    } else {
      console.log('❌ Reformulation message not found');
      
      // Debug: Check what we do have
      const progressDisplay = await page.locator('[data-testid="progress-display"]').isVisible().catch(() => false);
      const messagesContainer = await page.locator('[data-testid="messages-container"]').isVisible().catch(() => false);
      
      console.log(`Progress display visible: ${progressDisplay}`);
      console.log(`Messages container visible: ${messagesContainer}`);
      
      // Take a screenshot of what we see
      await appPage.takeFullPageScreenshot('reformulation-not-found');
      
      throw new Error('Reformulation message not found');
    }
  });
});