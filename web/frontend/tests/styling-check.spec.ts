import { test, expect } from '@playwright/test';

test('check current styling and layout', async ({ page }) => {
  // Navigate to the app
  await page.goto('http://localhost:3000');
  
  // Wait for app to load
  await page.waitForTimeout(3000);
  
  // Take full page screenshot to see current state
  await page.screenshot({ 
    path: 'test-results/screenshots/current-styling-state.png', 
    fullPage: true 
  });
  
  // Check for any console errors related to CSS/styling
  const logs: string[] = [];
  const errors: string[] = [];
  
  page.on('console', msg => {
    if (msg.type() === 'error' && (msg.text().includes('CSS') || msg.text().includes('style'))) {
      errors.push(`Styling Error: ${msg.text()}`);
    }
    logs.push(`${msg.type()}: ${msg.text()}`);
  });
  
  // Wait a bit more to catch any lazy-loaded styles
  await page.waitForTimeout(2000);
  
  // Check for specific styling elements
  const welcomeSection = page.locator('[data-testid="welcome-section"]');
  const sidebar = page.locator('[data-testid="sidebar"]');
  const chatInput = page.getByPlaceholder('Describe the 3D asset you want to create...');
  
  // Log what we see
  console.log('=== STYLING CHECK ===');
  
  // Check if Tailwind classes are being applied
  const bodyClasses = await page.locator('body').getAttribute('class');
  console.log('Body classes:', bodyClasses);
  
  // Check for any unstyled elements
  const unstyledElements = await page.locator('[class=""], [style=""]').count();
  console.log('Potentially unstyled elements:', unstyledElements);
  
  // Check if main layout is working
  const appRoot = page.locator('[data-testid="app-root"]');
  if (await appRoot.isVisible()) {
    const bounds = await appRoot.boundingBox();
    console.log('App root dimensions:', bounds);
  }
  
  // Take screenshot of specific sections
  if (await chatInput.isVisible()) {
    await chatInput.screenshot({ path: 'test-results/screenshots/chat-input-styling.png' });
  }
  
  // Print any styling errors
  if (errors.length > 0) {
    console.log('=== STYLING ERRORS ===');
    errors.forEach(error => console.log(error));
  }
  
  // Check computed styles of key elements
  const welcomeText = page.getByText('Welcome to LL3M');
  if (await welcomeText.isVisible()) {
    const fontSize = await welcomeText.evaluate(el => getComputedStyle(el).fontSize);
    const color = await welcomeText.evaluate(el => getComputedStyle(el).color);
    console.log('Welcome text styles - fontSize:', fontSize, 'color:', color);
  }
});