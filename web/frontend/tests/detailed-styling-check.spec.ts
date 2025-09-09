import { test, expect } from '@playwright/test';

test('detailed styling diagnosis', async ({ page }) => {
  // Capture all network requests
  const requests: any[] = [];
  const responses: any[] = [];

  page.on('request', request => {
    requests.push({
      url: request.url(),
      method: request.method(),
      resourceType: request.resourceType()
    });
  });

  page.on('response', response => {
    responses.push({
      url: response.url(),
      status: response.status(),
      contentType: response.headers()['content-type']
    });
  });

  // Navigate to app and wait for everything to load
  await page.goto('http://localhost:3000');
  await page.waitForLoadState('networkidle');
  await page.waitForTimeout(3000);

  // Take screenshot
  await page.screenshot({
    path: 'test-results/screenshots/detailed-styling-check.png',
    fullPage: true
  });

  console.log('=== DETAILED STYLING DIAGNOSIS ===');

  // Check if CSS files are being loaded
  const cssRequests = requests.filter(req =>
    req.resourceType === 'stylesheet' ||
    req.url.includes('.css') ||
    req.url.includes('tailwind')
  );

  console.log('CSS Requests:', cssRequests);

  // Check for any failed CSS requests
  const failedCssRequests = responses.filter(resp =>
    (resp.url.includes('.css') || resp.contentType?.includes('text/css')) &&
    resp.status >= 400
  );

  console.log('Failed CSS Requests:', failedCssRequests);

  // Check if main elements have the expected Tailwind classes
  const elementsToCheck = [
    { selector: 'body', expectedClasses: ['bg-gray-50'] },
    { selector: '[data-testid="app-root"]', expectedClasses: ['flex-1', 'flex', 'flex-col'] },
    { selector: 'h1', expectedClasses: ['text-lg', 'font-semibold'] },
    { selector: 'button', expectedClasses: ['px-4', 'py-2', 'rounded'] }
  ];

  for (const element of elementsToCheck) {
    const el = page.locator(element.selector).first();
    if (await el.isVisible()) {
      const className = await el.getAttribute('class');
      console.log(`${element.selector} classes:`, className);

      // Check if expected Tailwind classes are present
      const hasExpectedClasses = element.expectedClasses.some(cls =>
        className?.includes(cls)
      );
      console.log(`  Has expected Tailwind classes:`, hasExpectedClasses);
    }
  }

  // Check computed styles to see if Tailwind is actually working
  const welcomeText = page.getByText('Welcome to LL3M');
  if (await welcomeText.isVisible()) {
    const styles = await welcomeText.evaluate(el => {
      const computed = getComputedStyle(el);
      return {
        fontSize: computed.fontSize,
        fontWeight: computed.fontWeight,
        color: computed.color,
        margin: computed.margin,
        padding: computed.padding,
        backgroundColor: computed.backgroundColor
      };
    });
    console.log('Welcome text computed styles:', styles);
  }

  // Check if we can find any Tailwind-generated CSS rules
  const hasStyles = await page.evaluate(() => {
    const stylesheets = Array.from(document.styleSheets);
    let foundTailwind = false;
    let totalRules = 0;

    for (const sheet of stylesheets) {
      try {
        const rules = Array.from(sheet.cssRules);
        totalRules += rules.length;

        // Look for typical Tailwind patterns
        const tailwindPatterns = rules.some(rule =>
          rule.cssText.includes('--tw-') ||
          rule.cssText.includes('.text-') ||
          rule.cssText.includes('.bg-') ||
          rule.cssText.includes('.flex')
        );

        if (tailwindPatterns) foundTailwind = true;
      } catch (e) {
        // CORS or other issues accessing stylesheet
      }
    }

    return { foundTailwind, totalRules, totalSheets: stylesheets.length };
  });

  console.log('CSS Analysis:', hasStyles);

  // Check for any error messages in the page
  const errorElements = await page.locator('[class*="error"], [role="alert"], .text-red').all();
  console.log('Error elements found:', errorElements.length);

  if (errorElements.length > 0) {
    for (let i = 0; i < errorElements.length; i++) {
      const text = await errorElements[i].textContent();
      console.log(`Error ${i + 1}:`, text);
    }
  }
});
