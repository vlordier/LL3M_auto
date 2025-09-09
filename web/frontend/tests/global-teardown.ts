import { FullConfig } from '@playwright/test';

async function globalTeardown(config: FullConfig) {
  console.log('🧹 Cleaning up global test environment...');
  
  // Clean up any test data that was created
  const fs = await import('fs');
  const path = await import('path');
  const { fileURLToPath } = await import('url');
  
  const __dirname = path.dirname(fileURLToPath(import.meta.url));
  const testDataDir = path.join(__dirname, 'test-data');
  if (fs.existsSync(testDataDir)) {
    fs.rmSync(testDataDir, { recursive: true, force: true });
    console.log('✅ Test data cleaned up');
  }
  
  console.log('🏁 Global teardown complete!');
}

export default globalTeardown;