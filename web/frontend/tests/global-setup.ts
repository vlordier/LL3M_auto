import { chromium, FullConfig } from '@playwright/test';

async function globalSetup(config: FullConfig) {
  console.log('🔧 Setting up global test environment...');

  // Create browser instance for setup
  const browser = await chromium.launch();
  const context = await browser.newContext();
  const page = await context.newPage();

  // Wait for services to be ready
  try {
    console.log('⏳ Waiting for frontend to be ready...');
    await page.goto('http://localhost:3000', { waitUntil: 'networkidle' });
    await page.waitForSelector('[data-testid="app-root"]', { timeout: 30000 });
    console.log('✅ Frontend is ready');

    console.log('⏳ Waiting for backend API to be ready...');
    const response = await page.request.get('http://localhost:8000/health');
    if (response.ok()) {
      console.log('✅ Backend API is ready');
    } else {
      console.log('⚠️ Backend API health check failed, but continuing...');
    }
  } catch (error) {
    console.error('❌ Setup failed:', error);
    throw error;
  } finally {
    await browser.close();
  }

  // Set up test data directory
  const fs = await import('fs');
  const path = await import('path');
  const { fileURLToPath } = await import('url');

  const __dirname = path.dirname(fileURLToPath(import.meta.url));
  const testDataDir = path.join(__dirname, 'test-data');
  if (!fs.existsSync(testDataDir)) {
    fs.mkdirSync(testDataDir, { recursive: true });
  }

  // Create mock session data for testing
  const mockSessions = [
    {
      session_id: 'test-session-1',
      prompt: 'Create a wooden chair with curved armrests',
      current_phase: 'completed',
      created_at: new Date().toISOString(),
      traces: [
        {
          trace_id: 'trace-1',
          phase: 'prompt_analysis',
          start_time: new Date().toISOString(),
          end_time: new Date().toISOString(),
          execution_time: 1.5,
          outputs: { analysis: 'Furniture generation request' }
        }
      ],
      artifacts: [
        {
          artifact_id: 'artifact-1',
          artifact_type: 'model',
          relative_path: 'test-chair.gltf',
          size_bytes: 1024000
        }
      ],
      total_duration: 45.2
    }
  ];

  fs.writeFileSync(
    path.join(testDataDir, 'mock-sessions.json'),
    JSON.stringify(mockSessions, null, 2)
  );

  console.log('🚀 Global setup complete!');
}

export default globalSetup;
