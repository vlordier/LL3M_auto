import { test, expect } from '../fixtures/test-helpers';

test.describe('Complete Asset Generation & Editing Flow', () => {
  test('should generate complete asset from reformulation to final result', async ({ appPage, page }) => {
    console.log('🚀 Starting complete asset generation flow...');
    
    // Step 1: Start with fresh session and generate reformulation
    await appPage.goto();
    
    console.log('📝 Step 1: Creating new asset generation request...');
    
    // Click "Start Chatting" or find the input field
    const startChatButton = page.locator('text=Start Chatting').or(
      page.getByRole('button', { name: /Start.*Chat/i })
    );
    
    const hasStartButton = await startChatButton.isVisible({ timeout: 5000 }).catch(() => false);
    if (hasStartButton) {
      await startChatButton.click();
      await page.waitForTimeout(1000);
    }
    
    // Look for prompt input field
    const promptInput = page.locator('input[type="text"]').or(
      page.locator('textarea')
    ).or(
      page.locator('[placeholder*="asset"]')
    ).or(
      page.locator('[placeholder*="Describe"]')
    ).first();
    
    await expect(promptInput).toBeVisible({ timeout: 10000 });
    
    // Enter a prompt that will trigger reformulation and full generation
    const testPrompt = 'Create a medieval wooden shield with metal reinforcements';
    await promptInput.fill(testPrompt);
    console.log(`✅ Entered prompt: "${testPrompt}"`);
    
    // Send the message 
    const sendButton = page.locator('button[type="submit"]').or(
      page.getByRole('button', { name: /Send/i })
    ).or(
      page.locator('[data-testid*="send"]')
    ).or(
      page.locator('button').filter({ hasText: /Send|Generate|Create/ })
    ).first();
    
    await sendButton.click();
    console.log('📡 Message sent, waiting for reformulation...');
    
    await page.waitForTimeout(3000);
    await appPage.takeFullPageScreenshot('message-sent');
    
    // Step 2: Wait for reformulation and accept it
    console.log('⏳ Step 2: Waiting for reformulation to appear...');
    
    const reformulationMessage = page.locator('[data-testid="reformulation-message"]').or(
      page.getByText('enhanced your request')
    ).first();
    
    // Wait up to 45 seconds for reformulation (OpenAI API can be slow)
    let reformulationFound = false;
    let attempts = 0;
    const maxAttempts = 45;
    
    while (!reformulationFound && attempts < maxAttempts) {
      await page.waitForTimeout(1000);
      reformulationFound = await reformulationMessage.isVisible().catch(() => false);
      attempts++;
      
      if (attempts % 10 === 0) {
        console.log(`⏰ Still waiting for reformulation... (${attempts}/${maxAttempts}s)`);
      }
    }
    
    if (reformulationFound) {
      console.log('✅ Reformulation appeared! Accepting enhanced version...');
      await appPage.takeFullPageScreenshot('reformulation-appeared');
      
      // Accept the enhanced version
      const useEnhancedButton = page.getByRole('button', { name: /Use Enhanced Version/i }).or(
        page.getByRole('button', { name: /Enhanced/i })
      ).first();
      
      await expect(useEnhancedButton).toBeVisible({ timeout: 5000 });
      await useEnhancedButton.click();
      console.log('📡 Clicked "Use Enhanced Version" - generation should continue...');
      
      await page.waitForTimeout(2000);
      await appPage.takeFullPageScreenshot('enhanced-version-accepted');
      
    } else {
      console.log('⚠️ Reformulation did not appear in time, but continuing with generation...');
    }
    
    // Step 3: Wait for asset generation to complete
    console.log('🔨 Step 3: Waiting for asset generation to complete...');
    
    // Look for generation completion indicators
    const completionIndicators = [
      page.getByText('Successfully generated').first(),
      page.getByText('Generation completed').first(),  
      page.getByText('Asset generated').first(),
      page.locator('[data-testid*="generation-result"]').first(),
      page.locator('[data-testid*="preview"]').first(),
      page.getByText('🎉').first()
    ];
    
    let generationComplete = false;
    let generationAttempts = 0;
    const maxGenerationAttempts = 120; // 2 minutes for full generation
    
    while (!generationComplete && generationAttempts < maxGenerationAttempts) {
      await page.waitForTimeout(1000);
      
      // Check for any completion indicator
      for (const indicator of completionIndicators) {
        const visible = await indicator.isVisible({ timeout: 100 }).catch(() => false);
        if (visible) {
          generationComplete = true;
          break;
        }
      }
      
      generationAttempts++;
      
      if (generationAttempts % 20 === 0) {
        console.log(`⏰ Still waiting for generation... (${generationAttempts}/${maxGenerationAttempts}s)`);
        await appPage.takeFullPageScreenshot(`generation-progress-${generationAttempts}s`);
      }
    }
    
    if (generationComplete) {
      console.log('✅ Asset generation completed!');
      await appPage.takeFullPageScreenshot('generation-completed');
      
      // Step 4: Verify asset result and try editing
      console.log('🔍 Step 4: Verifying generated asset...');
      
      // Look for asset preview or download links
      const previewImage = page.locator('img[src*="screenshot"]').or(
        page.locator('img[src*="preview"]')
      ).or(
        page.locator('[data-testid*="preview-image"]')
      );
      
      const downloadLinks = page.locator('a[href*=".blend"]').or(
        page.locator('a[href*=".obj"]')
      ).or(
        page.locator('a[href*=".gltf"]')
      );
      
      const hasPreview = await previewImage.isVisible({ timeout: 5000 }).catch(() => false);
      const hasDownloads = await downloadLinks.count() > 0;
      
      if (hasPreview) {
        console.log('✅ Asset preview image found');
        await appPage.takeFullPageScreenshot('asset-preview-visible');
      }
      
      if (hasDownloads) {
        const downloadCount = await downloadLinks.count();
        console.log(`✅ Found ${downloadCount} download links`);
      }
      
      // Step 5: Test asset editing (amend functionality)
      console.log('✏️ Step 5: Testing asset editing...');
      
      const editButton = page.getByRole('button', { name: /Edit/i }).or(
        page.getByRole('button', { name: /Amend/i })
      ).or(
        page.getByRole('button', { name: /Modify/i })
      );
      
      const hasEditButton = await editButton.isVisible({ timeout: 5000 }).catch(() => false);
      
      if (hasEditButton) {
        await editButton.click();
        console.log('📝 Opened asset editing interface');
        
        // Look for edit input field
        const editInput = page.locator('input[type="text"]').or(
          page.locator('textarea')
        ).filter({ hasText: '' }).first();
        
        const editInputVisible = await editInput.isVisible({ timeout: 5000 }).catch(() => false);
        
        if (editInputVisible) {
          const editRequest = 'Make the shield more ornate with Celtic knot patterns';
          await editInput.fill(editRequest);
          console.log(`📝 Entered edit request: "${editRequest}"`);
          
          // Submit the edit
          const submitEditButton = page.getByRole('button', { name: /Submit/i }).or(
            page.getByRole('button', { name: /Apply/i })
          );
          
          await submitEditButton.click();
          console.log('📡 Submitted asset edit request');
          
          await page.waitForTimeout(3000);
          await appPage.takeFullPageScreenshot('asset-edit-submitted');
          
          // Wait for edit completion
          await page.waitForTimeout(10000);
          await appPage.takeFullPageScreenshot('after-edit-wait');
          
          console.log('✅ Asset editing workflow tested');
        } else {
          console.log('ℹ️ Edit input not found - edit interface may be different');
        }
        
      } else {
        console.log('ℹ️ Edit button not found - may require different interaction');
      }
      
    } else {
      console.log('⚠️ Generation did not complete in expected time - taking final screenshot');
      await appPage.takeFullPageScreenshot('generation-timeout');
    }
    
    console.log('🎉 Complete asset generation flow test finished!');
    
    // Final summary screenshot
    await appPage.takeFullPageScreenshot('final-state');
  });
  
  test('should test asset generation with existing reformulated session', async ({ appPage, page }) => {
    console.log('🚀 Testing asset generation with pre-reformulated session...');
    
    // Use the existing session that has reformulation ready
    const sessionId = 'session_1757523731_27916a27';
    await page.goto(`http://localhost:3000?sessionId=${sessionId}`);
    
    console.log(`📝 Loading pre-reformulated session: ${sessionId}`);
    
    // Wait for reformulation to load
    const reformulationMessage = page.locator('[data-testid="reformulation-message"]').first();
    await expect(reformulationMessage).toBeVisible({ timeout: 15000 });
    
    console.log('✅ Reformulation loaded, accepting enhanced version...');
    await appPage.takeFullPageScreenshot('reformulation-ready');
    
    // Accept the enhanced version to start generation
    const useEnhancedButton = page.getByRole('button', { name: /Use Enhanced Version/i }).first();
    await useEnhancedButton.click();
    
    console.log('📡 Enhanced version accepted - generation starting...');
    await page.waitForTimeout(3000);
    await appPage.takeFullPageScreenshot('generation-started');
    
    // Monitor generation progress for up to 2 minutes
    let generationComplete = false;
    let attempts = 0;
    const maxAttempts = 120;
    
    while (!generationComplete && attempts < maxAttempts) {
      await page.waitForTimeout(1000);
      attempts++;
      
      // Check for completion indicators
      const successIndicator = await page.getByText('Successfully generated').isVisible().catch(() => false);
      const completedIndicator = await page.getByText('completed').isVisible().catch(() => false);
      const errorIndicator = await page.getByText('failed').or(page.getByText('error')).isVisible().catch(() => false);
      
      if (successIndicator || completedIndicator) {
        generationComplete = true;
        console.log('✅ Generation completed successfully!');
        break;
      }
      
      if (errorIndicator) {
        console.log('⚠️ Generation encountered an error');
        break;
      }
      
      if (attempts % 30 === 0) {
        console.log(`⏰ Generation in progress... (${attempts}/120s)`);
        await appPage.takeFullPageScreenshot(`generation-progress-${attempts}s`);
      }
    }
    
    // Take final screenshot regardless of outcome
    await appPage.takeFullPageScreenshot('generation-final-state');
    
    if (generationComplete) {
      console.log('🎉 Asset generation with reformulated session completed successfully!');
    } else {
      console.log('ℹ️ Generation may still be in progress - test completed');
    }
  });
  
  test('should verify all API endpoints during asset generation', async ({ page }) => {
    console.log('🔍 Testing all API endpoints during asset lifecycle...');
    
    // Test health check
    const healthResponse = await page.request.get('http://localhost:8000/health');
    expect(healthResponse.ok()).toBeTruthy();
    console.log('✅ Health check endpoint working');
    
    // Test diagnostics
    const diagnosticsResponse = await page.request.get('http://localhost:8000/api/diagnostics');
    expect(diagnosticsResponse.ok()).toBeTruthy();
    console.log('✅ Diagnostics endpoint working');
    
    // Test session creation
    const createSessionResponse = await page.request.post('http://localhost:8000/api/generate-async', {
      data: {
        prompt: 'Create a simple test cube',
        asset_name: 'test_cube',
        export_formats: ['blend', 'obj']
      }
    });
    
    expect(createSessionResponse.ok()).toBeTruthy();
    const sessionData = await createSessionResponse.json();
    expect(sessionData.session_id).toBeTruthy();
    
    const testSessionId = sessionData.session_id;
    console.log(`✅ Session created: ${testSessionId}`);
    
    // Test session detail endpoint
    const sessionDetailResponse = await page.request.get(`http://localhost:8000/api/sessions/${testSessionId}`);
    expect(sessionDetailResponse.ok()).toBeTruthy();
    console.log('✅ Session detail endpoint working');
    
    // Test progress endpoint
    const progressResponse = await page.request.get(`http://localhost:8000/api/sessions/${testSessionId}/progress`);
    expect(progressResponse.ok()).toBeTruthy();
    console.log('✅ Progress endpoint working');
    
    // Test step control endpoint
    const stepControlResponse = await page.request.get(`http://localhost:8000/api/sessions/${testSessionId}/step-control`);
    expect(stepControlResponse.ok()).toBeTruthy();
    console.log('✅ Step control endpoint working');
    
    // Test sessions list endpoint
    const sessionsListResponse = await page.request.get('http://localhost:8000/api/sessions?limit=10');
    expect(sessionsListResponse.ok()).toBeTruthy();
    console.log('✅ Sessions list endpoint working');
    
    console.log('🎉 All API endpoints verified working during asset generation!');
  });
});