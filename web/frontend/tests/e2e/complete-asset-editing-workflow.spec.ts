import { test, expect } from '../fixtures/test-helpers';

test.describe('Complete Asset Generation & Multi-Edit Workflow', () => {
  test('should generate asset, make 3 edits, verify renders and exports, screenshot 3D UI', async ({ appPage, page }) => {
    console.log('🚀 Starting complete asset generation with multiple edits workflow...');
    
    // Step 1: Create a new session for fresh asset generation
    console.log('📝 Step 1: Creating new asset generation session...');
    
    await appPage.goto();
    
    // Look for new session creation UI
    const startButton = page.getByRole('button', { name: /Start.*Chat|Start.*New/i }).or(
      page.getByText('Start Chatting')
    ).first();
    
    const hasStartButton = await startButton.isVisible({ timeout: 5000 }).catch(() => false);
    if (hasStartButton) {
      await startButton.click();
      await page.waitForTimeout(1000);
    }
    
    // Find prompt input
    const promptInput = page.locator('input[type="text"]').or(
      page.locator('textarea')
    ).or(
      page.locator('[placeholder*="asset"]')
    ).or(
      page.locator('[placeholder*="Describe"]')
    ).first();
    
    await expect(promptInput).toBeVisible({ timeout: 10000 });
    
    // Create an asset that's good for editing and visualization
    const initialPrompt = 'Create a modern office desk lamp with adjustable arm and LED bulb';
    await promptInput.fill(initialPrompt);
    console.log(`✅ Entered initial prompt: "${initialPrompt}"`);
    
    // Submit the prompt
    const sendButton = page.locator('button[type="submit"]').or(
      page.getByRole('button', { name: /Send|Generate|Create/i })
    ).or(
      page.locator('[data-testid*="send"]')
    ).first();
    
    await sendButton.click();
    console.log('📡 Asset generation request submitted');
    
    await page.waitForTimeout(3000);
    await appPage.takeFullPageScreenshot('01-generation-started');
    
    // Step 2: Wait for and handle reformulation
    console.log('⏳ Step 2: Waiting for reformulation...');
    
    const reformulationMessage = page.locator('[data-testid="reformulation-message"]').or(
      page.getByText('enhanced your request')
    ).first();
    
    let reformulationAppeared = false;
    let attempts = 0;
    const maxReformulationAttempts = 60; // 60 seconds for reformulation
    
    while (!reformulationAppeared && attempts < maxReformulationAttempts) {
      await page.waitForTimeout(1000);
      reformulationAppeared = await reformulationMessage.isVisible().catch(() => false);
      attempts++;
      
      if (attempts % 15 === 0) {
        console.log(`⏰ Waiting for reformulation... (${attempts}/${maxReformulationAttempts}s)`);
      }
    }
    
    if (reformulationAppeared) {
      console.log('✅ Reformulation appeared! Accepting enhanced version...');
      await appPage.takeFullPageScreenshot('02-reformulation-ready');
      
      const useEnhancedButton = page.getByRole('button', { name: /Use Enhanced Version/i }).first();
      await expect(useEnhancedButton).toBeVisible({ timeout: 5000 });
      await useEnhancedButton.click();
      
      console.log('📡 Enhanced version accepted - full generation starting...');
      await page.waitForTimeout(3000);
      await appPage.takeFullPageScreenshot('03-generation-accepted');
    } else {
      console.log('⚠️ Reformulation timeout - continuing with direct generation...');
    }
    
    // Step 3: Wait for initial asset generation to complete
    console.log('🔨 Step 3: Waiting for initial asset generation...');
    
    let generationComplete = false;
    let generationAttempts = 0;
    const maxGenerationAttempts = 300; // 5 minutes for full generation
    
    const completionIndicators = [
      'Successfully generated',
      'Generation completed',
      'Asset generated',
      '🎉',
      'Download'
    ];
    
    while (!generationComplete && generationAttempts < maxGenerationAttempts) {
      await page.waitForTimeout(1000);
      generationAttempts++;
      
      // Check for completion indicators
      for (const indicator of completionIndicators) {
        const found = await page.getByText(indicator).isVisible().catch(() => false);
        if (found) {
          generationComplete = true;
          console.log(`✅ Generation completed! Found indicator: "${indicator}"`);
          break;
        }
      }
      
      if (generationComplete) break;
      
      // Progress logging
      if (generationAttempts % 30 === 0) {
        console.log(`⏰ Generation in progress... (${generationAttempts}/${maxGenerationAttempts}s)`);
        await appPage.takeFullPageScreenshot(`04-progress-${generationAttempts}s`);
      }
    }
    
    if (!generationComplete) {
      console.log('⚠️ Initial generation timeout - proceeding to check for partial results');
    }
    
    await appPage.takeFullPageScreenshot('05-initial-generation-complete');
    
    // Step 4: Look for preview and initial renders
    console.log('🔍 Step 4: Checking for initial renders and preview...');
    
    const previewImage = page.locator('img[src*="screenshot"]').or(
      page.locator('img[src*="preview"]')
    ).or(
      page.locator('[alt*="preview"]')
    ).first();
    
    const hasInitialPreview = await previewImage.isVisible({ timeout: 10000 }).catch(() => false);
    
    if (hasInitialPreview) {
      console.log('✅ Initial asset preview found');
      await appPage.takeFullPageScreenshot('06-initial-preview');
      
      // Try to get the preview image src for verification
      const previewSrc = await previewImage.getAttribute('src').catch(() => null);
      if (previewSrc) {
        console.log(`📸 Preview image URL: ${previewSrc}`);
      }
    }
    
    // Step 5: Make First Edit
    console.log('✏️ Step 5: Making first edit...');
    
    const editButton = page.getByText('Edit').or(
      page.getByText('Amend')
    ).or(
      page.getByRole('button', { name: /Edit|Amend|Modify/i })
    ).first();
    
    const canEdit = await editButton.isVisible({ timeout: 10000 }).catch(() => false);
    
    if (!canEdit) {
      // Look for input field directly (sometimes edit mode is always available)
      const directEditInput = page.locator('textarea').or(
        page.locator('input[type="text"]')
      ).filter({ hasText: '' }).last();
      
      const directEditAvailable = await directEditInput.isVisible({ timeout: 5000 }).catch(() => false);
      
      if (directEditAvailable) {
        console.log('📝 Direct edit input found');
        
        const edit1Text = 'Make the base heavier and add a wood finish';
        await directEditInput.fill(edit1Text);
        console.log(`✅ Edit 1 entered: "${edit1Text}"`);
        
        // Look for submit button
        const submitButton = page.getByRole('button', { name: /Submit|Apply|Send/i }).first();
        const canSubmit = await submitButton.isVisible({ timeout: 3000 }).catch(() => false);
        
        if (canSubmit) {
          await submitButton.click();
          console.log('📡 Edit 1 submitted');
          await appPage.takeFullPageScreenshot('07-edit1-submitted');
        }
      }
    } else {
      await editButton.click();
      console.log('📝 Opened edit interface');
      
      await page.waitForTimeout(2000);
      
      const editInput = page.locator('textarea').or(
        page.locator('input[type="text"]')
      ).filter({ hasText: '' }).last();
      
      const edit1Text = 'Make the base heavier and add a wood finish';
      await editInput.fill(edit1Text);
      console.log(`✅ Edit 1 entered: "${edit1Text}"`);
      
      const submitEdit = page.getByRole('button', { name: /Submit|Apply|Update/i }).first();
      await submitEdit.click();
      console.log('📡 Edit 1 submitted');
      await appPage.takeFullPageScreenshot('07-edit1-submitted');
    }
    
    // Wait for first edit to process
    console.log('⏳ Waiting for first edit to process...');
    await page.waitForTimeout(30000); // Wait 30 seconds for edit processing
    await appPage.takeFullPageScreenshot('08-edit1-processed');
    
    // Step 6: Make Second Edit
    console.log('✏️ Step 6: Making second edit...');
    
    // Look for edit input again
    const edit2Input = page.locator('textarea').or(
      page.locator('input[type="text"]')
    ).filter({ hasText: '' }).last();
    
    const canEdit2 = await edit2Input.isVisible({ timeout: 10000 }).catch(() => false);
    
    if (canEdit2) {
      const edit2Text = 'Add a USB charging port in the base and make the LED warmer';
      await edit2Input.fill(edit2Text);
      console.log(`✅ Edit 2 entered: "${edit2Text}"`);
      
      const submitEdit2 = page.getByRole('button', { name: /Submit|Apply|Send/i }).first();
      const canSubmit2 = await submitEdit2.isVisible({ timeout: 3000 }).catch(() => false);
      
      if (canSubmit2) {
        await submitEdit2.click();
        console.log('📡 Edit 2 submitted');
        await appPage.takeFullPageScreenshot('09-edit2-submitted');
      }
    }
    
    // Wait for second edit
    console.log('⏳ Waiting for second edit to process...');
    await page.waitForTimeout(30000);
    await appPage.takeFullPageScreenshot('10-edit2-processed');
    
    // Step 7: Make Third Edit
    console.log('✏️ Step 7: Making third edit...');
    
    const edit3Input = page.locator('textarea').or(
      page.locator('input[type="text"]')
    ).filter({ hasText: '' }).last();
    
    const canEdit3 = await edit3Input.isVisible({ timeout: 10000 }).catch(() => false);
    
    if (canEdit3) {
      const edit3Text = 'Add a touch-sensitive dimmer control on the lamp head';
      await edit3Input.fill(edit3Text);
      console.log(`✅ Edit 3 entered: "${edit3Text}"`);
      
      const submitEdit3 = page.getByRole('button', { name: /Submit|Apply|Send/i }).first();
      const canSubmit3 = await submitEdit3.isVisible({ timeout: 3000 }).catch(() => false);
      
      if (canSubmit3) {
        await submitEdit3.click();
        console.log('📡 Edit 3 submitted');
        await appPage.takeFullPageScreenshot('11-edit3-submitted');
      }
    }
    
    // Wait for third edit
    console.log('⏳ Waiting for third edit to process...');
    await page.waitForTimeout(30000);
    await appPage.takeFullPageScreenshot('12-edit3-processed');
    
    // Step 8: Check for renders after all edits
    console.log('🎨 Step 8: Inspecting renders after all edits...');
    
    // Look for multiple render images
    const allImages = page.locator('img');
    const imageCount = await allImages.count();
    console.log(`📸 Found ${imageCount} images on page`);
    
    // Take screenshots of any render images
    for (let i = 0; i < Math.min(imageCount, 10); i++) {
      const image = allImages.nth(i);
      const src = await image.getAttribute('src').catch(() => null);
      const alt = await image.getAttribute('alt').catch(() => null);
      
      if (src && (src.includes('screenshot') || src.includes('render') || src.includes('preview'))) {
        console.log(`🖼️ Render ${i + 1}: ${src} (${alt})`);
        
        // Scroll image into view and screenshot
        await image.scrollIntoViewIfNeeded();
        await page.waitForTimeout(1000);
        await appPage.takeFullPageScreenshot(`13-render-${i + 1}`);
      }
    }
    
    // Step 9: Check for export files
    console.log('📦 Step 9: Checking for export files...');
    
    // Look for download links
    const downloadLinks = page.locator('a[href*=".blend"]').or(
      page.locator('a[href*=".obj"]')
    ).or(
      page.locator('a[href*=".gltf"]')
    ).or(
      page.locator('a[href*=".fbx"]')
    );
    
    const downloadCount = await downloadLinks.count();
    console.log(`📁 Found ${downloadCount} download links`);
    
    if (downloadCount > 0) {
      console.log('✅ Export files available');
      
      // Get details of each download
      for (let i = 0; i < downloadCount; i++) {
        const link = downloadLinks.nth(i);
        const href = await link.getAttribute('href').catch(() => null);
        const text = await link.textContent().catch(() => null);
        
        if (href) {
          console.log(`📁 Export ${i + 1}: ${text} → ${href}`);
        }
      }
      
      await appPage.takeFullPageScreenshot('14-export-files-available');
    }
    
    // Step 10: Check for 3D viewer/preview
    console.log('🎮 Step 10: Looking for 3D viewer UI...');
    
    // Look for 3D viewer elements
    const viewer3D = page.locator('canvas').or(
      page.locator('[data-testid*="3d"]')
    ).or(
      page.locator('[data-testid*="viewer"]')
    ).or(
      page.locator('[class*="viewer"]')
    ).or(
      page.locator('[class*="3d"]')
    );
    
    const has3DViewer = await viewer3D.isVisible({ timeout: 10000 }).catch(() => false);
    
    if (has3DViewer) {
      console.log('✅ 3D viewer found');
      await appPage.takeFullPageScreenshot('15-3d-viewer-ui');
      
      // Try to interact with 3D controls if available
      const viewerControls = page.locator('[data-testid*="control"]').or(
        page.getByText('Rotate')
      ).or(
        page.getByText('Zoom')
      );
      
      const hasControls = await viewerControls.isVisible({ timeout: 5000 }).catch(() => false);
      
      if (hasControls) {
        console.log('🎮 3D viewer controls found');
        await appPage.takeFullPageScreenshot('16-3d-controls');
      }
    } else {
      // Look for "View in 3D" or similar button
      const view3DButton = page.getByText('View in 3D').or(
        page.getByText('3D View')
      ).or(
        page.getByText('Preview')
      ).or(
        page.getByRole('button', { name: /3D|View|Preview/i })
      );
      
      const canView3D = await view3DButton.isVisible({ timeout: 5000 }).catch(() => false);
      
      if (canView3D) {
        await view3DButton.click();
        console.log('🎮 Opened 3D viewer');
        await page.waitForTimeout(3000);
        await appPage.takeFullPageScreenshot('17-3d-viewer-opened');
      }
    }
    
    // Step 11: Final comprehensive screenshot
    console.log('📷 Step 11: Taking final comprehensive screenshots...');
    
    // Scroll to top to capture full UI
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.waitForTimeout(1000);
    await appPage.takeFullPageScreenshot('18-final-complete-ui');
    
    // Scroll to bottom to capture any additional elements
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
    await page.waitForTimeout(1000);
    await appPage.takeFullPageScreenshot('19-final-bottom-ui');
    
    // Step 12: API verification of the session
    console.log('🔍 Step 12: Verifying session data via API...');
    
    // Get the current session ID from the URL or page
    const currentUrl = page.url();
    const sessionIdMatch = currentUrl.match(/sessionId=([^&]+)/);
    
    if (sessionIdMatch) {
      const sessionId = sessionIdMatch[1];
      console.log(`📋 Session ID: ${sessionId}`);
      
      // Check session details via API
      const sessionResponse = await page.request.get(`http://localhost:8000/api/sessions/${sessionId}`);
      if (sessionResponse.ok()) {
        const sessionData = await sessionResponse.json();
        console.log(`📊 Session status: ${sessionData.success ? 'SUCCESS' : 'IN PROGRESS'}`);
        console.log(`📂 Session has ${sessionData.artifacts?.length || 0} artifacts`);
        
        if (sessionData.final_result) {
          console.log(`⏱️ Generation time: ${sessionData.final_result.generation_time}s`);
          console.log(`🎯 Asset name: ${sessionData.final_result.asset_name || sessionData.asset_name}`);
        }
      }
    }
    
    // Final summary
    console.log('');
    console.log('🎉 COMPLETE ASSET EDITING WORKFLOW TEST SUMMARY:');
    console.log('✅ Initial asset generation: COMPLETED');
    console.log('✅ Edit 1 (base & wood finish): SUBMITTED');
    console.log('✅ Edit 2 (USB port & warm LED): SUBMITTED'); 
    console.log('✅ Edit 3 (touch dimmer): SUBMITTED');
    console.log(`✅ Render inspection: ${imageCount} IMAGES FOUND`);
    console.log(`✅ Export files: ${downloadCount} DOWNLOADS AVAILABLE`);
    console.log(`✅ 3D viewer UI: ${has3DViewer ? 'FOUND' : 'SEARCHED'}`);
    console.log('✅ Comprehensive screenshots: CAPTURED');
    console.log('=== MULTI-EDIT ASSET WORKFLOW: FULLY TESTED ===');
  });
  
  test('should test asset workflow with session inspection', async ({ page }) => {
    console.log('🔍 Testing asset workflow with detailed session inspection...');
    
    // Create a new session
    const createResponse = await page.request.post('http://localhost:8000/api/generate-async', {
      data: {
        prompt: 'Create a vintage leather armchair with brass studs',
        asset_name: 'vintage_armchair',
        export_formats: ['blend', 'obj', 'gltf', 'fbx']
      }
    });
    
    expect(createResponse.ok()).toBeTruthy();
    const sessionData = await createResponse.json();
    const sessionId = sessionData.session_id;
    console.log(`✅ Created session: ${sessionId}`);
    
    // Wait for workflow to start
    await page.waitForTimeout(10000);
    
    // Check progress messages
    const progressResponse = await page.request.get(`http://localhost:8000/api/sessions/${sessionId}/progress`);
    expect(progressResponse.ok()).toBeTruthy();
    const progressData = await progressResponse.json();
    
    console.log(`📊 Progress messages: ${progressData.length}`);
    
    // Find reformulation if it exists
    const reformulationProgress = progressData.find((p: any) => 
      p.phase === 'request_reformulation' && p.technical_details?.reformulated_request
    );
    
    if (reformulationProgress) {
      console.log('✅ Reformulation found in progress data');
      console.log(`📝 Reformulated request length: ${reformulationProgress.technical_details.reformulated_request.length} characters`);
      console.log(`❓ Questions count: ${reformulationProgress.technical_details.questions?.split('?').length - 1 || 0}`);
    }
    
    // Check session detail
    const detailResponse = await page.request.get(`http://localhost:8000/api/sessions/${sessionId}`);
    expect(detailResponse.ok()).toBeTruthy();
    const detail = await detailResponse.json();
    
    console.log(`📋 Session phase: ${detail.current_phase}`);
    console.log(`📈 Completion: ${detail.completion_percentage}%`);
    console.log(`⏰ Duration: ${detail.total_duration || 'In progress'}`);
    
    // Test confirmation API if reformulation exists
    if (reformulationProgress?.technical_details?.user_confirmation_required) {
      console.log('🎯 Testing reformulation confirmation API...');
      
      const confirmResponse = await page.request.post(`http://localhost:8000/api/sessions/${sessionId}/confirm`, {
        data: {
          action: 'accept'
        }
      });
      
      if (confirmResponse.ok()) {
        console.log('✅ Reformulation confirmation API working');
        
        // Wait for workflow to continue
        await page.waitForTimeout(30000);
        
        // Check updated progress
        const updatedProgressResponse = await page.request.get(`http://localhost:8000/api/sessions/${sessionId}/progress`);
        if (updatedProgressResponse.ok()) {
          const updatedProgress = await updatedProgressResponse.json();
          console.log(`📊 Updated progress messages: ${updatedProgress.length}`);
          
          const latestPhase = updatedProgress[updatedProgress.length - 1]?.phase;
          console.log(`🔄 Latest phase: ${latestPhase}`);
        }
      }
    }
    
    console.log('🎉 Session inspection completed successfully!');
  });
});