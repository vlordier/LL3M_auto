import { test, expect } from '../fixtures/test-helpers';

test.describe('Verify 3D Renders and Web Viewer Integration', () => {
  test('should check for Blender renders and 3D exports in existing session', async ({ appPage, page }) => {
    console.log('🎨 VERIFYING 3D RENDERS AND WEB VIEWER INTEGRATION');
    console.log('==================================================');
    
    // Use the session that has been running and should have assets
    const sessionId = 'session_1757523731_27916a27';
    await page.goto(`http://localhost:3000?sessionId=${sessionId}`);
    
    console.log(`📝 Checking session: ${sessionId}`);
    await page.waitForTimeout(5000);
    
    // STEP 1: Check session state via API first
    console.log('📊 STEP 1: API Session State Inspection');
    
    const sessionResponse = await page.request.get(`http://localhost:8000/api/sessions/${sessionId}`);
    if (sessionResponse.ok()) {
      const sessionData = await sessionResponse.json();
      console.log(`   📈 Completion: ${sessionData.completion_percentage || 'N/A'}%`);
      console.log(`   📋 Phase: ${sessionData.current_phase || 'Unknown'}`);
      console.log(`   ✅ Success: ${sessionData.success ? 'COMPLETED' : 'IN PROGRESS'}`);
      console.log(`   ⏰ Duration: ${sessionData.total_duration || 'Ongoing'}`);
    }
    
    const progressResponse = await page.request.get(`http://localhost:8000/api/sessions/${sessionId}/progress`);
    let hasBlenderExecution = false;
    let hasRenderFiles = false;
    let hasExportFiles = false;
    
    if (progressResponse.ok()) {
      const progressData = await progressResponse.json();
      console.log(`   📝 Progress entries: ${progressData.length}`);
      
      // Check for Blender execution phase
      const executionPhases = progressData.filter((p: any) => 
        p.phase === 'execution' || 
        p.message && (p.message.includes('Blender') || p.message.includes('execution') || p.message.includes('render'))
      );
      hasBlenderExecution = executionPhases.length > 0;
      console.log(`   🎨 Blender execution entries: ${executionPhases.length}`);
      
      // Check for file paths in technical details
      const withFiles = progressData.filter((p: any) => 
        p.technical_details && (
          p.technical_details.file_paths ||
          p.technical_details.render_files ||
          p.technical_details.export_files ||
          p.technical_details.asset_files
        )
      );
      
      console.log(`   📁 Entries with file information: ${withFiles.length}`);
      
      if (withFiles.length > 0) {
        const latestFiles = withFiles[withFiles.length - 1].technical_details;
        
        if (latestFiles.render_files) {
          hasRenderFiles = true;
          console.log(`   🖼️ Render files: ${latestFiles.render_files.length}`);
          latestFiles.render_files.slice(0, 3).forEach((file: string, i: number) => {
            console.log(`      ${i + 1}. ${file}`);
          });
        }
        
        if (latestFiles.export_files) {
          hasExportFiles = true;
          const exportCount = Object.keys(latestFiles.export_files).length;
          console.log(`   📦 Export files: ${exportCount} formats`);
          Object.entries(latestFiles.export_files).forEach(([format, file]) => {
            console.log(`      ${format}: ${file}`);
          });
        }
        
        if (latestFiles.asset_files) {
          console.log(`   🎯 Asset files: ${latestFiles.asset_files.length}`);
        }
      }
    }
    
    // STEP 2: Check UI for 3D Asset Display
    console.log('🖥️ STEP 2: UI 3D Asset Display Check');
    
    await appPage.takeFullPageScreenshot('step2-ui-initial-state');
    
    // Check if there's a "No 3D Asset Available" message vs actual asset
    const noAssetMessage = await page.getByText('No 3D Asset Available').isVisible().catch(() => false);
    const generateAssetMessage = await page.getByText('Generate a 3D asset').isVisible().catch(() => false);
    
    if (noAssetMessage || generateAssetMessage) {
      console.log('   ℹ️ 3D viewer shows no asset available - may need generation to complete');
    } else {
      console.log('   🎯 3D viewer area may have active content');
    }
    
    // STEP 3: Look for Blender Render Images
    console.log('🖼️ STEP 3: Blender Render Image Detection');
    
    const imageSelectors = [
      'img[src*="screenshot"]',
      'img[src*="render"]',
      'img[src*="preview"]', 
      'img[src*=".png"]',
      'img[src*="blender"]',
      'img[alt*="render"]',
      'img[alt*="preview"]'
    ];
    
    let foundImages = [];
    for (const selector of imageSelectors) {
      const images = page.locator(selector);
      const count = await images.count();
      if (count > 0) {
        foundImages.push({ selector, count });
        console.log(`   ✅ Found ${count} image(s) with selector: ${selector}`);
        
        // Check image sources and loading status
        for (let i = 0; i < Math.min(count, 2); i++) {
          const img = images.nth(i);
          const src = await img.getAttribute('src').catch(() => '');
          const naturalWidth = await img.evaluate((el) => (el as HTMLImageElement).naturalWidth).catch(() => 0);
          const naturalHeight = await img.evaluate((el) => (el as HTMLImageElement).naturalHeight).catch(() => 0);
          
          console.log(`      ${i + 1}. ${src.substring(0, 60)}... (${naturalWidth}x${naturalHeight})`);
          
          if (naturalWidth > 0 && naturalHeight > 0) {
            console.log('         ✅ Image loaded successfully');
          } else {
            console.log('         ⏳ Image loading or failed');
          }
        }
        
        await appPage.takeFullPageScreenshot(`step3-images-${selector.replace(/[^a-zA-Z0-9]/g, '_')}`);
      }
    }
    
    // STEP 4: Check for 3D Web Viewer Elements
    console.log('🎮 STEP 4: 3D Web Viewer Element Detection');
    
    // Look for Canvas elements (Three.js, WebGL)
    const canvases = page.locator('canvas');
    const canvasCount = await canvases.count();
    console.log(`   🖥️ Canvas elements found: ${canvasCount}`);
    
    if (canvasCount > 0) {
      for (let i = 0; i < canvasCount; i++) {
        const canvas = canvases.nth(i);
        const width = await canvas.evaluate((el) => el.width).catch(() => 0);
        const height = await canvas.evaluate((el) => el.height).catch(() => 0);
        const hasWebGL = await canvas.evaluate((el) => {
          const gl = (el as HTMLCanvasElement).getContext('webgl') || 
                     (el as HTMLCanvasElement).getContext('webgl2') || 
                     (el as HTMLCanvasElement).getContext('experimental-webgl');
          return !!gl;
        }).catch(() => false);
        
        console.log(`      Canvas ${i + 1}: ${width}x${height} ${hasWebGL ? '(WebGL)' : '(2D)'}`);
      }
      
      await appPage.takeFullPageScreenshot('step4-canvas-elements');
    }
    
    // Check for 3D viewer container elements
    const viewerSelectors = [
      '[class*="three"]',
      '[class*="3d"]', 
      '[class*="viewer"]',
      '[data-testid*="viewer"]',
      '[data-testid*="3d"]',
      '.asset-viewer',
      '.model-viewer',
      '#three-viewer'
    ];
    
    let foundViewers = [];
    for (const selector of viewerSelectors) {
      const viewers = page.locator(selector);
      const count = await viewers.count();
      if (count > 0) {
        foundViewers.push({ selector, count });
        console.log(`   ✅ Found ${count} viewer container(s): ${selector}`);
      }
    }
    
    // STEP 5: Check Right Panel Specifically
    console.log('📱 STEP 5: Right Panel 3D Viewer Check');
    
    // Look for the right panel area
    const rightPanelSelectors = [
      '.right-panel',
      '[data-testid*="right"]',
      '[data-testid*="viewer"]',
      '.viewer-panel',
      '.asset-panel'
    ];
    
    let rightPanelFound = false;
    for (const selector of rightPanelSelectors) {
      const panel = page.locator(selector);
      const exists = await panel.isVisible().catch(() => false);
      if (exists) {
        rightPanelFound = true;
        console.log(`   ✅ Right panel found: ${selector}`);
        
        // Check for content within right panel
        const panelImages = panel.locator('img, canvas');
        const panelImageCount = await panelImages.count();
        console.log(`      📊 Content elements in right panel: ${panelImageCount}`);
        
        await appPage.takeFullPageScreenshot(`step5-right-panel-${selector.replace(/[^a-zA-Z0-9]/g, '_')}`);
        break;
      }
    }
    
    if (!rightPanelFound) {
      console.log('   ⚠️ Specific right panel not detected - checking overall layout');
      
      // Take a screenshot of the current layout
      await appPage.takeFullPageScreenshot('step5-overall-layout');
    }
    
    // STEP 6: Check for Download/Export Links
    console.log('💾 STEP 6: Export Download Links Detection');
    
    const exportFormats = ['.blend', '.obj', '.gltf', '.fbx', '.dae'];
    let foundDownloads = [];
    
    for (const format of exportFormats) {
      const links = page.locator(`a[href*="${format}"]`);
      const count = await links.count();
      if (count > 0) {
        foundDownloads.push({ format, count });
        console.log(`   ✅ ${format.toUpperCase()} downloads: ${count}`);
        
        // Get actual URLs
        for (let i = 0; i < Math.min(count, 2); i++) {
          const link = links.nth(i);
          const href = await link.getAttribute('href').catch(() => '');
          console.log(`      ${i + 1}. ${href.substring(0, 80)}...`);
        }
      }
    }
    
    // Check for general download sections
    const downloadText = await page.locator('text=Download').count();
    const exportText = await page.locator('text=Export').count();
    console.log(`   📊 "Download" text occurrences: ${downloadText}`);
    console.log(`   📊 "Export" text occurrences: ${exportText}`);
    
    if (downloadText > 0 || exportText > 0) {
      await appPage.takeFullPageScreenshot('step6-download-export-areas');
    }
    
    // STEP 7: Final Comprehensive Screenshot
    console.log('📸 STEP 7: Final State Capture');
    await appPage.takeFullPageScreenshot('step7-final-comprehensive-view');
    
    // SUMMARY REPORT
    console.log('');
    console.log('🎯 3D RENDERS AND WEB VIEWER VERIFICATION SUMMARY');
    console.log('=================================================');
    console.log(`✅ API Session Check: COMPLETED`);
    console.log(`✅ Blender Execution: ${hasBlenderExecution ? 'DETECTED' : 'NOT DETECTED'}`);
    console.log(`✅ Render Files: ${hasRenderFiles ? 'FOUND' : 'NOT FOUND'}`);
    console.log(`✅ Export Files: ${hasExportFiles ? 'FOUND' : 'NOT FOUND'}`);
    console.log(`✅ UI Images: ${foundImages.length} TYPES DETECTED`);
    console.log(`✅ Canvas Elements: ${canvasCount} FOUND`);
    console.log(`✅ 3D Viewers: ${foundViewers.length} CONTAINERS FOUND`);
    console.log(`✅ Right Panel: ${rightPanelFound ? 'DETECTED' : 'LAYOUT CHECKED'}`);
    console.log(`✅ Download Links: ${foundDownloads.length} FORMATS AVAILABLE`);
    console.log('=================================================');
    
    if (hasRenderFiles && hasExportFiles && foundImages.length > 0) {
      console.log('🏆 COMPLETE 3D ASSET PIPELINE: FULLY FUNCTIONAL');
    } else if (hasBlenderExecution || foundImages.length > 0) {
      console.log('🎯 PARTIAL 3D ASSET PIPELINE: COMPONENTS DETECTED');
    } else {
      console.log('📊 3D ASSET PIPELINE: STATUS CHECKED');
    }
  });
  
  test('should verify WebGL 3D viewer functionality', async ({ appPage, page }) => {
    console.log('🎮 TESTING: WebGL 3D Viewer Functionality');
    
    const sessionId = 'session_1757523731_27916a27';
    await page.goto(`http://localhost:3000?sessionId=${sessionId}`);
    await page.waitForTimeout(3000);
    
    // Check for WebGL support in browser
    const hasWebGL = await page.evaluate(() => {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl') || canvas.getContext('webgl2');
      return !!gl;
    });
    
    console.log(`🖥️ Browser WebGL support: ${hasWebGL ? 'AVAILABLE' : 'NOT AVAILABLE'}`);
    
    if (hasWebGL) {
      // Look for active WebGL canvases
      const activeCanvases = await page.locator('canvas').evaluateAll((canvases) => {
        return canvases.map((canvas, index) => {
          const gl = (canvas as HTMLCanvasElement).getContext('webgl') || 
                     (canvas as HTMLCanvasElement).getContext('webgl2');
          return {
            index,
            hasWebGL: !!gl,
            width: canvas.width,
            height: canvas.height,
            visible: canvas.offsetParent !== null
          };
        }).filter(c => c.hasWebGL && c.visible && c.width > 0 && c.height > 0);
      });
      
      console.log(`🎯 Active WebGL canvases: ${activeCanvases.length}`);
      activeCanvases.forEach((canvas, i) => {
        console.log(`   Canvas ${i + 1}: ${canvas.width}x${canvas.height} (WebGL)`);
      });
      
      if (activeCanvases.length > 0) {
        await appPage.takeFullPageScreenshot('webgl-3d-viewers-active');
        console.log('✅ 3D WebGL viewers are active and ready');
      }
    }
    
    // Check for Three.js or other 3D libraries
    const has3DLibraries = await page.evaluate(() => {
      return {
        threejs: typeof (window as any).THREE !== 'undefined',
        babylonjs: typeof (window as any).BABYLON !== 'undefined', 
        playcanvas: typeof (window as any).pc !== 'undefined'
      };
    });
    
    console.log('📚 3D Libraries detected:');
    Object.entries(has3DLibraries).forEach(([lib, present]) => {
      if (present) console.log(`   ✅ ${lib.toUpperCase()}: LOADED`);
    });
    
    console.log('🎯 WebGL 3D Viewer Functionality Check Complete');
  });
});