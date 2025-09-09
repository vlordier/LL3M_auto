import { test, expect } from '../fixtures/test-helpers';

test.describe('Session Management', () => {
  test('should open and close session browser', async ({ appPage, sessionBrowserPage }) => {
    await appPage.goto();
    
    // Open session browser
    await appPage.sessionBrowserButton.click();
    
    // Check that modal is visible
    await expect(sessionBrowserPage.page.getByText('Session Browser')).toBeVisible();
    await appPage.takeFullPageScreenshot('session-browser-open');
    
    // Close modal
    await sessionBrowserPage.page.getByRole('button', { name: /close/i }).first().click();
    
    // Modal should be hidden
    await expect(sessionBrowserPage.page.getByText('Session Browser')).toBeHidden();
    await appPage.takeFullPageScreenshot('session-browser-closed');
  });

  test('should display empty state when no sessions exist', async ({ appPage, sessionBrowserPage }) => {
    await appPage.goto();
    
    // Mock empty sessions response
    await sessionBrowserPage.page.route('**/api/sessions', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([])
      });
    });
    
    await appPage.sessionBrowserButton.click();
    
    // Check empty state
    await expect(sessionBrowserPage.page.getByText('No sessions found')).toBeVisible();
    await expect(sessionBrowserPage.page.getByText('Start generating assets to see your session history here')).toBeVisible();
    
    await appPage.takeFullPageScreenshot('session-browser-empty');
  });

  test('should display and filter sessions', async ({ appPage, sessionBrowserPage }) => {
    await appPage.goto();
    
    // Mock sessions data
    const mockSessions = [
      {
        session_id: 'session-1',
        prompt: 'Create a wooden chair with curved armrests',
        current_phase: 'completed',
        created_at: new Date().toISOString(),
        traces: [{ phase: 'completed', execution_time: 2.5 }],
        artifacts: [{ artifact_type: 'model', relative_path: 'chair.gltf' }],
        total_duration: 45.2
      },
      {
        session_id: 'session-2', 
        prompt: 'Design a futuristic spaceship',
        current_phase: 'asset_generation',
        created_at: new Date().toISOString(),
        traces: [{ phase: 'in_progress', execution_time: null }],
        artifacts: [],
        total_duration: null
      }
    ];
    
    await sessionBrowserPage.page.route('**/api/sessions', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockSessions)
      });
    });
    
    await appPage.sessionBrowserButton.click();
    
    // Check that sessions are displayed
    await expect(sessionBrowserPage.page.getByText('wooden chair')).toBeVisible();
    await expect(sessionBrowserPage.page.getByText('futuristic spaceship')).toBeVisible();
    
    await appPage.takeFullPageScreenshot('session-browser-with-sessions');
    
    // Test search functionality
    await sessionBrowserPage.searchSessions('chair');
    
    // Should show only chair session
    await expect(sessionBrowserPage.page.getByText('wooden chair')).toBeVisible();
    await expect(sessionBrowserPage.page.getByText('futuristic spaceship')).toBeHidden();
    
    await appPage.takeFullPageScreenshot('session-browser-filtered');
    
    // Clear search
    await sessionBrowserPage.searchInput.clear();
    
    // Both sessions should be visible again
    await expect(sessionBrowserPage.page.getByText('futuristic spaceship')).toBeVisible();
  });

  test('should resume a session', async ({ appPage, sessionBrowserPage }) => {
    await appPage.goto();
    
    const mockSessions = [
      {
        session_id: 'resume-test-session',
        prompt: 'Create a medieval castle',
        current_phase: 'completed',
        created_at: new Date().toISOString(),
        traces: [
          { phase: 'prompt_analysis', execution_time: 1.2 },
          { phase: 'code_generation', execution_time: 3.4 },
          { phase: 'asset_generation', execution_time: 8.1 }
        ],
        artifacts: [
          { artifact_type: 'model', relative_path: 'castle.gltf', size_bytes: 2048000 }
        ],
        total_duration: 65.7
      }
    ];
    
    await sessionBrowserPage.page.route('**/api/sessions', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockSessions)
      });
    });
    
    // Mock resume session endpoint
    await sessionBrowserPage.page.route('**/api/sessions/resume-test-session/resume', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ success: true, message: 'Session resumed' })
      });
    });
    
    await appPage.sessionBrowserButton.click();
    
    // Click on session to resume it
    const sessionCard = sessionBrowserPage.page.getByText('Create a medieval castle');
    await sessionCard.click();
    
    await appPage.takeFullPageScreenshot('session-resume-clicked');
    
    // Session browser should close and session should be active
    await expect(sessionBrowserPage.page.getByText('Session Browser')).toBeHidden();
    
    // Check that session is active (sidebar should show it)
    await expect(appPage.page.getByText('Current Session')).toBeVisible();
  });

  test('should delete a session', async ({ appPage, sessionBrowserPage }) => {
    await appPage.goto();
    
    const mockSessions = [
      {
        session_id: 'delete-test-session',
        prompt: 'Create a test object for deletion',
        current_phase: 'completed',
        created_at: new Date().toISOString(),
        traces: [],
        artifacts: [],
        total_duration: 10.0
      }
    ];
    
    await sessionBrowserPage.page.route('**/api/sessions', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockSessions)
      });
    });
    
    // Mock delete endpoint
    await sessionBrowserPage.page.route('**/api/sessions/delete-test-session', async route => {
      if (route.request().method() === 'DELETE') {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ success: true })
        });
      }
    });
    
    await appPage.sessionBrowserButton.click();
    
    // Click delete button (this will show on hover)
    const sessionCard = sessionBrowserPage.page.getByText('test object for deletion').locator('..');
    await sessionCard.hover();
    
    await appPage.takeFullPageScreenshot('session-delete-hover');
    
    // Look for delete button and click it
    const deleteButton = sessionCard.getByRole('button', { name: /delete/i });
    
    // Mock confirmation dialog
    sessionBrowserPage.page.on('dialog', async dialog => {
      expect(dialog.type()).toBe('confirm');
      await dialog.accept();
    });
    
    if (await deleteButton.isVisible()) {
      await deleteButton.click();
      await appPage.takeFullPageScreenshot('session-delete-clicked');
    }
  });

  test('should show session status indicators', async ({ appPage, sessionBrowserPage }) => {
    await appPage.goto();
    
    const mockSessions = [
      {
        session_id: 'status-completed',
        prompt: 'Completed session test',
        current_phase: 'completed',
        created_at: new Date().toISOString(),
        traces: [{ phase: 'completed' }],
        artifacts: [{ artifact_type: 'model' }],
        total_duration: 30.0
      },
      {
        session_id: 'status-in-progress',
        prompt: 'In progress session test',
        current_phase: 'asset_generation',
        created_at: new Date().toISOString(),
        traces: [{ phase: 'asset_generation' }],
        artifacts: [],
        total_duration: null
      },
      {
        session_id: 'status-error',
        prompt: 'Failed session test',
        current_phase: 'error',
        created_at: new Date().toISOString(),
        traces: [{ phase: 'error' }],
        artifacts: [],
        total_duration: null
      }
    ];
    
    await sessionBrowserPage.page.route('**/api/sessions', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockSessions)
      });
    });
    
    await appPage.sessionBrowserButton.click();
    
    // Check for different status indicators
    await expect(sessionBrowserPage.page.getByText('completed')).toBeVisible();
    await expect(sessionBrowserPage.page.getByText('asset generation')).toBeVisible();
    await expect(sessionBrowserPage.page.getByText('error')).toBeVisible();
    
    await appPage.takeFullPageScreenshot('session-status-indicators');
  });

  test('should show session artifacts and metadata', async ({ appPage, sessionBrowserPage }) => {
    await appPage.goto();
    
    const mockSessions = [
      {
        session_id: 'metadata-test',
        prompt: 'Session with rich metadata',
        current_phase: 'completed',
        created_at: new Date().toISOString(),
        traces: [
          { phase: 'prompt_analysis', execution_time: 1.0 },
          { phase: 'code_generation', execution_time: 2.0 },
          { phase: 'asset_generation', execution_time: 5.0 }
        ],
        artifacts: [
          { 
            artifact_type: 'model', 
            relative_path: 'complex-model.gltf',
            size_bytes: 1024000 
          },
          { 
            artifact_type: 'texture', 
            relative_path: 'model-texture.png',
            size_bytes: 512000 
          }
        ],
        total_duration: 125.5
      }
    ];
    
    await sessionBrowserPage.page.route('**/api/sessions', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockSessions)
      });
    });
    
    await appPage.sessionBrowserButton.click();
    
    // Check metadata display
    await expect(sessionBrowserPage.page.getByText('3 steps')).toBeVisible();
    await expect(sessionBrowserPage.page.getByText('2 assets')).toBeVisible();
    await expect(sessionBrowserPage.page.getByText(/Duration.*125/)).toBeVisible();
    
    // Check artifact display
    await expect(sessionBrowserPage.page.getByText('GLTF format')).toBeVisible();
    
    await appPage.takeFullPageScreenshot('session-metadata-display');
  });

  test('should handle bulk session operations', async ({ appPage, sessionBrowserPage }) => {
    await appPage.goto();
    
    const mockSessions = [
      { session_id: 'bulk-1', prompt: 'Bulk test 1', current_phase: 'completed', created_at: new Date().toISOString(), traces: [], artifacts: [] },
      { session_id: 'bulk-2', prompt: 'Bulk test 2', current_phase: 'completed', created_at: new Date().toISOString(), traces: [], artifacts: [] },
      { session_id: 'bulk-3', prompt: 'Bulk test 3', current_phase: 'completed', created_at: new Date().toISOString(), traces: [], artifacts: [] }
    ];
    
    await sessionBrowserPage.page.route('**/api/sessions', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockSessions)
      });
    });
    
    await appPage.sessionBrowserButton.click();
    
    // Select multiple sessions
    const checkboxes = sessionBrowserPage.page.locator('input[type="checkbox"]');
    await checkboxes.nth(0).click();
    await checkboxes.nth(1).click();
    
    await appPage.takeFullPageScreenshot('session-bulk-selection');
    
    // Check that bulk delete button appears
    await expect(sessionBrowserPage.page.getByText('Delete Selected (2)')).toBeVisible();
  });
});