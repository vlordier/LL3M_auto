import { test, expect } from '../fixtures/test-helpers';

test.describe('Workflow Visualization', () => {
  test('should display workflow steps in sidebar', async ({ appPage, workflowPage }) => {
    await appPage.goto();
    
    // Mock a session with workflow traces
    await workflowPage.page.route('**/api/generate', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          success: true,
          session_id: 'workflow-test',
          traces: [
            {
              trace_id: 'trace-1',
              phase: 'prompt_analysis',
              start_time: new Date().toISOString(),
              end_time: new Date().toISOString(),
              execution_time: 1.5,
              outputs: { analysis: 'Prompt analyzed successfully' }
            },
            {
              trace_id: 'trace-2',
              phase: 'code_generation',
              start_time: new Date().toISOString(),
              end_time: null,
              execution_time: null,
              outputs: null
            }
          ]
        })
      });
    });
    
    // Send message to trigger workflow
    await appPage.sendMessage('Create a simple object');
    
    // Wait for workflow to appear in sidebar
    await expect(workflowPage.page.getByText('Generation Pipeline')).toBeVisible({ timeout: 10000 });
    
    await appPage.takeFullPageScreenshot('workflow-steps-sidebar');
    
    // Check individual steps
    await expect(workflowPage.page.getByText('Prompt Analysis')).toBeVisible();
    await expect(workflowPage.page.getByText('Code Generation')).toBeVisible();
    await expect(workflowPage.page.getByText('Asset Generation')).toBeVisible();
  });

  test('should show step status indicators', async ({ appPage, workflowPage }) => {
    await appPage.goto();
    
    // Create a session with various step statuses
    await workflowPage.page.addInitScript(() => {
      window.mockWorkflowData = {
        traces: [
          {
            trace_id: 'completed-step',
            phase: 'prompt_analysis',
            start_time: new Date(Date.now() - 5000).toISOString(),
            end_time: new Date().toISOString(),
            execution_time: 2.1,
            outputs: { status: 'completed' }
          },
          {
            trace_id: 'running-step',
            phase: 'code_generation',
            start_time: new Date().toISOString(),
            end_time: null,
            execution_time: null,
            outputs: null
          },
          {
            trace_id: 'pending-step',
            phase: 'asset_generation',
            start_time: null,
            end_time: null,
            execution_time: null,
            outputs: null
          }
        ]
      };
    });
    
    // Look for status indicators
    const statusElements = [
      workflowPage.page.locator('.text-green-600'), // Completed
      workflowPage.page.locator('.text-blue-600'),  // Running
      workflowPage.page.locator('.text-gray-400'),  // Pending
      workflowPage.page.locator('.animate-spin')    // Loading spinner
    ];
    
    await appPage.takeFullPageScreenshot('workflow-status-indicators');
  });

  test('should show execution times and progress', async ({ appPage, workflowPage }) => {
    await appPage.goto();
    
    // Mock detailed workflow data
    await workflowPage.page.route('**/api/session/*/traces', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([
          {
            trace_id: 'trace-timing',
            phase: 'prompt_analysis',
            start_time: new Date(Date.now() - 10000).toISOString(),
            end_time: new Date(Date.now() - 8500).toISOString(),
            execution_time: 1.5,
            outputs: { 
              prompt_complexity: 'medium',
              estimated_time: '30s'
            }
          }
        ])
      });
    });
    
    // Look for timing information
    const timingElements = [
      workflowPage.page.getByText(/\d+\.\d+s/), // Execution time format
      workflowPage.page.getByText(/\d+ of \d+ steps/), // Progress format
    ];
    
    await appPage.takeFullPageScreenshot('workflow-timing-info');
  });

  test('should show step details and outputs', async ({ appPage, workflowPage }) => {
    await appPage.goto();
    
    // Send a message to create workflow
    await appPage.sendMessage('Generate a test object');
    
    // Look for step details
    const detailElements = [
      workflowPage.page.getByText('Understanding your requirements'),
      workflowPage.page.getByText('Writing Blender scripts'),
      workflowPage.page.getByText('Creating 3D models'),
    ];
    
    // Check for output information
    await workflowPage.page.waitForTimeout(2000);
    await appPage.takeFullPageScreenshot('workflow-step-details');
  });

  test('should update progress bar', async ({ appPage, workflowPage }) => {
    await appPage.goto();
    
    // Mock progressive workflow updates
    let stepCount = 0;
    await workflowPage.page.route('**/api/generate', async route => {
      stepCount++;
      const traces = [
        {
          phase: 'prompt_analysis',
          start_time: new Date().toISOString(),
          end_time: stepCount >= 1 ? new Date().toISOString() : null,
          execution_time: stepCount >= 1 ? 1.2 : null
        },
        {
          phase: 'code_generation', 
          start_time: stepCount >= 2 ? new Date().toISOString() : null,
          end_time: stepCount >= 3 ? new Date().toISOString() : null,
          execution_time: stepCount >= 3 ? 2.5 : null
        }
      ];
      
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ traces, session_id: 'progress-test' })
      });
    });
    
    await appPage.sendMessage('Test progress bar');
    
    // Look for progress indicators
    const progressBar = workflowPage.page.locator('[data-testid="progress-bar"]');
    if (await progressBar.isVisible()) {
      await appPage.takeFullPageScreenshot('workflow-progress-bar');
    }
    
    // Check overall progress text
    await expect(workflowPage.page.getByText(/\d+ \/ \d+ steps/)).toBeVisible();
  });

  test('should handle workflow errors gracefully', async ({ appPage, workflowPage }) => {
    await appPage.goto();
    
    // Mock workflow with error
    await workflowPage.page.route('**/api/generate', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          success: false,
          traces: [
            {
              trace_id: 'error-trace',
              phase: 'code_generation',
              start_time: new Date().toISOString(),
              end_time: new Date().toISOString(),
              execution_time: 3.2,
              error_message: 'Blender script compilation failed',
              outputs: null
            }
          ]
        })
      });
    });
    
    await appPage.sendMessage('This will cause an error');
    
    // Look for error indicators
    await expect(workflowPage.page.getByText(/error.*occurred/i)).toBeVisible({ timeout: 10000 });
    await expect(workflowPage.page.getByText('Blender script compilation failed')).toBeVisible();
    
    await appPage.takeFullPageScreenshot('workflow-error-handling');
  });

  test('should show code version indicators', async ({ appPage, workflowPage }) => {
    await appPage.goto();
    
    // Mock workflow with code generation
    await workflowPage.page.route('**/api/generate', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json', 
        body: JSON.stringify({
          traces: [
            {
              phase: 'code_generation',
              start_time: new Date().toISOString(),
              end_time: new Date().toISOString(),
              execution_time: 4.1,
              outputs: { code_generated: true, version: 'v1.0' }
            }
          ],
          code_versions: [
            {
              version_id: 'v1',
              code_content: 'import bpy\n# Generated code...',
              source_phase: 'code_generation'
            }
          ]
        })
      });
    });
    
    await appPage.sendMessage('Generate with code versions');
    
    // Look for code version indicators
    await expect(workflowPage.page.getByText('Code version generated')).toBeVisible({ timeout: 10000 });
    
    await appPage.takeFullPageScreenshot('workflow-code-versions');
  });

  test('should show preview renders for completed steps', async ({ appPage, workflowPage }) => {
    await appPage.goto();
    
    // Mock completed asset generation with preview
    await workflowPage.page.route('**/api/generate', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          traces: [
            {
              phase: 'asset_generation',
              start_time: new Date().toISOString(),
              end_time: new Date().toISOString(),
              execution_time: 12.5,
              outputs: { asset_created: true, preview_available: true }
            }
          ],
          preview_url: 'test-preview.png'
        })
      });
    });
    
    await appPage.sendMessage('Generate with preview');
    
    // Look for preview placeholder or actual preview
    await workflowPage.page.waitForTimeout(3000);
    
    const previewElements = [
      workflowPage.page.getByText('Preview render'),
      workflowPage.page.locator('[data-testid="step-preview"]'),
      workflowPage.page.locator('img[alt*="preview"]')
    ];
    
    await appPage.takeFullPageScreenshot('workflow-preview-renders');
  });

  test('should be collapsible/expandable', async ({ appPage, workflowPage }) => {
    await appPage.goto();
    
    // Look for expand/collapse controls
    const expandControls = [
      workflowPage.page.getByRole('button', { name: /expand/i }),
      workflowPage.page.getByRole('button', { name: /collapse/i }),
      workflowPage.page.locator('[data-testid="workflow-toggle"]')
    ];
    
    for (const control of expandControls) {
      if (await control.isVisible()) {
        await control.click();
        await appPage.takeFullPageScreenshot('workflow-expanded-collapsed');
        break;
      }
    }
  });
});