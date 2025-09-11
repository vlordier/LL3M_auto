import { test, expect } from '../fixtures/test-helpers';

test.describe('User Confirmation Workflow', () => {
  test('should complete full user confirmation flow - accept', async ({ appPage, page }) => {
    await appPage.goto();

    // Mock the backend API endpoints
    let reformulationReceived = false;
    let confirmationReceived = false;
    
    // Mock the generation endpoint to return reformulation state
    await page.route('**/api/generate', async route => {
      reformulationReceived = true;
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          success: true,
          session_id: 'test-session-reformulation',
          message: 'Reformulation in progress',
          reformulation_data: {
            original_request: 'Create a cube',
            reformulated_request: 'Create a detailed geometric cube with smooth edges and proper UV mapping for 3D visualization'
          }
        })
      });
    });

    // Mock the step control endpoint to show awaiting confirmation
    await page.route('**/api/sessions/*/step-control', async route => {
      if (reformulationReceived && !confirmationReceived) {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            can_continue: false,
            current_step: 'reformulation',
            awaiting_user_confirmation: true,
            reformulation_data: {
              original_request: 'Create a cube',
              reformulated_request: 'Create a detailed geometric cube with smooth edges and proper UV mapping for 3D visualization'
            }
          })
        });
      } else {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            can_continue: true,
            current_step: 'completed',
            awaiting_user_confirmation: false
          })
        });
      }
    });

    // Mock the confirmation endpoint
    await page.route('**/api/sessions/*/confirm', async route => {
      confirmationReceived = true;
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          success: true,
          action: 'accepted'
        })
      });
    });

    // Mock progress messages endpoint
    await page.route('**/api/sessions/*/progress', async route => {
      const messages = [];
      
      if (reformulationReceived && !confirmationReceived) {
        messages.push({
          timestamp: new Date().toISOString(),
          phase: 'request_reformulation',
          title: 'Request Analysis',
          description: 'Analyzing and reformulating your request for better 3D asset generation',
          progress_percent: 10,
          technical_details: {
            user_confirmation_required: true,
            original_request: 'Create a cube',
            reformulated_request: 'Create a detailed geometric cube with smooth edges and proper UV mapping for 3D visualization',
            improvements: 'Added specific geometric and UV mapping details'
          }
        });
      } else if (confirmationReceived) {
        messages.push({
          timestamp: new Date().toISOString(),
          phase: 'generation',
          title: 'Generating Asset',
          description: 'User confirmation received - continuing with generation',
          progress_percent: 50,
          technical_details: {}
        });
      }

      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(messages)
      });
    });

    // Start the generation process
    await appPage.sendMessage('Create a cube');
    
    // Wait for the reformulation section to appear (look for "Reformulated Request" text)
    await expect(page.getByText(/reformulated request/i)).toBeVisible({ timeout: 10000 });
    await appPage.takeFullPageScreenshot('reformulation-received');

    // Check that confirmation buttons are visible
    const acceptButton = page.getByText(/accept reformulation/i);
    const modifyButton = page.getByText(/modify request/i);
    const rejectButton = page.getByText(/use original/i);
    
    await expect(acceptButton).toBeVisible({ timeout: 5000 });
    await expect(modifyButton).toBeVisible({ timeout: 5000 });
    await expect(rejectButton).toBeVisible({ timeout: 5000 });
    
    await appPage.takeFullPageScreenshot('confirmation-buttons-visible');

    // Click accept button
    await acceptButton.click();
    
    // Wait for confirmation to be processed
    await page.waitForTimeout(2000);
    
    // Verify that confirmation was sent
    expect(confirmationReceived).toBe(true);
    
    // Check that the workflow continues
    await expect(page.getByText(/user confirmation received/i)).toBeVisible({ timeout: 10000 });
    
    await appPage.takeFullPageScreenshot('confirmation-accepted');
  });

  test('should complete full user confirmation flow - modify', async ({ appPage, page }) => {
    await appPage.goto();

    let reformulationReceived = false;
    let confirmationReceived = false;
    let modifiedPrompt = '';
    
    // Mock endpoints similar to accept test but handle modify action
    await page.route('**/api/generate', async route => {
      reformulationReceived = true;
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          success: true,
          session_id: 'test-session-modify',
          reformulation_data: {
            original_request: 'Create a sphere',
            reformulated_request: 'Create a perfect geometric sphere with smooth surface normals'
          }
        })
      });
    });

    await page.route('**/api/sessions/*/step-control', async route => {
      if (reformulationReceived && !confirmationReceived) {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            can_continue: false,
            current_step: 'reformulation',
            awaiting_user_confirmation: true,
            reformulation_data: {
              original_request: 'Create a sphere',
              reformulated_request: 'Create a perfect geometric sphere with smooth surface normals'
            }
          })
        });
      } else {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            can_continue: true,
            current_step: 'completed'
          })
        });
      }
    });

    await page.route('**/api/sessions/*/confirm', async route => {
      const request = await route.request().postDataJSON();
      confirmationReceived = true;
      modifiedPrompt = request.modified_prompt || '';
      
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          success: true,
          action: 'modified',
          modified_prompt: modifiedPrompt
        })
      });
    });

    await page.route('**/api/sessions/*/progress', async route => {
      const messages = [];
      
      if (reformulationReceived && !confirmationReceived) {
        messages.push({
          timestamp: new Date().toISOString(),
          phase: 'request_reformulation',
          title: 'Request Analysis',
          description: 'Analyzing and reformulating your request for better 3D asset generation',
          progress_percent: 10,
          technical_details: {
            user_confirmation_required: true,
            original_request: 'Create a sphere',
            reformulated_request: 'Create a perfect geometric sphere with smooth surface normals'
          }
        });
      } else if (confirmationReceived) {
        messages.push({
          timestamp: new Date().toISOString(),
          phase: 'generation',
          title: 'Generating Asset',
          description: `User modified request: ${modifiedPrompt}`,
          progress_percent: 50,
          technical_details: {}
        });
      }

      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(messages)
      });
    });

    // Start the generation process
    await appPage.sendMessage('Create a sphere');
    
    // Wait for reformulation and click modify
    await expect(page.getByText(/reformulated request/i)).toBeVisible({ timeout: 10000 });
    
    const modifyButton = page.getByText(/modify request/i);
    await expect(modifyButton).toBeVisible();
    await modifyButton.click();
    
    // Fill in the modification input
    const modifyInput = page.getByPlaceholder(/enter your modified request/i);
    await expect(modifyInput).toBeVisible({ timeout: 5000 });
    
    const modifiedText = 'Create a textured sphere with metallic material';
    await modifyInput.fill(modifiedText);
    
    await appPage.takeFullPageScreenshot('modify-input-filled');
    
    // Submit the modification
    const submitModifyButton = page.getByText(/submit modification/i);
    await expect(submitModifyButton).toBeVisible();
    await submitModifyButton.click();
    
    // Wait for confirmation processing
    await page.waitForTimeout(2000);
    
    // Verify the modification was received
    expect(confirmationReceived).toBe(true);
    expect(modifiedPrompt).toBe(modifiedText);
    
    await appPage.takeFullPageScreenshot('modification-submitted');
  });

  test('should complete full user confirmation flow - reject', async ({ appPage, page }) => {
    await appPage.goto();

    let reformulationReceived = false;
    let confirmationReceived = false;
    
    // Mock endpoints for reject flow
    await page.route('**/api/generate', async route => {
      reformulationReceived = true;
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          success: true,
          session_id: 'test-session-reject',
          reformulation_data: {
            original_request: 'Create a cylinder',
            reformulated_request: 'Create a parametric cylinder with customizable height and radius'
          }
        })
      });
    });

    await page.route('**/api/sessions/*/step-control', async route => {
      if (reformulationReceived && !confirmationReceived) {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            can_continue: false,
            current_step: 'reformulation',
            awaiting_user_confirmation: true
          })
        });
      } else {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            can_continue: true,
            current_step: 'generation_with_original'
          })
        });
      }
    });

    await page.route('**/api/sessions/*/confirm', async route => {
      confirmationReceived = true;
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          success: true,
          action: 'rejected'
        })
      });
    });

    await page.route('**/api/sessions/*/progress', async route => {
      const messages = [];
      
      if (reformulationReceived && !confirmationReceived) {
        messages.push({
          timestamp: new Date().toISOString(),
          phase: 'request_reformulation',
          title: 'Request Analysis',
          description: 'Analyzing and reformulating your request for better 3D asset generation',
          progress_percent: 10,
          technical_details: {
            user_confirmation_required: true,
            original_request: 'Create a cylinder',
            reformulated_request: 'Create a parametric cylinder with customizable height and radius'
          }
        });
      } else if (confirmationReceived) {
        messages.push({
          timestamp: new Date().toISOString(),
          phase: 'generation',
          title: 'Generating Asset',
          description: 'Using original request as requested by user',
          progress_percent: 50,
          technical_details: {}
        });
      }

      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(messages)
      });
    });

    // Start the generation process
    await appPage.sendMessage('Create a cylinder');
    
    // Wait for reformulation and click reject
    await expect(page.getByText(/reformulated request/i)).toBeVisible({ timeout: 10000 });
    
    const rejectButton = page.getByText(/use original/i);
    await expect(rejectButton).toBeVisible();
    await rejectButton.click();
    
    // Wait for confirmation processing
    await page.waitForTimeout(2000);
    
    // Verify rejection was processed
    expect(confirmationReceived).toBe(true);
    
    // Check that workflow continues with original request
    await expect(page.getByText(/using original request/i)).toBeVisible({ timeout: 10000 });
    
    await appPage.takeFullPageScreenshot('rejection-processed');
  });

  test('should handle user confirmation timeout gracefully', async ({ appPage, page }) => {
    await appPage.goto();

    // Mock a scenario where confirmation is required but never submitted
    await page.route('**/api/generate', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          success: true,
          session_id: 'test-session-timeout',
          reformulation_data: {
            original_request: 'Create a pyramid',
            reformulated_request: 'Create a geometric pyramid with precise angles'
          }
        })
      });
    });

    await page.route('**/api/sessions/*/step-control', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          can_continue: false,
          current_step: 'reformulation',
          awaiting_user_confirmation: true
        })
      });
    });

    await page.route('**/api/sessions/*/progress', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([{
          timestamp: new Date().toISOString(),
          phase: 'request_reformulation',
          title: 'Request Analysis',
          description: 'Request reformulated - waiting for user confirmation',
          progress_percent: 10,
          technical_details: {
            user_confirmation_required: true,
            original_request: 'Create a pyramid',
            reformulated_request: 'Create a geometric pyramid with precise angles'
          }
        }])
      });
    });

    // Start the generation process
    await appPage.sendMessage('Create a pyramid');
    
    // Wait for reformulation message
    await expect(page.getByText(/reformulated request/i)).toBeVisible({ timeout: 10000 });
    
    // Verify confirmation buttons are present but don't click them
    await expect(page.getByText(/accept reformulation/i)).toBeVisible();
    await expect(page.getByText(/modify request/i)).toBeVisible();
    await expect(page.getByText(/use original/i)).toBeVisible();
    
    await appPage.takeFullPageScreenshot('confirmation-waiting');
    
    // Wait longer to simulate user thinking time
    await page.waitForTimeout(5000);
    
    // Verify the UI is still in waiting state
    await expect(page.getByText(/accept reformulation/i)).toBeVisible();
    
    await appPage.takeFullPageScreenshot('confirmation-still-waiting');
  });

  test('should display reformulation details correctly', async ({ appPage, page }) => {
    await appPage.goto();

    const originalRequest = 'Make a chair';
    const reformulatedRequest = 'Create an ergonomic office chair with adjustable height, lumbar support, and modern design suitable for 3D visualization and rendering';

    await page.route('**/api/generate', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          success: true,
          session_id: 'test-session-details',
          reformulation_data: {
            original_request: originalRequest,
            reformulated_request: reformulatedRequest
          }
        })
      });
    });

    await page.route('**/api/sessions/*/step-control', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          can_continue: false,
          current_step: 'reformulation',
          awaiting_user_confirmation: true,
          reformulation_data: {
            original_request: originalRequest,
            reformulated_request: reformulatedRequest
          }
        })
      });
    });

    await page.route('**/api/sessions/*/progress', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([{
          timestamp: new Date().toISOString(),
          phase: 'request_reformulation',
          title: 'Request Analysis',
          description: 'Request reformulated for enhanced 3D generation',
          progress_percent: 15,
          technical_details: {
            user_confirmation_required: true,
            original_request: originalRequest,
            reformulated_request: reformulatedRequest
          }
        }])
      });
    });

    // Start the generation
    await appPage.sendMessage(originalRequest);
    
    // Wait for reformulation display
    await expect(page.getByText(/reformulated request/i)).toBeVisible({ timeout: 10000 });
    
    // Check that both original and reformulated requests are displayed
    await expect(page.getByText(originalRequest)).toBeVisible();
    await expect(page.getByText(reformulatedRequest)).toBeVisible();
    
    // Check for proper labeling
    await expect(page.getByText(/original.*request/i)).toBeVisible();
    await expect(page.getByText(/reformulated.*request/i)).toBeVisible();
    
    await appPage.takeFullPageScreenshot('reformulation-details-displayed');
  });
});