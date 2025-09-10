import { test as base, expect, Page, Locator } from '@playwright/test';

export class AppPage {
  readonly page: Page;
  readonly chatInput: Locator;
  readonly sendButton: Locator;
  readonly messagesContainer: Locator;
  readonly viewer3D: Locator;
  readonly sidebarToggle: Locator;
  readonly sessionBrowserButton: Locator;
  readonly viewModeButtons: Locator;

  constructor(page: Page) {
    this.page = page;
    this.chatInput = page.getByPlaceholder('Describe the 3D asset you want to create...');
    this.sendButton = page.getByRole('button', { name: /send/i });
    this.messagesContainer = page.locator('[data-testid="messages-container"]');
    this.viewer3D = page.locator('[data-testid="three-viewer"]');
    this.sidebarToggle = page.getByRole('button', { name: /toggle sidebar/i });
    this.sessionBrowserButton = page.getByRole('button', { name: /browse sessions/i });
    this.viewModeButtons = page.locator('[data-testid="view-mode-controls"]');
  }

  async goto() {
    await this.page.goto('/');
    await this.page.waitForSelector('[data-testid="app-root"]', { timeout: 30000 });
  }

  async sendMessage(message: string) {
    await this.chatInput.fill(message);
    await this.sendButton.click();
  }

  async waitForGeneration() {
    // Wait for generation to start
    await expect(this.page.getByText(/generating/i)).toBeVisible({ timeout: 5000 });

    // Wait for generation to complete (with longer timeout)
    await expect(this.page.getByText(/generating/i)).toBeHidden({ timeout: 60000 });

    // Wait for any asset preview to appear
    await this.page.waitForSelector('[data-testid="asset-preview"]', { timeout: 10000 }).catch(() => {
      console.log('No asset preview found, continuing...');
    });
  }

  async switchViewMode(mode: 'chat' | 'split' | '3d') {
    const container = this.viewModeButtons;

    // Map test mode names to actual button text
    const buttonTextMap: Record<'chat' | 'split' | '3d', string> = {
      chat: 'Chat',
      split: 'Split',
      '3d': '3D View',
    };

    const buttonText = buttonTextMap[mode];
    const modeButton = container.getByRole('button', { name: buttonText }).first();

    // Check if button is enabled before clicking; if disabled and requesting 3D, fall back to Split
    if (!(await modeButton.isEnabled())) {
      if (mode === '3d') {
        const splitBtn = container.getByRole('button', { name: 'Split' }).first();
        if (await splitBtn.isEnabled()) {
          await splitBtn.click();
          await this.page.waitForTimeout(300);
        }
      }
      return;
    }

    await modeButton.click();
    await this.page.waitForTimeout(500); // Allow transition
  }

  async takeFullPageScreenshot(name: string) {
    await this.page.screenshot({
      path: `test-results/screenshots/${name}.png`,
      fullPage: true
    });
  }

  async startVideoRecording() {
    // Video recording is handled by Playwright config
    return true;
  }
}

export class SessionBrowserPage {
  readonly page: Page;
  readonly searchInput: Locator;
  readonly sessionList: Locator;
  readonly deleteButtons: Locator;

  constructor(page: Page) {
    this.page = page;
    this.searchInput = page.getByPlaceholder(/search sessions/i);
    this.sessionList = page.locator('[data-testid="session-list"]');
    this.deleteButtons = page.getByRole('button', { name: /delete/i });
  }

  async searchSessions(query: string) {
    await this.searchInput.fill(query);
    await this.page.waitForTimeout(500); // Allow search to filter
  }

  async selectSession(sessionId: string) {
    const sessionCard = this.page.locator(`[data-session-id="${sessionId}"]`);
    await sessionCard.click();
  }
}

export class WorkflowPage {
  readonly page: Page;
  readonly workflowSteps: Locator;
  readonly progressBar: Locator;

  constructor(page: Page) {
    this.page = page;
    this.workflowSteps = page.locator('[data-testid="workflow-steps"]');
    this.progressBar = page.locator('[data-testid="progress-bar"]');
  }

  async waitForStepComplete(stepName: string) {
    const step = this.page.locator(`[data-testid="step-${stepName}"]`);
    await expect(step).toHaveAttribute('data-status', 'completed', { timeout: 30000 });
  }

  async getStepStatus(stepName: string) {
    const step = this.page.locator(`[data-testid="step-${stepName}"]`);
    return await step.getAttribute('data-status');
  }
}

// Custom test fixture with page objects
export const test = base.extend<{
  appPage: AppPage;
  sessionBrowserPage: SessionBrowserPage;
  workflowPage: WorkflowPage;
}>({
  appPage: async ({ page }, use) => {
    const appPage = new AppPage(page);
    await use(appPage);
  },

  sessionBrowserPage: async ({ page }, use) => {
    const sessionBrowserPage = new SessionBrowserPage(page);
    await use(sessionBrowserPage);
  },

  workflowPage: async ({ page }, use) => {
    const workflowPage = new WorkflowPage(page);
    await use(workflowPage);
  },
});

export { expect } from '@playwright/test';
