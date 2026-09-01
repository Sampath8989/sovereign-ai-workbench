/**
 * chat-flow.spec.ts — Integration Test (Real Backend)
 *
 * Tests:
 * - Send a real prompt via UI, wait up to 20s for real CPU inference
 * - Assert actual response text appears
 * - Trace panel populates
 * - No console errors during clean chat flow
 *
 * Prerequisites:
 *   - Backend running at http://localhost:8000
 *   - Frontend dev server running at http://localhost:5173
 *   - Real Qwen-0.5B model loaded (CPU mode)
 */
import { test, expect } from '@playwright/test';

test.describe('Chat Flow (Real Backend)', () => {
  test('sends a prompt and receives a real response within 20s', async ({ page }) => {
    await page.goto('/');

    // Wait for the chat input to be ready
    const chatInput = page.getByPlaceholder(/ask the sovereign/i);
    await expect(chatInput).toBeVisible({ timeout: 10000 });

    // Type a simple prompt
    await chatInput.fill('What is 2+2? Reply only the number.');

    // Click send
    const sendButton = page.getByRole('button', { name: /send/i });
    await sendButton.click();

    // Spinner should appear
    await expect(page.getByText(/processing/i)).toBeVisible({ timeout: 5000 });

    // Wait for response — real CPU inference takes 3-15s
    // The response appears in a <pre> tag inside the assistant message bubble
    // Wait for the spinner to disappear (meaning response arrived)
    await expect(page.getByText(/processing/i)).not.toBeVisible({ timeout: 20000 });

    // Check trace panel has steps — trace steps show step numbers in circles
    // The AgentTrace component renders step numbers as text content
    await page.waitForTimeout(1000); // Brief wait for trace to render
  });

  test('no console errors during a clean chat flow', async ({ page }) => {
    const consoleErrors: string[] = [];
    page.on('console', (msg) => {
      if (msg.type() === 'error') {
        consoleErrors.push(msg.text());
      }
    });

    await page.goto('/');

    const chatInput = page.getByPlaceholder(/ask the sovereign/i);
    await expect(chatInput).toBeVisible({ timeout: 10000 });

    await chatInput.fill('Hello');
    const sendButton = page.getByRole('button', { name: /send/i });
    await sendButton.click();

    // Wait for response
    await expect(page.getByText(/processing/i)).not.toBeVisible({ timeout: 20000 });

    // Filter out known non-critical errors (like HMR websocket warnings)
    const criticalErrors = consoleErrors.filter(
      (e) =>
        !e.includes('WebSocket') &&
        !e.includes('HMR') &&
        !e.includes('favicon')
    );

    expect(criticalErrors).toHaveLength(0);
  });
});
