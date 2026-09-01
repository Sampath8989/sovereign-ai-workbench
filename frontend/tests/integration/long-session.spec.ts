/**
 * long-session.spec.ts — Integration Test (Real Backend)
 *
 * Tests:
 * - 5 consecutive real chat exchanges back-to-back
 * - Assert no memory leak, no stale state, consistent timing
 * - Closest proxy to actual demo conditions (multiple judge questions)
 */
import { test, expect } from '@playwright/test';

const EXCHANGES = [
  'What is 2+2?',
  'What is the capital of France?',
  'Explain what RAG means in one sentence.',
  'What is 10 divided by 2?',
  'Name a programming language.',
];

test.describe('Long Session (Real Backend)', () => {
  test('5 consecutive chat exchanges with consistent timing', async ({ page }) => {
    const timings: number[] = [];

    await page.goto('/');

    for (let i = 0; i < EXCHANGES.length; i++) {
      const prompt = EXCHANGES[i];

      // Fill and send
      const chatInput = page.getByPlaceholder(/ask a question/i);
      await expect(chatInput).toBeVisible();
      await chatInput.fill(prompt);

      const sendButton = page.getByRole('button', { name: /send/i });
      const start = Date.now();
      await sendButton.click();

      // Wait for response (spinner disappears)
      await expect(page.getByText(/processing/i)).not.toBeVisible({ timeout: 15000 });
      const elapsed = Date.now() - start;
      timings.push(elapsed);

      // Verify response appeared
      const responseArea = page.locator('[data-testid="response"]');
      await expect(responseArea.first()).toBeVisible();

      // Verify trace appeared
      const traceSteps = page.locator('[data-testid="trace-step"]');
      await expect(traceSteps.first()).toBeVisible();

      // Brief pause between exchanges (simulating judge reading)
      await page.waitForTimeout(500);
    }

    // Verify all exchanges completed
    expect(timings).toHaveLength(5);

    // Verify timing consistency (no exchange should be >3x the slowest average)
    const avgTiming = timings.reduce((a, b) => a + b, 0) / timings.length;
    for (let i = 0; i < timings.length; i++) {
      expect(
        timings[i],
        `Exchange ${i + 1} took ${timings[i]}ms, avg is ${avgTiming}ms`
      ).toBeLessThan(avgTiming * 3);
    }

    // Verify no memory leak by checking response count
    const responseCount = await page.locator('[data-testid="response"]').count();
    expect(responseCount).toBe(5);

    // All traces should be present
    const allTraces = await page.locator('[data-testid="trace-step"]').count();
    expect(allTraces).toBeGreaterThanOrEqual(5);

    console.log('Timings:', timings);
    console.log('Average:', avgTiming.toFixed(0), 'ms');
  });
});
