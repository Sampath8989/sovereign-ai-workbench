/**
 * demo-budget.spec.ts — Integration Test (Real Backend)
 *
 * The SINGLE MOST IMPORTANT test: scripts the exact demo_script.md flow
 * through the UI with real timers and asserts total elapsed time stays
 * within the 4.5-minute demo budget.
 *
 * Previously verified against MockLLM (0.98s total — meaningless).
 * Now re-verified against real Qwen-0.5B CPU inference (~4-10s per call).
 *
 * Budget: 270 seconds (4.5 minutes)
 * Expected: ~60-100s with real inference (each of 4 demos = 10-15s)
 */
import { test, expect } from '@playwright/test';

const DEMO_BUDGET_MS = 270_000; // 4.5 minutes

const DEMO_STEPS = [
  {
    name: 'Demo A (RAG → Word)',
    prompt: 'Create a standard operating procedure document about equipment maintenance',
    timeoutMs: 30_000,
  },
  {
    name: 'Demo B (Calculator → Excel)',
    prompt: 'Calculate NPSH required for a pump with flow rate 100 m3/h and create a spreadsheet',
    timeoutMs: 30_000,
  },
  {
    name: 'Demo C (P&ID → PPT)',
    prompt: 'Analyze the topology of P&ID at workspace/sandbox_files/test_pid.png and create a presentation',
    timeoutMs: 30_000,
  },
  {
    name: 'Demo D (Sovereignty)',
    prompt: 'Show me the system sovereignty status',
    timeoutMs: 30_000,
  },
];

test.describe('Demo Budget (Real Backend)', () => {
  test('full demo sequence completes within 4.5-minute budget', async ({ page }) => {
    const timings: { name: string; elapsed: number }[] = [];
    const demoStart = Date.now();

    await page.goto('/');

    for (const step of DEMO_STEPS) {
      const stepStart = Date.now();

      // Fill and send
      const chatInput = page.getByPlaceholder(/ask a question/i);
      await expect(chatInput).toBeVisible({ timeout: 5000 });
      await chatInput.fill(step.prompt);

      const sendButton = page.getByRole('button', { name: /send/i });
      await sendButton.click();

      // Wait for response (with generous timeout for real CPU inference)
      await expect(page.getByText(/processing/i)).not.toBeVisible({
        timeout: step.timeoutMs,
      });

      const elapsed = Date.now() - stepStart;
      timings.push({ name: step.name, elapsed });

      console.log(`${step.name}: ${(elapsed / 1000).toFixed(1)}s`);

      // Brief pause between demos (simulating presenter narration)
      await page.waitForTimeout(1000);
    }

    const totalElapsed = Date.now() - demoStart;
    const totalSeconds = (totalElapsed / 1000).toFixed(1);

    console.log(`\nTotal demo time: ${totalSeconds}s`);
    console.log(`Budget: ${DEMO_BUDGET_MS / 1000}s`);
    console.log(`Remaining budget: ${((DEMO_BUDGET_MS - totalElapsed) / 1000).toFixed(1)}s`);

    // Assert within budget
    expect(
      totalElapsed,
      `Demo took ${totalSeconds}s, budget is ${DEMO_BUDGET_MS / 1000}s — OVER BUDGET!`
    ).toBeLessThan(DEMO_BUDGET_MS);

    // Log individual timings for reporting
    for (const t of timings) {
      console.log(`  ${t.name}: ${(t.elapsed / 1000).toFixed(1)}s`);
    }
  });

  test('each demo step completes within its individual timeout', async ({ page }) => {
    await page.goto('/');

    for (const step of DEMO_STEPS) {
      const chatInput = page.getByPlaceholder(/ask a question/i);
      await expect(chatInput).toBeVisible({ timeout: 5000 });
      await chatInput.fill(step.prompt);

      const sendButton = page.getByRole('button', { name: /send/i });
      const start = Date.now();
      await sendButton.click();

      // Each step must complete within its timeout
      await expect(page.getByText(/processing/i)).not.toBeVisible({
        timeout: step.timeoutMs,
      });

      const elapsed = Date.now() - start;
      expect(
        elapsed,
        `${step.name} took ${(elapsed / 1000).toFixed(1)}s, timeout is ${step.timeoutMs / 1000}s`
      ).toBeLessThan(step.timeoutMs);

      await page.waitForTimeout(500);
    }
  });
});
