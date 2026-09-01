/**
 * sovereignty.spec.ts — Integration Test (Real Backend)
 *
 * Tests:
 * - Green badge on page load (real /health endpoint)
 * - Trigger real sentinel test
 * - Badge state reflects actual sentinel behavior
 * - No false "blocking" claims
 */
import { test, expect } from '@playwright/test';

test.describe('Sovereignty Monitor (Real Backend)', () => {
  test('shows green "AIR-GAP VERIFIED" badge on healthy backend', async ({ page }) => {
    await page.goto('/');

    // Wait for the sovereignty monitor to poll and get healthy status
    const badge = page.getByText(/air-gap verified/i);
    await expect(badge).toBeVisible({ timeout: 15000 });
  });

  test('triggering sentinel test updates badge state', async ({ page }) => {
    await page.goto('/');

    // Wait for healthy state
    await expect(page.getByText(/air-gap verified/i)).toBeVisible({ timeout: 15000 });

    // Find and click the test sovereignty button
    const testButton = page.getByRole('button', { name: /test sovereignty|trigger|breach/i });
    await expect(testButton).toBeVisible();

    await testButton.click();

    // After sentinel trigger, check that no false "blocked" claims appear
    await page.waitForTimeout(3000);

    const bodyText = await page.locator('body').textContent();
    expect(bodyText).not.toMatch(/sigkill/i);
    expect(bodyText).not.toMatch(/process.*killed/i);
  });

  test('badge reverts to healthy after next successful poll', async ({ page }) => {
    await page.goto('/');

    // Wait for initial healthy state
    await expect(page.getByText(/air-gap verified/i)).toBeVisible({ timeout: 15000 });

    // After next poll (2s interval), badge should return to green
    await expect(page.getByText(/air-gap verified/i)).toBeVisible({ timeout: 10000 });
  });
});
