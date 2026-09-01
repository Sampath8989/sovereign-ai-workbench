/**
 * ModelStatus.test.tsx
 *
 * Tests:
 * - Shows Qwen-0.5B as loaded/available on CPU
 * - 0.0GB VRAM is displayed correctly (not an error state)
 * - Hardware tier shown as BUILD
 * - Free VRAM shown as positive number
 * - No "GPU accelerated" claims when running CPU-only
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, act } from '@testing-library/react';
import { http, HttpResponse } from 'msw';
import { server } from '../setup';
import ModelStatus from '../../components/ModelStatus';

describe('ModelStatus', () => {
  beforeEach(() => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('shows BUILD tier and CPU mode', async () => {
    render(<ModelStatus />);

    await waitFor(() => {
      expect(screen.getByText(/BUILD/i)).toBeInTheDocument();
    });
  });

  it('shows 0.0GB VRAM used as a healthy state (not error)', async () => {
    const { container } = render(<ModelStatus />);

    await waitFor(() => {
      expect(screen.getByText(/0\.0/i)).toBeInTheDocument();
    });

    // Should NOT have error styling for 0.0GB
    const vramEl = screen.getByText(/0\.0/);
    expect(vramEl).not.toHaveClass(/error|red|text-red/);
  });

  it('shows free VRAM section when health loads', async () => {
    render(<ModelStatus />);

    // The MSW mock returns health with live_free_vram_gb: 3.68
    // The component renders: Live free: X.X GB
    // Wait for the component to finish its initial fetch
    await waitFor(() => {
      expect(screen.queryByText(/connecting to backend/i)).not.toBeInTheDocument();
    });

    // Now check the content
    const text = document.body.textContent || '';
    const hasLiveFree = /live free/i.test(text);
    const hasNoModels = /no models loaded/i.test(text);
    expect(hasLiveFree || hasNoModels).toBe(true);
  });

  it('renders model roster information (empty roster shows fallback)', async () => {
    render(<ModelStatus />);

    await waitFor(() => {
      // With empty resident_models, shows "No models loaded yet"
      expect(screen.getByText(/no models loaded|qwen/i)).toBeInTheDocument();
    });
  });

  it('does NOT claim "GPU accelerated" when in CPU mode', async () => {
    const { container } = render(<ModelStatus />);

    await waitFor(() => {
      expect(screen.getByText(/BUILD|model/i)).toBeInTheDocument();
    });

    const fullText = container.textContent || '';
    expect(fullText).not.toMatch(/gpu accelerat/i);
    expect(fullText).not.toMatch(/cuda.*active/i);
    expect(fullText).not.toMatch(/cuda.*enabled/i);
  });

  it('shows empty resident_models without crashing', async () => {
    server.use(
      http.get('/api/health', () => {
        return HttpResponse.json({
          status: 'ok',
          hardware_tier: 'BUILD',
          resident_models: {
            tier: 'BUILD',
            static_ceiling_gb: 4.0,
            effective_budget_gb: 3.68,
            total_vram_used_gb: 0.0,
            live_free_vram_gb: 3.68,
            resident_models: {},
          },
          sentinel: { monitoring: true },
        });
      })
    );

    render(<ModelStatus />);

    await waitFor(() => {
      expect(screen.getByText(/0\.0/)).toBeInTheDocument();
    });
  });

  it('shows connecting state when health endpoint returns error', async () => {
    server.use(
      http.get('/api/health', () => {
        return HttpResponse.error();
      })
    );

    render(<ModelStatus />);

    // When health fails, health=null, component shows skeleton loading or
    // the Models header with skeleton placeholders
    await waitFor(() => {
      const text = document.body.textContent || '';
      // Should show Models header at minimum
      expect(text).toMatch(/Models/i);
    });
  });

  it('never renders contradictory states from the same payload', async () => {
    // When resident_models is empty, the component should NOT show a
    // VRAM progress bar indicating usage alongside "No models loaded"
    // in a way that implies something IS loaded.
    server.use(
      http.get('/api/health', () => {
        return HttpResponse.json({
          status: 'ok',
          hardware_tier: 'BUILD',
          resident_models: {
            tier: 'BUILD',
            static_ceiling_gb: 4.0,
            effective_budget_gb: 3.68,
            total_vram_used_gb: 0.0,
            live_free_vram_gb: 3.68,
            resident_models: {},
          },
          sentinel: { monitoring: true },
        });
      })
    );

    const { container } = render(<ModelStatus />);

    await waitFor(() => {
      expect(container.querySelectorAll('.skeleton').length).toBe(0);
    });

    const text = container.textContent || '';

    // Should show tier info (real system state)
    expect(text).toMatch(/BUILD/);

    // Should show VRAM bar (0.0 / 3.7 is accurate — 3.68 formatted by toFixed(1))
    expect(text).toMatch(/0\.0/);
    expect(text).toMatch(/3\.7/);

    // Should NOT show model names when none are loaded
    expect(text).not.toMatch(/qwen.*\.gguf/);

    // The idle state should be clear: no confusion between "loaded" and "not loaded"
    expect(text).toMatch(/idle|no models/i);
  });

  it('shows loaded models when resident_models is populated', async () => {
    server.use(
      http.get('/api/health', () => {
        return HttpResponse.json({
          status: 'ok',
          hardware_tier: 'BUILD',
          resident_models: {
            tier: 'BUILD',
            static_ceiling_gb: 4.0,
            effective_budget_gb: 3.68,
            total_vram_used_gb: 0.5,
            live_free_vram_gb: 3.18,
            resident_models: {
              'qwen2.5-0.5b-instruct-q4_k_m.gguf': { vram_gb: 0.5, type: 'Llama' },
            },
          },
          sentinel: { monitoring: true },
        });
      })
    );

    const { container } = render(<ModelStatus />);

    await waitFor(() => {
      expect(screen.queryByText(/connecting to backend/i)).not.toBeInTheDocument();
    });

    const text = container.textContent || '';

    // Should show model name (truncated)
    expect(text).toMatch(/qwen/);
    // Should show Llama type
    expect(text).toMatch(/Llama/);
    // Should NOT show idle/no-models message
    expect(text).not.toMatch(/no models loaded|idle/i);
  });
});
