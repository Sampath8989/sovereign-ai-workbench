/**
 * SovereigntyMonitor.test.tsx
 *
 * Tests:
 * - Green badge on healthy /api/health
 * - On sentinel trigger: badge shows "detected" language, NOT "blocked"/"SIGKILL'd"
 * - Poll interval test with fake timers (2s)
 * - Error state when backend is down
 * - Honesty: no banned phrases about "blocking" or "SIGKILL"
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';
import { server } from '../setup';
import SovereigntyMonitor from '../../components/SovereigntyMonitor';

describe('SovereigntyMonitor', () => {
  beforeEach(() => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('shows green "AIR-GAP VERIFIED" badge when backend is healthy', async () => {
    const onTrigger = vi.fn();
    render(<SovereigntyMonitor onTrigger={onTrigger} />);

    await waitFor(() => {
      expect(screen.getByText(/air-gap verified/i)).toBeInTheDocument();
    });
  });

  it('polls /api/health every 2 seconds', async () => {
    const onTrigger = vi.fn();
    let pollCount = 0;

    server.use(
      http.get('/api/health', () => {
        pollCount++;
        return HttpResponse.json({
          status: 'ok',
          hardware_tier: 'BUILD',
          resident_models: {
            tier: 'BUILD',
            static_ceiling_gb: 4.0,
            effective_budget_gb: 3.68,
            live_free_vram_gb: 3.68,
            total_vram_used_gb: 0.0,
            resident_models: {},
          },
          sentinel: {
            monitoring: true,
            os: 'Linux',
            ebpf_available: true,
            psutil_available: true,
            iptables_active: false,
            allow_list: ['127.0.0.1'],
            breach_count: 0,
            tracked_pids: [],
            enforce_kills: false,
          },
        });
      })
    );

    render(<SovereigntyMonitor onTrigger={onTrigger} />);

    // Initial poll
    await waitFor(() => {
      expect(pollCount).toBeGreaterThanOrEqual(1);
    });

    // Advance by 2s
    await act(async () => {
      vi.advanceTimersByTime(2500);
    });

    // Should have polled again
    expect(pollCount).toBeGreaterThanOrEqual(2);
  });

  it('calls onTrigger callback after sentinel test', async () => {
    const onTrigger = vi.fn();
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });

    server.use(
      http.post('/api/test/sentinel', () => {
        return HttpResponse.json({
          status: 'test_completed',
          breach_event: {
            action: 'safety_abort_untracked',
            message: 'External connection detected. Logged only.',
          },
        });
      })
    );

    render(<SovereigntyMonitor onTrigger={onTrigger} />);

    await waitFor(() => {
      expect(screen.getByText(/air-gap verified/i)).toBeInTheDocument();
    });

    const testButton = screen.getByRole('button', { name: /test sovereignty|trigger|breach/i });
    await user.click(testButton);

    await waitFor(() => {
      expect(onTrigger).toHaveBeenCalled();
    });
  });

  it('shows red badge when backend is unreachable', async () => {
    const onTrigger = vi.fn();

    server.use(
      http.get('/api/health', () => {
        return HttpResponse.error();
      })
    );

    render(<SovereigntyMonitor onTrigger={onTrigger} />);

    await waitFor(() => {
      expect(screen.getByText(/offline|error|unreachable/i)).toBeInTheDocument();
    });

    expect(screen.queryByText(/air-gap verified/i)).not.toBeInTheDocument();
  });

  it('NO banned phrases about blocking/SIGKILL in component text', async () => {
    const onTrigger = vi.fn();
    const bannedPhrases = [
      /sigkill/i,
      /process.*killed/i,
      /blocked.*traffic/i,
      /prevents.*egress/i,
      /enforces.*block/i,
      /iptables.*block/i,
    ];

    const { container } = render(<SovereigntyMonitor onTrigger={onTrigger} />);

    await waitFor(() => {
      expect(screen.getByText(/air-gap|breach|offline/i)).toBeInTheDocument();
    });

    const fullText = container.textContent || '';

    for (const phrase of bannedPhrases) {
      expect(fullText).not.toMatch(phrase);
    }
  });

  it('shows monitoring status details', async () => {
    const onTrigger = vi.fn();
    render(<SovereigntyMonitor onTrigger={onTrigger} />);

    await waitFor(() => {
      expect(screen.getByText(/sentinel/i)).toBeInTheDocument();
    });

    // Redesigned component shows 'Sentinel' and 'iptables' as status labels
    expect(screen.getByText(/iptables/i)).toBeInTheDocument();
  });
});
