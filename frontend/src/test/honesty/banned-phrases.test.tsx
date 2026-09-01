/**
 * Honesty/Copy Regression Tests
 *
 * These tests guard against the exact class of bug that already happened once:
 * oversold claims about sentinel behavior, GPU usage, and blocking guarantees.
 *
 * If any component renders banned phrases, these tests will catch it immediately.
 * This is a permanent guard rail against regression.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, act } from '@testing-library/react';
import { http, HttpResponse } from 'msw';
import { server } from '../setup';

// Import all components that might contain claims
import SovereigntyMonitor from '../../components/SovereigntyMonitor';
import ModelStatus from '../../components/ModelStatus';
import ChatCanvas from '../../components/ChatCanvas';
import AgentTrace from '../../components/AgentTrace';
import DeliverableViewer from '../../components/DeliverableViewer';
import RoleSwitcher from '../../components/RoleSwitcher';

// ============================================================
// BANNED PHRASES — edit this list to add new restrictions
// ============================================================
const BANNED_PHRASES = [
  // Sentinel: must NOT claim blocking/killing (Option B = detect + log only)
  { pattern: /sigkill/i, category: 'sentinel-blocking', component: 'any' },
  { pattern: /process.*killed/i, category: 'sentinel-blocking', component: 'any' },
  { pattern: /process.*terminated/i, category: 'sentinel-blocking', component: 'any' },
  { pattern: /blocked.*traffic/i, category: 'sentinel-blocking', component: 'any' },
  { pattern: /prevents.*egress/i, category: 'sentinel-blocking', component: 'any' },
  { pattern: /blocks.*connection/i, category: 'sentinel-blocking', component: 'any' },
  { pattern: /enforces.*block/i, category: 'sentinel-blocking', component: 'any' },
  { pattern: /iptables.*block/i, category: 'sentinel-blocking', component: 'any' },
  { pattern: /no.*phone.*home/i, category: 'sentinel-blocking', component: 'any' },
  { pattern: /cannot.*phone/i, category: 'sentinel-blocking', component: 'any' },

  // GPU: must NOT claim GPU acceleration when CPU-only
  { pattern: /gpu accelerat/i, category: 'gpu-claims', component: 'ModelStatus' },
  { pattern: /cuda.*active/i, category: 'gpu-claims', component: 'ModelStatus' },
  { pattern: /cuda.*enabled/i, category: 'gpu-claims', component: 'ModelStatus' },

  // VRAM: must NOT imply VRAM is being used when 0.0
  { pattern: /vram.*in use/i, category: 'vram-claims', component: 'ModelStatus' },
];

// Static components (no API calls) — can test synchronously
const STATIC_COMPONENTS: Array<{
  name: string;
  Component: React.ComponentType<any>;
  props: Record<string, any>;
}> = [
  { name: 'ChatCanvas', Component: ChatCanvas, props: { onSend: () => {}, response: null, loading: false, error: null, role: 'engineer' } },
  { name: 'AgentTrace', Component: AgentTrace, props: { trace: ['Step 1', 'Step 2'] } },
  { name: 'DeliverableViewer', Component: DeliverableViewer, props: { response: 'File at workspace/outputs/test.docx' } },
  { name: 'RoleSwitcher', Component: RoleSwitcher, props: { role: 'engineer', onChange: () => {} } },
];

describe('Honesty: No banned phrases in static component text', () => {
  for (const { name: compName, Component, props } of STATIC_COMPONENTS) {
    it(`${compName}: no banned phrases in rendered text`, () => {
      const { container } = render(<Component {...props} />);

      const fullText = (container.textContent || '').toLowerCase();

      for (const { pattern, category, component } of BANNED_PHRASES) {
        if (component !== 'any' && component !== compName) continue;

        const match = fullText.match(pattern);
        expect(
          match,
          `${compName} contains banned phrase "${pattern}" (category: ${category}). ` +
          `Found: "${match?.[0] || 'null'}". This is an oversold claim that must be fixed.`
        ).toBeNull();
      }
    });
  }
});

// Async components (fetch /api/health) — need waitFor for API response
describe('Honesty: No banned phrases in async components', () => {
  it('SovereigntyMonitor: no banned phrases', async () => {
    const { container } = render(<SovereigntyMonitor onTrigger={() => {}} />);

    await waitFor(() => {
      expect(screen.getByText(/air-gap|breach|offline|checking/i)).toBeInTheDocument();
    });

    const fullText = (container.textContent || '').toLowerCase();

    for (const { pattern, category, component } of BANNED_PHRASES) {
      if (component !== 'any' && component !== 'SovereigntyMonitor') continue;

      const match = fullText.match(pattern);
      expect(
        match,
        `SovereigntyMonitor contains banned phrase "${pattern}" (category: ${category}). ` +
        `Found: "${match?.[0] || 'null'}".`
      ).toBeNull();
    }
  });

  it('ModelStatus: no banned phrases', async () => {
    const { container } = render(<ModelStatus />);

    await waitFor(() => {
      expect(screen.getByText(/build|tier|connecting/i)).toBeInTheDocument();
    });

    const fullText = (container.textContent || '').toLowerCase();

    for (const { pattern, category, component } of BANNED_PHRASES) {
      if (component !== 'any' && component !== 'ModelStatus') continue;

      const match = fullText.match(pattern);
      expect(
        match,
        `ModelStatus contains banned phrase "${pattern}" (category: ${category}). ` +
        `Found: "${match?.[0] || 'null'}".`
      ).toBeNull();
    }
  });
});

describe('Honesty: Sentinel badge language correctness', () => {
  it('SovereigntyMonitor shows honest language after sentinel trigger', async () => {
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

    const { container } = render(<SovereigntyMonitor onTrigger={() => {}} />);

    await waitFor(() => {
      expect(screen.getByText(/air-gap|breach|offline/i)).toBeInTheDocument();
    });

    const fullText = (container.textContent || '').toLowerCase();
    expect(fullText).not.toMatch(/sigkill/i);
    expect(fullText).not.toMatch(/process.*killed/i);
  });
});

describe('Honesty: ModelStatus VRAM display', () => {
  it('0.0GB VRAM is displayed as normal state, not error', async () => {
    render(<ModelStatus />);

    await waitFor(() => {
      expect(screen.getByText(/build|tier|connecting/i)).toBeInTheDocument();
    });

    const vramText = screen.queryByText(/0\.0/);
    if (vramText) {
      expect(vramText).not.toHaveClass(/error|text-red/);
    }
  });

  it('no GPU accelerated claims in text', async () => {
    const { container } = render(<ModelStatus />);

    await waitFor(() => {
      expect(screen.getByText(/build|tier|connecting/i)).toBeInTheDocument();
    });

    const fullText = (container.textContent || '').toLowerCase();
    expect(fullText).not.toMatch(/gpu accelerat/i);
    expect(fullText).not.toMatch(/cuda.*enabl/i);
  });
});

describe('Honesty: No false claims about accuracy', () => {
  it('AgentTrace does not claim specific accuracy percentages', () => {
    const trace = ['Planner: Decomposed', 'Synthesizer: Generated'];
    const { container } = render(<AgentTrace trace={trace} />);

    const fullText = (container.textContent || '').toLowerCase();
    expect(fullText).not.toMatch(/\d+\.?\d*%.*accurac/i);
    expect(fullText).not.toMatch(/99\.?\d*%/i);
    expect(fullText).not.toMatch(/100%.*accur/i);
  });
});
