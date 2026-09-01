import '@testing-library/jest-dom/vitest';
import { afterEach, beforeAll, afterAll, vi } from 'vitest';
import { cleanup } from '@testing-library/react';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';

// Polyfill scrollIntoView for jsdom
if (!Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = function() {};
}

// Mock API handlers
const handlers = [
  http.get('/api/health', () => {
    return HttpResponse.json({
      status: 'ok',
      os: 'Linux',
      hardware_tier: 'BUILD',
      max_vram_gb: 4.0,
      model_roster: {
        'qwen2.5-0.5b-instruct-q4_k_m.gguf': 0.5,
        'qwen2.5-coder-3b-instruct-q4_k_m.gguf': 1.5,
      },
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
        allow_list: ['127.0.0.1', '0.0.0.0', '::1'],
        breach_count: 0,
        tracked_pids: [],
        seen_connections: 0,
        enforce_kills: false,
        poll_interval: 0.2,
      },
    });
  }),

  http.post('/api/chat', async () => {
    return HttpResponse.json({
      response: 'The sum of 2+2 is 4.',
      trace: [
        'Planner: Decomposed into 1 step(s)',
        'Executor: llm.summarize()',
        'Retriever: No matching sources found',
        'Verifier: Grounding check incomplete (no sources)',
        'Synthesizer: Generated final response',
      ],
    });
  }),

  http.post('/api/test/sentinel', () => {
    return HttpResponse.json({
      status: 'test_completed',
      breach_event: {
        timestamp: new Date().toISOString(),
        action: 'safety_abort_untracked',
        pid: 12345,
        connection: { dest_ip: '8.8.8.8', dest_port: 53 },
        message: 'External connection detected. Process not in tracked set. Logged only (enforcement disabled).',
      },
    });
  }),

  http.get('/api/download', () => {
    return new HttpResponse('fake file content', {
      headers: {
        'Content-Type': 'application/octet-stream',
        'Content-Disposition': 'attachment; filename="test.docx"',
      },
    });
  }),

  // Catch-all for unmatched requests
  http.all('*', ({ request }) => {
    console.warn(`Unhandled request: ${request.method} ${request.url}`);
    return new HttpResponse(null, { status: 404 });
  }),
];

export const server = setupServer(...handlers);

beforeAll(() => server.listen({ onUnhandledRequest: 'warn' }));
afterEach(() => {
  cleanup();
  server.resetHandlers();
});
afterAll(() => server.close());
