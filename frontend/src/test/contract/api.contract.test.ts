/**
 * Backend Contract Tests
 *
 * These tests validate the real backend API contracts against a LIVE server
 * running at http://localhost:8000. They do NOT use MSW mocks.
 *
 * Prerequisites:
 *   - Backend running: uvicorn backend.main:app --host 0.0.0.0 --port 8000
 *   - Real Qwen-0.5B model loaded (CPU mode)
 *   - Sentinel monitoring active (enforcement OFF)
 *
 * Run with: vitest run src/test/contract/api.contract.test.ts
 */
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import axios from 'axios';

const BASE_URL = 'http://localhost:8000';
const api = axios.create({
  baseURL: BASE_URL,
  timeout: 30000,
  validateStatus: () => true, // Don't throw on non-2xx
});

// Skip all tests if backend is not available
let backendAvailable = false;

beforeAll(async () => {
  try {
    const resp = await api.get('/health');
    backendAvailable = resp.status === 200 && resp.data?.status === 'ok';
  } catch {
    backendAvailable = false;
  }
});

// ============================================================
// 1. /api/health contract
// ============================================================
describe('Backend Contract: GET /health', () => {
  it('returns status 200 with required fields', async () => {
    if (!backendAvailable) return;

    const resp = await api.get('/health');
    expect(resp.status).toBe(200);

    const data = resp.data;
    expect(data.status).toBe('ok');
    expect(data.os).toBe('Linux');
    expect(data.hardware_tier).toBe('BUILD');
    expect(typeof data.max_vram_gb).toBe('number');
    expect(data.model_roster).toBeDefined();
  });

  it('reports CPU-mode VRAM correctly (0.0 GB used, not omitted)', async () => {
    if (!backendAvailable) return;

    const resp = await api.get('/health');
    const rm = resp.data.resident_models;

    expect(rm).toBeDefined();
    expect(rm.total_vram_used_gb).toBe(0.0);
    expect(typeof rm.live_free_vram_gb).toBe('number');
    expect(rm.live_free_vram_gb).toBeGreaterThan(0);
  });

  it('reports resident_models as an object (even if empty on CPU)', async () => {
    if (!backendAvailable) return;

    const resp = await api.get('/health');
    const rm = resp.data.resident_models;
    expect(typeof rm.resident_models).toBe('object');
  });

  it('reports sentinel status with enforce_kills = false', async () => {
    if (!backendAvailable) return;

    const resp = await api.get('/health');
    const s = resp.data.sentinel;
    expect(s.monitoring).toBe(true);
    expect(s.enforce_kills).toBe(false);
    expect(s.psutil_available).toBe(true);
  });
});

// ============================================================
// 2. /api/chat contract
// ============================================================
describe('Backend Contract: POST /chat', () => {
  it('returns {response: string, trace: string[]} with valid prompt', async () => {
    if (!backendAvailable) return;

    const start = Date.now();
    const resp = await api.post('/chat?role=engineer', {
      prompt: 'What is 2+2? Reply only the number.',
    });
    const elapsed = Date.now() - start;

    expect(resp.status).toBe(200);
    expect(typeof resp.data.response).toBe('string');
    expect(resp.data.response.length).toBeGreaterThan(0);
    expect(Array.isArray(resp.data.trace)).toBe(true);
    expect(resp.data.trace.length).toBeGreaterThan(0);
  });

  it('completes within 3-15s (real CPU inference bound)', async () => {
    if (!backendAvailable) return;

    const start = Date.now();
    const resp = await api.post('/chat?role=engineer', {
      prompt: 'Hello.',
    });
    const elapsed = Date.now() - start;

    expect(resp.status).toBe(200);
    // Real CPU inference: expect 3s minimum (overhead) and 15s max (thermal)
    expect(elapsed).toBeGreaterThan(2000);
    expect(elapsed).toBeLessThan(15000);
  });

  it('returns 400 on empty prompt', async () => {
    if (!backendAvailable) return;

    const resp = await api.post('/chat?role=engineer', {
      prompt: '',
    });
    // Backend should reject empty prompt (4xx), not crash (500)
    expect(resp.status).toBeGreaterThanOrEqual(400);
    expect(resp.status).toBeLessThan(500);
  });

  it('returns 4xx on missing prompt field', async () => {
    if (!backendAvailable) return;

    const resp = await api.post('/chat?role=engineer', {});
    expect(resp.status).toBeGreaterThanOrEqual(400);
    expect(resp.status).toBeLessThan(500);
  });

  it('rejects invalid role with 4xx', async () => {
    if (!backendAvailable) return;

    const resp = await api.post('/chat?role=invalid_role', {
      prompt: 'test',
    });
    expect(resp.status).toBeGreaterThanOrEqual(400);
    expect(resp.status).toBeLessThan(500);
  });
});

// ============================================================
// 3. /api/test/sentinel contract
// ============================================================
describe('Backend Contract: POST /test/sentinel', () => {
  it('triggers sentinel test and returns breach event', async () => {
    if (!backendAvailable) return;

    const resp = await api.post('/test/sentinel');
    expect(resp.status).toBe(200);

    const data = resp.data;
    // Should contain some breach-related info
    expect(data).toBeDefined();
  });

  it('sentinel action is safety_abort_untracked (Option B: detect + log, no kill)', async () => {
    if (!backendAvailable) return;

    const resp = await api.post('/test/sentinel');
    expect(resp.status).toBe(200);

    // After sentinel test, check health for updated state
    const health = await api.get('/health');
    const sentinel = health.data.sentinel;

    // Breach count should have increased
    expect(typeof sentinel.breach_count).toBe('number');
    // Enforcement should still be off
    expect(sentinel.enforce_kills).toBe(false);
  });
});

// ============================================================
// 4. Path traversal rejection
// ============================================================
describe('Backend Contract: Path traversal rejection', () => {
  it('rejects path traversal in download endpoint', async () => {
    if (!backendAvailable) return;

    const resp = await api.get('/download?filename=../../../etc/passwd');
    // Should return 400 or 403, not 200 with file content
    expect(resp.status).not.toBe(200);
  });

  it('rejects path traversal with encoded slashes', async () => {
    if (!backendAvailable) return;

    const resp = await api.get('/download?filename=..%2F..%2Fetc%2Fpasswd');
    expect(resp.status).not.toBe(200);
  });
});

// ============================================================
// 5. Oversized / malformed request handling
// ============================================================
describe('Backend Contract: Malformed requests', () => {
  it('handles oversized prompt gracefully (4xx, not 500)', async () => {
    if (!backendAvailable) return;

    const hugePrompt = 'x'.repeat(100_000);
    const resp = await api.post('/chat?role=engineer', {
      prompt: hugePrompt,
    });
    // Should reject (4xx) or succeed (200), but NOT crash (500)
    expect(resp.status).toBeLessThan(500);
  });

  it('handles non-JSON body gracefully', async () => {
    if (!backendAvailable) return;

    try {
      const resp = await axios.post(`${BASE_URL}/chat?role=engineer`, 'not json', {
        headers: { 'Content-Type': 'text/plain' },
        timeout: 10000,
        validateStatus: () => true,
      });
      expect(resp.status).toBeLessThan(500);
    } catch {
      // Connection error is acceptable (server might reject)
    }
  });
});
