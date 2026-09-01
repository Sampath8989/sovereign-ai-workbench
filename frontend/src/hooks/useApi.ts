import axios from 'axios'

const API_BASE = '/api'

const api = axios.create({
  baseURL: API_BASE,
  timeout: 120000, // 120s for local multi-step LLM inference
  headers: { 'Content-Type': 'application/json' },
})

export interface HealthResponse {
  status: string
  os: string
  hardware_tier: string
  max_vram_gb: number
  resident_models: {
    tier: string
    static_ceiling_gb: number
    effective_budget_gb: number
    live_free_vram_gb: number | null
    total_vram_used_gb: number
    resident_models: Record<string, { vram_gb: number; type: string }>
  }
  sentinel: {
    monitoring: boolean
    os: string
    ebpf_available: boolean
    psutil_available: boolean
    iptables_active: boolean
    allow_list: string[]
    breach_count: number
    tracked_pids: number[]
    enforce_kills: boolean
  }
}

export interface ChatResponse {
  response: string
  model_used?: string
  trace?: string[]
}

export interface UploadResponse {
  status: string
  filename: string
  path: string
  size_bytes: number
}

export interface IngestResponse {
  status: string
  files_processed: number
  chunks_added: number
}

export interface SentinelResponse {
  status: string
  detail: Record<string, unknown>
}

export interface AuditEntry {
  timestamp: number
  sequence: number
  event_type: string
  details: Record<string, unknown>
}

// Health check
export async function fetchHealth(): Promise<HealthResponse> {
  const { data } = await api.get<HealthResponse>('/health')
  return data
}

// Chat
export async function sendChat(prompt: string, role: string = 'engineer'): Promise<ChatResponse> {
  const { data } = await api.post<ChatResponse>('/chat', { prompt }, { params: { role } })
  return data
}

// Sentinel test
export async function triggerSentinel(): Promise<SentinelResponse> {
  const { data } = await api.post<SentinelResponse>('/test/sentinel')
  return data
}

// Audit log
export async function fetchAuditLog(): Promise<{ entries: AuditEntry[] }> {
  const { data } = await api.get<{ entries: AuditEntry[] }>('/audit/log')
  return data
}

// Audit chain check
export async function verifyAudit(): Promise<{ valid: boolean; entry_count: number; details: string }> {
  const { data } = await api.post<{ valid: boolean; entry_count: number; details: string }>('/test/audit')
  return data
}

// Ingest
export async function ingestDirectory(directory: string): Promise<IngestResponse> {
  const { data } = await api.post<IngestResponse>('/ingest', { directory })
  return data
}

// Upload file to sandbox
export async function uploadFile(file: File): Promise<UploadResponse> {
  const formData = new FormData()
  formData.append('file', file)
  const { data } = await api.post<UploadResponse>(`/upload?target_filename=${encodeURIComponent(file.name)}`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

// Build download URL
export function getDownloadUrl(filename: string): string {
  return `${API_BASE}/download?filename=${encodeURIComponent(filename)}`
}

export default api
