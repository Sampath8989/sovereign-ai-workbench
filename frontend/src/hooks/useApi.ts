import axios from 'axios'

const API_BASE = '/api'

const api = axios.create({
  baseURL: API_BASE,
  timeout: 600000, // 600s (10 min) for local multi-step LLM inference on CPU
  headers: { 'Content-Type': 'application/json' },
})

// Surface the server's structured error body on ANY failed request instead of
// axios's generic "Request failed with status code 500". The backend reports
// model failures as HTTP 503 with {"error": "vision_model_oom", "detail": ...};
// validation failures use a plain string detail.
api.interceptors.response.use(
  (res) => res,
  (err: unknown) => {
    const axiosErr = err as { response?: { data?: unknown }; message?: string }
    const data = axiosErr.response?.data as
      | { detail?: unknown; error?: unknown }
      | undefined
    const detail = data?.detail
    let msg: string | null = null
    if (typeof detail === 'string' && detail) {
      msg = detail
    } else if (detail && typeof detail === 'object' && !Array.isArray(detail)) {
      const d = detail as { error?: unknown; detail?: unknown }
      if (d.error && d.detail) msg = `[${String(d.error)}] ${String(d.detail)}`
      else if (d.detail) msg = String(d.detail)
      else msg = JSON.stringify(detail)
    } else if (data?.error) {
      msg = String(data.error)
    }
    if (msg && axiosErr) {
      axiosErr.message = msg
    }
    return Promise.reject(err)
  }
)

export interface ModelInfo {
  id: string
  name: string
  category: string
  param_size?: string
  vram_gb: number
  size_gb: number
  description: string
  is_present: boolean
}

export interface ModelsResponse {
  models: ModelInfo[]
  default: string
  active?: string
}

export interface HealthResponse {
  status: string
  os: string
  hardware_tier: string
  max_vram_gb: number
  model_roster?: Record<string, number>
  available_models?: ModelInfo[]
  resident_models: {
    tier: string
    static_ceiling_gb: number
    effective_budget_gb: number
    live_free_vram_gb: number | null
    live_used_vram_gb?: number | null
    live_total_vram_gb?: number | null
    total_vram_used_gb: number
    pinned_model?: string | null
    resident_models: Record<string, { vram_gb: number; type: string; pinned?: boolean }>
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
  deliverables?: string[]
  session_id?: string
}

export interface SessionSummary {
  id: string
  title: string
  project_id: string | null
  created_at: number
  updated_at: number
  message_count: number
  last_message: string
}

export interface SessionMessage {
  id: number
  role: 'user' | 'assistant'
  content: string
  model_used?: string
  trace?: string[]
  deliverables?: string[]
  created_at: number
}

export interface SessionDetail {
  id: string
  title: string
  project_id: string | null
  created_at: number
  updated_at: number
  message_count: number
  messages: SessionMessage[]
}

export interface ProjectSummary {
  id: string
  name: string
  description?: string
  created_at: number
  updated_at: number
  session_count: number
}

export interface ProjectDetail extends ProjectSummary {
  sessions: Array<{
    id: string
    title: string
    created_at: number
    updated_at: number
  }>
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
  passed?: boolean
  message?: string
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

// Models list
export async function fetchModels(): Promise<ModelsResponse> {
  const { data } = await api.get<ModelsResponse>('/models')
  return data
}

// Chat
export async function sendChat(
  prompt: string,
  role: string = 'engineer',
  model: string = 'auto',
  sessionId?: string,
  projectId?: string
): Promise<ChatResponse> {
  const { data } = await api.post<ChatResponse>(
    '/chat',
    { prompt, model, session_id: sessionId, project_id: projectId },
    { params: { role }, timeout: 900000 }
  )
  return data
}

// Sessions
export async function fetchSessions(
  limit: number = 50,
  offset: number = 0,
  projectId?: string
): Promise<SessionSummary[]> {
  const params: Record<string, any> = { limit, offset }
  if (projectId) params.project_id = projectId
  const { data } = await api.get<SessionSummary[]>('/sessions', { params })
  return data
}

export async function fetchSession(sessionId: string): Promise<SessionDetail> {
  const { data } = await api.get<SessionDetail>(`/sessions/${encodeURIComponent(sessionId)}`)
  return data
}

export async function createSession(title?: string, projectId?: string): Promise<SessionDetail> {
  const { data } = await api.post<SessionDetail>('/sessions', { title, project_id: projectId })
  return data
}

export async function deleteSession(sessionId: string): Promise<{ status: string; session_id: string }> {
  const { data } = await api.delete<{ status: string; session_id: string }>(`/sessions/${encodeURIComponent(sessionId)}`)
  return data
}

export async function updateSessionTitle(sessionId: string, title: string): Promise<{ status: string; session_id: string; title: string }> {
  const { data } = await api.patch<{ status: string; session_id: string; title: string }>(`/sessions/${encodeURIComponent(sessionId)}`, { title })
  return data
}

// Projects
export async function fetchProjects(): Promise<ProjectSummary[]> {
  const { data } = await api.get<ProjectSummary[]>('/projects')
  return data
}

export async function createProject(name: string, description?: string): Promise<ProjectSummary> {
  const { data } = await api.post<ProjectSummary>('/projects', { name, description })
  return data
}

export async function deleteProject(projectId: string): Promise<{ status: string; project_id: string }> {
  const { data } = await api.delete<{ status: string; project_id: string }>(`/projects/${encodeURIComponent(projectId)}`)
  return data
}

export async function assignSessionToProject(
  sessionId: string,
  projectId: string | null
): Promise<{ status: string; session_id: string; project_id: string | null }> {
  const { data } = await api.patch<{ status: string; session_id: string; project_id: string | null }>(
    `/sessions/${encodeURIComponent(sessionId)}/project`,
    { project_id: projectId }
  )
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
// onProgress is invoked with a 0-100 percentage as bytes stream to the server,
// powering the visible uploading/progress state in the chat input bar.
export async function uploadFile(
  file: File,
  onProgress?: (percent: number) => void
): Promise<UploadResponse> {
  const formData = new FormData()
  formData.append('file', file)
  // Do NOT pin Content-Type to 'multipart/form-data' here: the browser (or
  // axios) must set the full header including the multipart boundary, or the
  // server cannot parse the body and the upload fails.
  try {
    const { data } = await api.post<UploadResponse>(`/upload?target_filename=${encodeURIComponent(file.name)}`, formData, {
      headers: { 'Content-Type': undefined },
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          onProgress(Math.round((progressEvent.loaded / progressEvent.total) * 100))
        }
      },
    })
    return { ...data, filename: data.filename || file.name }
  } catch (err: unknown) {
    // Surface the server's human-readable detail (e.g. size limit, traversal,
    // storage failure) instead of axios's generic "Request failed with status
    // code 500".
    const detail = (err as { response?: { data?: { detail?: unknown } } })?.response?.data?.detail
    if (typeof detail === 'string' && detail) {
      throw new Error(detail)
    }
    throw err
  }
}

// Build download URL
export function getDownloadUrl(filename: string): string {
  return `${API_BASE}/download?filename=${encodeURIComponent(filename)}`
}

export default api
