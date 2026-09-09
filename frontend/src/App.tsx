import { useState, useCallback, useEffect } from 'react'
import { Shield } from 'lucide-react'
import ChatCanvas, { type ChatMessage } from './components/ChatCanvas'
import DeliverableViewer from './components/DeliverableViewer'
import SovereigntyMonitor from './components/SovereigntyMonitor'
import ModelStatus from './components/ModelStatus'
import AgentTrace from './components/AgentTrace'
import RoleSwitcher from './components/RoleSwitcher'
import ModelSelector from './components/ModelSelector'
import SessionSidebar from './components/SessionSidebar'
import { sendChat, fetchSession, type ChatResponse } from './hooks/useApi'

export default function App() {
  const [role, setRole] = useState<'engineer' | 'manager'>('engineer')
  const [selectedModel, setSelectedModel] = useState<string>('auto')
  const [chatResponse, setChatResponse] = useState<ChatResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [sentinelTriggered, setSentinelTriggered] = useState(false)

  // Session and project persistence state
  const [activeSessionId, setActiveSessionId] = useState<string | null>(() => {
    return localStorage.getItem('active_session_id') || null
  })
  const [activeProjectId, setActiveProjectId] = useState<string | null>(() => {
    return localStorage.getItem('active_project_id') || null
  })
  const [sessionMessages, setSessionMessages] = useState<ChatMessage[]>([])
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)

  // Rehydrate conversation from local store on mount or session switch
  useEffect(() => {
    if (activeSessionId) {
      fetchSession(activeSessionId)
        .then((detail) => {
          if (detail && detail.messages) {
            const msgs: ChatMessage[] = detail.messages.map((m) => ({
              role: m.role,
              content: m.content,
              model_used: m.model_used,
              deliverables: m.deliverables,
            }))
            setSessionMessages(msgs)
            const lastAssistant = [...detail.messages].reverse().find((m) => m.role === 'assistant')
            if (lastAssistant) {
              setChatResponse({
                response: lastAssistant.content,
                model_used: lastAssistant.model_used,
                trace: lastAssistant.trace,
                deliverables: lastAssistant.deliverables,
                session_id: detail.id,
              })
            }
          }
        })
        .catch((err) => {
          console.warn('Could not restore session:', err)
          localStorage.removeItem('active_session_id')
          setActiveSessionId(null)
          setSessionMessages([])
          setChatResponse(null)
        })
    } else {
      setSessionMessages([])
      setChatResponse(null)
    }
  }, [activeSessionId])

  const handleSelectSession = (sid: string | null) => {
    if (!sid) {
      localStorage.removeItem('active_session_id')
      setActiveSessionId(null)
      setSessionMessages([])
      setChatResponse(null)
    } else {
      localStorage.setItem('active_session_id', sid)
      setActiveSessionId(sid)
    }
  }

  const handleSelectProject = (pid: string | null) => {
    if (!pid) {
      localStorage.removeItem('active_project_id')
      setActiveProjectId(null)
    } else {
      localStorage.setItem('active_project_id', pid)
      setActiveProjectId(pid)
    }
  }

  const handleSend = useCallback(async (prompt: string) => {
    setLoading(true)
    setError(null)
    try {
      const targetProjectId = activeProjectId === 'standalone' ? undefined : (activeProjectId || undefined)
      const res = await sendChat(prompt, role, selectedModel, activeSessionId || undefined, targetProjectId)
      setChatResponse(res)
      if (res.session_id) {
        setActiveSessionId(res.session_id)
        localStorage.setItem('active_session_id', res.session_id)
      }
    } catch (err: unknown) {
      let msg = err instanceof Error ? err.message : 'Request failed'
      if (typeof err === 'object' && err !== null) {
        const anyErr = err as { code?: string; message?: string }
        if (anyErr.code === 'ECONNABORTED' || (anyErr.message && anyErr.message.toLowerCase().includes('timeout'))) {
          msg = 'Inference timed out. The local CPU model is processing a complex multi-part query. Try selecting a faster 3B model or shortening the prompt.'
        }
      }
      setError(msg)
    } finally {
      setLoading(false)
    }
  }, [role, selectedModel, activeSessionId, activeProjectId])

  return (
    <div className="h-screen flex flex-col" style={{ background: 'var(--bg-base)' }}>
      {/* Top bar */}
      <header
        className="relative z-50 flex items-center justify-between px-6 py-3"
        style={{
          background: 'var(--bg-surface)',
          backdropFilter: 'blur(16px)',
          borderBottom: '1px solid var(--border-subtle)',
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.25)',
        }}
      >
        <div className="flex items-center gap-3">
          <div
            className="w-9 h-9 rounded-xl flex items-center justify-center"
            style={{
              background: 'rgba(0, 229, 160, 0.1)',
              border: '1px solid rgba(0, 229, 160, 0.25)',
              boxShadow: '0 0 16px rgba(0, 229, 160, 0.15)',
            }}
          >
            <Shield className="w-5 h-5" style={{ color: 'var(--accent)' }} />
          </div>
          <div>
            <h1 className="text-sm font-bold tracking-tight" style={{ color: 'var(--text-primary)' }}>
              Sovereign AI Workbench
            </h1>
            <p className="text-[10px] font-medium flex items-center gap-1.5" style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
              <span className="w-1.5 h-1.5 rounded-full" style={{ background: 'var(--accent)' }} />
              Air-Gapped Local Hardware Execution · Linux Pop!_OS & 3B / 7B / 14B Ready
            </p>
          </div>
        </div>

        {/* Controls: Model Engine Dropdown + Role Switcher */}
        <div className="flex items-center gap-2.5">
          <ModelSelector
            selectedModel={selectedModel}
            onSelectModel={setSelectedModel}
          />
          <RoleSwitcher role={role} onChange={setRole} />
        </div>
      </header>

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Leftmost pane: Session History Sidebar */}
        <SessionSidebar
          activeSessionId={activeSessionId}
          onSelectSession={handleSelectSession}
          activeProjectId={activeProjectId}
          onSelectProject={handleSelectProject}
          collapsed={sidebarCollapsed}
          onToggleCollapse={() => setSidebarCollapsed((prev) => !prev)}
        />

        {/* Center pane: Chat + Deliverables */}
        <div className="flex-[7] flex flex-col min-w-0" style={{ borderRight: '1px solid var(--border-subtle)' }}>
          <div className="flex-1 overflow-hidden">
            <ChatCanvas
              onSend={handleSend}
              response={chatResponse}
              loading={loading}
              error={error}
              role={role}
              sessionId={activeSessionId}
              initialMessages={sessionMessages}
            />
          </div>
          {chatResponse && (
            <DeliverableViewer response={chatResponse.response} deliverables={chatResponse.deliverables} />
          )}
        </div>

        {/* Right pane: Sovereignty + Models + Trace Sidebar */}
        <div
          className="flex-[3] flex flex-col gap-3.5 p-4 overflow-y-auto"
          style={{
            background: 'var(--bg-surface)',
            backdropFilter: 'blur(16px)',
          }}
        >
          <SovereigntyMonitor onTrigger={() => setSentinelTriggered(true)} />
          <ModelStatus selectedModel={selectedModel} />
          <AgentTrace trace={chatResponse?.trace} />
        </div>
      </div>
    </div>
  )
}
