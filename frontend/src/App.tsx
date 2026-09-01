import { useState, useCallback } from 'react'
import { Shield } from 'lucide-react'
import ChatCanvas from './components/ChatCanvas'
import DeliverableViewer from './components/DeliverableViewer'
import SovereigntyMonitor from './components/SovereigntyMonitor'
import ModelStatus from './components/ModelStatus'
import AgentTrace from './components/AgentTrace'
import RoleSwitcher from './components/RoleSwitcher'
import { sendChat, type ChatResponse } from './hooks/useApi'

export default function App() {
  const [role, setRole] = useState<'engineer' | 'manager'>('engineer')
  const [chatResponse, setChatResponse] = useState<ChatResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [sentinelTriggered, setSentinelTriggered] = useState(false)

  const handleSend = useCallback(async (prompt: string) => {
    setLoading(true)
    setError(null)
    try {
      const res = await sendChat(prompt, role)
      setChatResponse(res)
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Request failed'
      setError(msg)
    } finally {
      setLoading(false)
    }
  }, [role])

  return (
    <div className="h-screen flex flex-col" style={{ background: 'var(--bg-base)' }}>
      {/* Top bar */}
      <header
        className="flex items-center justify-between px-6 py-3"
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
              SIH 2026 · Air-Gapped Local Hardware Execution
            </p>
          </div>
        </div>
        <RoleSwitcher role={role} onChange={setRole} />
      </header>

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left pane: Chat + Deliverables */}
        <div className="flex-[7] flex flex-col" style={{ borderRight: '1px solid var(--border-subtle)' }}>
          <div className="flex-1 overflow-hidden">
            <ChatCanvas
              onSend={handleSend}
              response={chatResponse}
              loading={loading}
              error={error}
              role={role}
            />
          </div>
          {chatResponse?.response && (
            <DeliverableViewer response={chatResponse.response} />
          )}
        </div>

        {/* Right pane: Sidebar */}
        <div
          className="flex-[3] flex flex-col gap-3.5 p-4 overflow-y-auto"
          style={{
            background: 'var(--bg-surface)',
            backdropFilter: 'blur(16px)',
          }}
        >
          <SovereigntyMonitor onTrigger={() => setSentinelTriggered(true)} />
          <ModelStatus />
          <AgentTrace trace={chatResponse?.trace} />
        </div>
      </div>
    </div>
  )
}
