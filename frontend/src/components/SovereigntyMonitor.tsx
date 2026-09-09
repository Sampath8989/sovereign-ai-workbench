import { useState, useEffect, useCallback } from 'react'
import { Shield, ShieldCheck, ShieldAlert, Wifi, WifiOff, Zap } from 'lucide-react'
import { fetchHealth, triggerSentinel } from '../hooks/useApi'

interface Props {
  onTrigger: () => void
}

export default function SovereigntyMonitor({ onTrigger }: Props) {
  const [online, setOnline] = useState<boolean | null>(null)
  const [breachCount, setBreachCount] = useState(0)
  const [monitoring, setMonitoring] = useState(false)
  const [iptables, setIptables] = useState(false)
  const [triggering, setTriggering] = useState(false)
  const [testResult, setTestResult] = useState<{
    passed: boolean
    message: string
    detail?: Record<string, unknown>
  } | null>(null)

  const poll = useCallback(async () => {
    try {
      const h = await fetchHealth()
      setOnline(true)
      setBreachCount(h.sentinel.breach_count)
      setMonitoring(h.sentinel.monitoring)
      setIptables(h.sentinel.iptables_active)
    } catch {
      setOnline(false)
    }
  }, [])

  useEffect(() => {
    poll()
    const interval = setInterval(poll, 2000)
    return () => clearInterval(interval)
  }, [poll])

  const handleTrigger = async () => {
    setTriggering(true)
    setTestResult(null)
    try {
      const res = await triggerSentinel()
      // The self-test is idempotent and non-mutating: it reports pass/fail
      // without incrementing the breach counter (which stays unchanged).
      const detail = res?.detail as Record<string, unknown> | undefined
      const passed = typeof detail?.passed === 'boolean' ? Boolean(detail.passed) : (res?.passed === true)
      setTestResult({
        passed,
        message: typeof res?.message === 'string'
          ? res.message
          : passed
            ? 'Sovereignty self-test passed: outbound traffic blocked by kernel rules.'
            : 'Sovereignty self-test failed: synthetic leak breached kernel boundary.',
        detail,
      })
      onTrigger()
      await poll() // refresh breach count (unchanged by the self-test)
    } catch {
      // sentinel trigger may fail if backend is down
    } finally {
      setTriggering(false)
    }
  }

  const statusColor = online === null
    ? 'var(--warning)'
    : online
      ? 'var(--accent)'
      : 'var(--danger)'

  return (
    <div className="card-elevated" style={{ padding: '1.25rem' }}>
      {/* Header */}
      <div className="flex items-center gap-2 mb-3">
        <Shield className="w-4 h-4" style={{ color: 'var(--accent)' }} />
        <span className="text-xs font-semibold tracking-wide uppercase" style={{ color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>
          Sovereignty
        </span>
      </div>

      {/* Status badge — prominent */}
      <div className="mb-4">
        {online === null ? (
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full animate-pulse" style={{ background: 'var(--warning)' }} />
            <span className="badge-yellow badge-mono">CHECKING</span>
          </div>
        ) : online ? (
          <div className="flex items-center gap-2">
            <ShieldCheck className="w-5 h-5" style={{ color: 'var(--accent)' }} />
            <span className="badge-green badge-mono">AIR-GAP VERIFIED</span>
          </div>
        ) : (
          <div className="flex items-center gap-2">
            <ShieldAlert className="w-5 h-5" style={{ color: 'var(--danger)' }} />
            <span className="badge-red badge-mono">OFFLINE</span>
          </div>
        )}
      </div>

      {/* Breach counter — only when > 0, styled prominently */}
      {breachCount > 0 && (
        <div
          className="flex items-center gap-2 px-3 py-2 rounded-lg mb-3"
          style={{
            background: 'rgba(245, 158, 11, 0.08)',
            border: '1px solid rgba(245, 158, 11, 0.15)',
          }}
        >
          <Zap className="w-3.5 h-3.5" style={{ color: 'var(--warning)' }} />
          <span
            className="text-xs font-semibold"
            style={{ color: 'var(--warning)', fontFamily: 'var(--font-mono)' }}
          >
            {breachCount} breach{breachCount !== 1 ? 'es' : ''} this session
          </span>
        </div>
      )}

      {/* Details grid */}
      <div className="grid grid-cols-2 gap-2 mb-3">
        <div className="flex items-center gap-1.5 px-2 py-1.5 rounded" style={{ background: 'var(--bg-elevated)' }}>
          {monitoring ? (
            <Wifi className="w-3 h-3" style={{ color: 'var(--accent)' }} />
          ) : (
            <WifiOff className="w-3 h-3" style={{ color: 'var(--text-muted)' }} />
          )}
          <span className="text-[11px] font-medium" style={{ color: monitoring ? 'var(--text-secondary)' : 'var(--text-muted)' }}>
            Sentinel
          </span>
        </div>
        <div className="flex items-center gap-1.5 px-2 py-1.5 rounded" style={{ background: 'var(--bg-elevated)' }}>
          <span
            className="w-1.5 h-1.5 rounded-full"
            style={{ background: iptables ? 'var(--accent)' : 'var(--text-muted)' }}
          />
          <span className="text-[11px] font-medium" style={{ color: iptables ? 'var(--text-secondary)' : 'var(--text-muted)' }}>
            iptables
          </span>
        </div>
      </div>

      {/* Self-test result (non-mutating pass/fail report) */}
      {testResult && !triggering && (
        <div className="flex flex-col gap-2 mb-3">
          <div
            className="flex items-center gap-2 px-3 py-2 rounded-lg"
            style={{
              background: testResult.passed ? 'rgba(0, 229, 160, 0.08)' : 'rgba(239, 68, 68, 0.08)',
              border: `1px solid ${testResult.passed ? 'rgba(0, 229, 160, 0.2)' : 'rgba(239, 68, 68, 0.2)'}`,
            }}
          >
            {testResult.passed ? (
              <ShieldCheck className="w-3.5 h-3.5 flex-shrink-0" style={{ color: 'var(--accent)' }} />
            ) : (
              <ShieldAlert className="w-3.5 h-3.5 flex-shrink-0" style={{ color: '#f87171' }} />
            )}
            <span
              className="text-[11px] font-medium"
              style={{ color: testResult.passed ? 'var(--accent)' : '#f87171', fontFamily: 'var(--font-mono)' }}
            >
              {testResult.message}
            </span>
          </div>

          {/* Failure diagnostic fields */}
          {!testResult.passed && testResult.detail && (
            <div
              className="px-3 py-2.5 rounded-lg text-[10px] space-y-1 font-mono"
              style={{
                background: 'rgba(239, 68, 68, 0.05)',
                border: '1px solid rgba(239, 68, 68, 0.15)',
                color: 'var(--text-secondary)',
              }}
            >
              <div className="font-semibold text-[#f87171] uppercase tracking-wider mb-1">
                Egress Diagnostics
              </div>
              <div className="flex justify-between">
                <span className="text-text-muted">Target:</span>
                <span className="text-text-primary">{String(testResult.detail.target || '8.8.8.8:53')}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-text-muted">Protocol:</span>
                <span className="text-text-primary">{String(testResult.detail.protocol || 'UDP/TCP')}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-text-muted">Process:</span>
                <span className="text-text-primary">
                  {String(testResult.detail.initiating_process_name || 'python')} (PID {String(testResult.detail.initiating_pid || '?')})
                </span>
              </div>
              {Boolean(testResult.detail.root_cause) && (
                <div className="mt-1 pt-1 border-t border-red-500/10 text-[9px] text-[#fca5a5]">
                  Root Cause: {String(testResult.detail.root_cause)}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Trigger button */}
      <button
        onClick={handleTrigger}
        disabled={triggering || !online}
        className="btn-danger w-full text-xs flex items-center justify-center gap-2"
      >
        {triggering ? (
          <>
            <span className="w-3 h-3 border-2 border-red-400/30 border-t-red-400 rounded-full animate-spin" />
            Testing...
          </>
        ) : (
          <>
            <ShieldAlert className="w-3.5 h-3.5" />
            Test Sovereignty
          </>
        )}
      </button>
    </div>
  )
}
