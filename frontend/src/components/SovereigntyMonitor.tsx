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
    try {
      await triggerSentinel()
      onTrigger()
      await poll() // refresh breach count
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
