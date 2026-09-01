import { useState, useEffect, useCallback } from 'react'
import { Cpu, HardDrive, Server } from 'lucide-react'
import { fetchHealth, type HealthResponse } from '../hooks/useApi'

export default function ModelStatus() {
  const [health, setHealth] = useState<HealthResponse | null>(null)

  const poll = useCallback(async () => {
    try {
      const h = await fetchHealth()
      setHealth(h)
    } catch {
      setHealth(null)
    }
  }, [])

  useEffect(() => {
    poll()
    const interval = setInterval(poll, 5000)
    return () => clearInterval(interval)
  }, [poll])

  if (!health) {
    return (
      <div className="card">
        <div className="flex items-center gap-2 mb-2">
          <Server className="w-4 h-4" style={{ color: 'var(--text-muted)' }} />
          <span className="text-xs font-semibold tracking-wide uppercase" style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
            Models
          </span>
        </div>
        <div className="skeleton h-4 w-3/4 rounded mb-2" />
        <div className="skeleton h-3 w-1/2 rounded" />
      </div>
    )
  }

  const mm = health.resident_models
  const models = mm.resident_models
  const usedGB = mm.total_vram_used_gb
  const totalGB = mm.effective_budget_gb
  const freeGB = mm.live_free_vram_gb
  const modelsList = Object.entries(models)
  const hasModels = modelsList.length > 0
  const vramPercent = totalGB > 0 ? (usedGB / totalGB) * 100 : 0

  return (
    <div className="card">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Server className="w-4 h-4" style={{ color: hasModels ? 'var(--accent)' : 'var(--text-muted)' }} />
          <span className="text-xs font-semibold tracking-wide uppercase" style={{ color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>
            Models
          </span>
        </div>
        <span className="badge-mono px-2 py-0.5 rounded" style={{
          background: 'var(--bg-elevated)',
          color: 'var(--text-secondary)',
          border: '1px solid var(--border-subtle)',
        }}>
          {health.hardware_tier}
        </span>
      </div>

      {/* VRAM bar */}
      <div className="mb-3">
        <div className="flex items-center justify-between mb-1.5">
          <span className="flex items-center gap-1 text-[11px]" style={{ color: 'var(--text-muted)' }}>
            <HardDrive className="w-3 h-3" />
            VRAM
          </span>
          <span className="text-[11px] font-medium" style={{ color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>
            {usedGB.toFixed(1)} / {totalGB.toFixed(1)} GB
          </span>
        </div>
        <div className="w-full h-1.5 rounded-full overflow-hidden" style={{ background: 'var(--bg-elevated)' }}>
          <div
            className="h-full rounded-full transition-all duration-500"
            style={{
              width: `${Math.min(vramPercent, 100)}%`,
              background: vramPercent > 80 ? 'var(--danger)' : 'var(--accent)',
            }}
          />
        </div>
        {freeGB !== null && (
          <p className="text-[10px] mt-1" style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
            Free: {freeGB.toFixed(1)} GB
          </p>
        )}
      </div>

      {/* Resident models */}
      {hasModels ? (
        <div className="space-y-1.5">
          {modelsList.map(([name, info]) => (
            <div
              key={name}
              className="flex items-center justify-between px-2 py-1.5 rounded"
              style={{ background: 'var(--bg-elevated)' }}
            >
              <span
                className="text-[11px] truncate max-w-[140px]"
                title={name}
                style={{ color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}
              >
                {name.replace(/\.gguf$/, '').split('-').slice(0, 3).join('-')}
              </span>
              <span className="flex items-center gap-1">
                <span className="text-[10px] px-1.5 py-0.5 rounded" style={{
                  background: info.type === 'Llama' ? 'var(--accent-muted)' : 'var(--bg-hover)',
                  color: info.type === 'Llama' ? 'var(--accent)' : 'var(--text-muted)',
                  fontFamily: 'var(--font-mono)',
                }}>
                  {info.type}
                </span>
                <span className="text-[10px]" style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                  {info.vram_gb}GB
                </span>
              </span>
            </div>
          ))}
        </div>
      ) : (
        <div
          className="flex items-center gap-2 px-2 py-2 rounded text-[11px]"
          style={{ background: 'var(--bg-elevated)', color: 'var(--text-muted)' }}
        >
          <span className="w-1.5 h-1.5 rounded-full" style={{ background: 'var(--border-default)' }} />
          Idle — no models loaded
        </div>
      )}
    </div>
  )
}
