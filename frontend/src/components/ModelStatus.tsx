import { useState, useEffect, useCallback } from 'react'
import { Cpu, HardDrive, Server, Sparkles, Layers } from 'lucide-react'
import { fetchHealth, type HealthResponse } from '../hooks/useApi'

interface Props {
  selectedModel?: string
}

export default function ModelStatus({ selectedModel = 'auto' }: Props) {
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
  const models = mm.resident_models || {}
  const liveUsedGB = mm.live_used_vram_gb !== null && mm.live_used_vram_gb !== undefined ? mm.live_used_vram_gb : mm.total_vram_used_gb
  const liveTotalGB = mm.live_total_vram_gb !== null && mm.live_total_vram_gb !== undefined ? mm.live_total_vram_gb : mm.effective_budget_gb
  const effectiveBudgetGB = mm.effective_budget_gb
  const freeGB = mm.live_free_vram_gb
  const modelsList = Object.entries(models)
  const hasModels = modelsList.length > 0
  const vramPercent = liveTotalGB > 0 ? (liveUsedGB / liveTotalGB) * 100 : 0
  const availableCount = health.available_models ? health.available_models.length : 10

  // Check VRAM requirements against tier budget
  const roster = health.model_roster || {}
  const selectedReqVram = selectedModel !== 'auto' ? (roster[selectedModel] || 0) : 0
  const exceedsVram = selectedModel !== 'auto' && selectedReqVram > effectiveBudgetGB
  const isResident = selectedModel !== 'auto' && Boolean(models[selectedModel])
  const isPinned = selectedModel !== 'auto' && (mm.pinned_model === selectedModel || models[selectedModel]?.pinned)

  const getStatusBadge = () => {
    if (selectedModel === 'auto') {
      return { label: 'Dynamic', color: 'var(--text-secondary)' }
    }
    if (exceedsVram) {
      return { label: `⚠️ Exceeds VRAM (${selectedReqVram}GB > ${effectiveBudgetGB.toFixed(1)}GB)`, color: '#f87171' }
    }
    if (isPinned) {
      return { label: 'Pinned (Resident)', color: '#4ade80' }
    }
    if (isResident) {
      return { label: 'Loaded (LRU)', color: '#38bdf8' }
    }
    return { label: 'Routing Preference (On-Demand)', color: 'var(--text-muted)' }
  }

  const getSelectedLabel = () => {
    if (selectedModel === 'auto') return '⚡ Auto (Intelligent Routing)'
    const clean = selectedModel.replace(/\.gguf$/, '')
    if (clean.includes('deepseek')) return 'DeepSeek R1 7B'
    if (clean.includes('phi4')) return 'Phi-4 14B'
    if (clean.includes('coder-7b')) return 'Qwen 2.5 Coder 7B'
    if (clean.includes('llava')) return 'LLaVA 7B (Vision)'
    if (clean.includes('7b-instruct')) return 'Qwen 2.5 7B Instruct'
    if (clean.includes('7b')) return 'Qwen 2.5 7B'
    if (clean.includes('4b')) return 'Qwen 1.5 4B'
    if (clean.includes('coder-3b')) return 'Qwen 2.5 Coder 3B'
    if (clean.includes('0.5b')) return 'Qwen 2.5 0.5B'
    return clean
  }

  const badge = getStatusBadge()

  return (
    <div className="card">
      {/* Header */}
      <div className="flex items-center justify-between mb-2.5">
        <div className="flex items-center gap-2">
          <Server className="w-4 h-4" style={{ color: hasModels ? 'var(--accent)' : 'var(--text-muted)' }} />
          <span className="text-xs font-semibold tracking-wide uppercase" style={{ color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>
            Models
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="text-[10px] px-1.5 py-0.5 rounded font-mono" style={{
            background: 'rgba(0, 229, 160, 0.1)',
            color: 'var(--accent)',
            border: '1px solid rgba(0, 229, 160, 0.25)',
          }}>
            {availableCount} Models
          </span>
          <span className="badge-mono px-2 py-0.5 rounded" style={{
            background: 'var(--bg-elevated)',
            color: 'var(--text-secondary)',
            border: '1px solid var(--border-subtle)',
          }}>
            {health.hardware_tier}
          </span>
        </div>
      </div>

      {/* Selected Active Engine Mode */}
      <div
        className="mb-3 px-2.5 py-1.5 rounded-lg flex flex-col gap-1 text-[11px]"
        style={{
          background: exceedsVram ? 'rgba(239, 68, 68, 0.08)' : (selectedModel === 'auto' ? 'rgba(0, 229, 160, 0.08)' : 'rgba(168, 85, 247, 0.08)'),
          border: exceedsVram ? '1px solid rgba(239, 68, 68, 0.25)' : (selectedModel === 'auto' ? '1px solid rgba(0, 229, 160, 0.2)' : '1px solid rgba(168, 85, 247, 0.2)'),
          fontFamily: 'var(--font-mono)',
        }}
      >
        <div className="flex items-center justify-between">
          <span className="flex items-center gap-1.5" style={{ color: exceedsVram ? '#f87171' : (selectedModel === 'auto' ? 'var(--accent)' : '#c084fc') }}>
            {selectedModel === 'auto' ? <Sparkles className="w-3.5 h-3.5" /> : <Cpu className="w-3.5 h-3.5" />}
            <span className="font-semibold">{getSelectedLabel()}</span>
          </span>
          <span className="text-[10px]" style={{ color: badge.color }}>
            {badge.label}
          </span>
        </div>
        {exceedsVram && (
          <p className="text-[10px] text-red-400 font-mono mt-0.5">
            ⚠️ Model requires {selectedReqVram} GB VRAM, exceeding the {effectiveBudgetGB.toFixed(1)} GB tier budget.
          </p>
        )}
      </div>

      {/* VRAM bar */}
      <div className="mb-3">
        <div className="flex items-center justify-between mb-1.5">
          <span className="flex items-center gap-1 text-[11px]" style={{ color: 'var(--text-muted)' }}>
            <HardDrive className="w-3 h-3" />
            Live VRAM
          </span>
          <span className="text-[11px] font-medium" style={{ color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>
            {liveUsedGB.toFixed(1)} / {liveTotalGB.toFixed(1)} GB
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
        <div className="flex items-center justify-between text-[10px] mt-1" style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
          {freeGB !== null && (
            <span>Free: {freeGB.toFixed(1)} GB</span>
          )}
          <span>Budget: {effectiveBudgetGB.toFixed(1)} GB</span>
        </div>
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
                {info.pinned && (
                  <span className="text-[9px] px-1 py-0.5 rounded font-mono bg-green-950/40 text-green-400 border border-green-500/30">
                    PINNED
                  </span>
                )}
                <span className="text-[10px] px-1.5 py-0.5 rounded" style={{
                  background: info.type === 'Llama' ? 'var(--accent-muted)' : 'var(--bg-hover)',
                  color: info.type === 'Llama' ? 'var(--accent)' : 'var(--text-muted)',
                  fontFamily: 'var(--font-mono)',
                }}>
                  {info.type}
                </span>
                <span className="text-[10px]" style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                  {info.vram_gb.toFixed(1)}GB
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
          No models currently resident in VRAM
        </div>
      )}
    </div>
  )
}
