import { useState, useEffect, useRef } from 'react'
import { Sparkles, Cpu, ChevronDown, Check, Zap, BrainCircuit, Code2, Eye, MessageSquare, ShieldCheck } from 'lucide-react'
import { fetchModels, type ModelInfo } from '../hooks/useApi'

interface Props {
  selectedModel: string
  onSelectModel: (modelId: string) => void
}

const FALLBACK_MODELS: ModelInfo[] = [
  {
    id: 'auto',
    name: 'Auto (Intelligent Routing)',
    category: 'AUTO',
    param_size: 'Auto',
    vram_gb: 0,
    size_gb: 0,
    description: 'Automatically selects the best local 7B/14B model based on task intent (Coding, Math, Vision, or Chat).',
    is_present: true,
  },
  {
    id: 'deepseek-r1-7b.gguf',
    name: 'DeepSeek R1 7B',
    category: 'REASONING',
    param_size: '7B',
    vram_gb: 4.5,
    size_gb: 4.36,
    description: 'Distilled reasoning model for math, complex logic, and step-by-step verification.',
    is_present: true,
  },
  {
    id: 'phi4-14b.gguf',
    name: 'Phi-4 14B',
    category: 'REASONING',
    param_size: '14B',
    vram_gb: 9.0,
    size_gb: 8.43,
    description: 'High-capacity 14B reasoning powerhouse for deep synthesis, complex problem solving, and architecture.',
    is_present: true,
  },
  {
    id: 'qwen2.5-coder-7b-instruct-q3_k_m.gguf',
    name: 'Qwen 2.5 Coder 7B',
    category: 'CODE',
    param_size: '7B',
    vram_gb: 4.0,
    size_gb: 3.55,
    description: 'Specialized coding and deliverable synthesis model for Python scripts, debugging, and documents.',
    is_present: true,
  },
  {
    id: 'llava-7b.gguf',
    name: 'LLaVA 7B (Vision)',
    category: 'VISION',
    param_size: '7B',
    vram_gb: 4.5,
    size_gb: 3.83,
    description: 'Multimodal vision-language model for image reasoning, diagram analysis, and document OCR.',
    is_present: true,
  },
  {
    id: 'qwen2.5-7b-instruct-q3_k_m.gguf',
    name: 'Qwen 2.5 7B Instruct',
    category: 'GENERAL',
    param_size: '7B',
    vram_gb: 4.0,
    size_gb: 3.55,
    description: 'High-accuracy general instruction-tuned model for structured analysis, reporting, and Q&A.',
    is_present: true,
  },
  {
    id: 'qwen2.5-7b.gguf',
    name: 'Qwen 2.5 7B',
    category: 'GENERAL',
    param_size: '7B',
    vram_gb: 4.5,
    size_gb: 4.36,
    description: 'General conversational 7B foundational model for multi-domain queries.',
    is_present: true,
  },
  {
    id: 'qwen1_5-4b-chat-q4_k_m.gguf',
    name: 'Qwen 1.5 4B Chat',
    category: 'GENERAL',
    param_size: '4B',
    vram_gb: 2.8,
    size_gb: 2.29,
    description: 'Fast, low-latency conversational model optimized for 4GB VRAM hardware.',
    is_present: true,
  },
  {
    id: 'qwen2.5-coder-3b-instruct-q4_k_m.gguf',
    name: 'Qwen 2.5 Coder 3B',
    category: 'CODE',
    param_size: '3B',
    vram_gb: 2.0,
    size_gb: 1.30,
    description: 'Lightweight code generator for quick script synthesis.',
    is_present: true,
  },
  {
    id: 'qwen2.5-0.5b-instruct-q4_k_m.gguf',
    name: 'Qwen 2.5 0.5B',
    category: 'FALLBACK',
    param_size: '0.5B',
    vram_gb: 0.8,
    size_gb: 0.46,
    description: 'Ultra-low memory emergency fallback model.',
    is_present: true,
  },
]

export default function ModelSelector({ selectedModel, onSelectModel }: Props) {
  const [isOpen, setIsOpen] = useState(false)
  const [models, setModels] = useState<ModelInfo[]>(FALLBACK_MODELS)
  const dropdownRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    fetchModels()
      .then((res) => {
        if (res.models && res.models.length > 0) {
          setModels(res.models)
        }
      })
      .catch(() => {
        // use fallback list
      })
  }, [])

  // Close dropdown on outside click
  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setIsOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const currentModel = models.find((m) => m.id === selectedModel) || models[0]

  const getCategoryBadgeStyle = (category: string) => {
    switch (category) {
      case 'AUTO':
        return {
          bg: 'rgba(0, 229, 160, 0.15)',
          color: 'var(--accent)',
          border: '1px solid rgba(0, 229, 160, 0.3)',
        }
      case 'REASONING':
        return {
          bg: 'rgba(168, 85, 247, 0.15)',
          color: '#c084fc',
          border: '1px solid rgba(168, 85, 247, 0.3)',
        }
      case 'CODE':
        return {
          bg: 'rgba(56, 189, 248, 0.15)',
          color: '#38bdf8',
          border: '1px solid rgba(56, 189, 248, 0.3)',
        }
      case 'VISION':
        return {
          bg: 'rgba(251, 146, 60, 0.15)',
          color: '#fb923c',
          border: '1px solid rgba(251, 146, 60, 0.3)',
        }
      case 'GENERAL':
        return {
          bg: 'rgba(96, 165, 250, 0.15)',
          color: '#60a5fa',
          border: '1px solid rgba(96, 165, 250, 0.3)',
        }
      default:
        return {
          bg: 'rgba(148, 163, 184, 0.15)',
          color: '#94a3b8',
          border: '1px solid rgba(148, 163, 184, 0.3)',
        }
    }
  }

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'AUTO':
        return <Zap className="w-3.5 h-3.5" style={{ color: 'var(--accent)' }} />
      case 'REASONING':
        return <BrainCircuit className="w-3.5 h-3.5" style={{ color: '#c084fc' }} />
      case 'CODE':
        return <Code2 className="w-3.5 h-3.5" style={{ color: '#38bdf8' }} />
      case 'VISION':
        return <Eye className="w-3.5 h-3.5" style={{ color: '#fb923c' }} />
      case 'GENERAL':
        return <MessageSquare className="w-3.5 h-3.5" style={{ color: '#60a5fa' }} />
      default:
        return <Cpu className="w-3.5 h-3.5" style={{ color: 'var(--text-muted)' }} />
    }
  }

  return (
    <div className="relative" ref={dropdownRef}>
      {/* Dropdown Trigger Button */}
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        aria-haspopup="listbox"
        aria-expanded={isOpen}
        className="flex items-center gap-2.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all group"
        style={{
          background: 'var(--bg-elevated)',
          border: isOpen ? '1px solid var(--accent)' : '1px solid var(--border-subtle)',
          boxShadow: isOpen ? '0 0 12px var(--accent-glow)' : 'none',
        }}
      >
        <span className="flex items-center gap-1.5">
          {getCategoryIcon(currentModel.category)}
          <span className="font-bold" style={{ color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
            {currentModel.name}
          </span>
        </span>

        {currentModel.id === 'auto' ? (
          <span
            className="text-[10px] px-1.5 py-0.5 rounded font-bold tracking-wider"
            style={getCategoryBadgeStyle('AUTO')}
          >
            AUTO
          </span>
        ) : (
          <span
            className="text-[10px] px-1.5 py-0.5 rounded font-mono"
            style={getCategoryBadgeStyle(currentModel.category)}
          >
            {currentModel.param_size || currentModel.category}
          </span>
        )}

        <ChevronDown
          className={`w-3.5 h-3.5 transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`}
          style={{ color: 'var(--text-muted)' }}
        />
      </button>

      {/* Dropdown Menu Modal / Popover */}
      {isOpen && (
        <div
          className="absolute right-0 mt-2 w-80 rounded-xl overflow-hidden z-50 animate-in fade-in zoom-in-95 duration-150"
          style={{
            background: 'rgba(15, 20, 30, 0.96)',
            backdropFilter: 'blur(20px)',
            border: '1px solid var(--border-default)',
            boxShadow: '0 12px 40px rgba(0, 0, 0, 0.6), 0 0 20px rgba(0, 229, 160, 0.1)',
          }}
          role="listbox"
        >
          {/* Menu Header */}
          <div
            className="px-3.5 py-2.5 border-b flex items-center justify-between"
            style={{
              borderColor: 'var(--border-subtle)',
              background: 'rgba(255, 255, 255, 0.02)',
            }}
          >
            <div className="flex items-center gap-2">
              <Cpu className="w-3.5 h-3.5" style={{ color: 'var(--accent)' }} />
              <span className="text-[11px] font-bold uppercase tracking-wider" style={{ color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>
                Local Model Engine
              </span>
            </div>
            <span className="text-[10px] px-2 py-0.5 rounded flex items-center gap-1" style={{ background: 'rgba(0, 229, 160, 0.1)', color: 'var(--accent)', border: '1px solid rgba(0, 229, 160, 0.2)' }}>
              <ShieldCheck className="w-3 h-3" />
              Air-Gapped
            </span>
          </div>

          {/* Model Options List */}
          <div className="max-h-[360px] overflow-y-auto py-1 divide-y divide-white/5">
            {models.map((m) => {
              const isSelected = m.id === selectedModel
              const isAuto = m.id === 'auto'
              const badgeStyle = getCategoryBadgeStyle(m.category)

              return (
                <button
                  key={m.id}
                  type="button"
                  role="option"
                  aria-selected={isSelected}
                  onClick={() => {
                    onSelectModel(m.id)
                    setIsOpen(false)
                  }}
                  className="w-full text-left px-3.5 py-2.5 transition-colors flex items-start justify-between gap-3 group"
                  style={{
                    background: isSelected ? 'rgba(0, 229, 160, 0.08)' : 'transparent',
                  }}
                  onMouseEnter={(e) => {
                    if (!isSelected) e.currentTarget.style.background = 'rgba(255, 255, 255, 0.04)'
                  }}
                  onMouseLeave={(e) => {
                    if (!isSelected) e.currentTarget.style.background = 'transparent'
                  }}
                >
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="flex-shrink-0">{getCategoryIcon(m.category)}</span>
                      <span
                        className="text-xs font-semibold truncate"
                        style={{
                          color: isSelected ? 'var(--accent)' : 'var(--text-primary)',
                          fontFamily: 'var(--font-mono)',
                        }}
                      >
                        {m.name}
                      </span>
                      <span
                        className="text-[9px] px-1.5 py-0.2 rounded font-mono font-bold tracking-tight"
                        style={badgeStyle}
                      >
                        {m.category}
                      </span>
                    </div>

                    <p className="text-[11px] leading-tight text-left line-clamp-2" style={{ color: 'var(--text-muted)' }}>
                      {m.description}
                    </p>

                    {!isAuto && (
                      <div className="flex items-center gap-2 mt-1 text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>
                        <span>Size: <strong style={{ color: 'var(--text-secondary)' }}>{m.size_gb}GB</strong></span>
                        <span>•</span>
                        <span>VRAM: <strong style={{ color: 'var(--text-secondary)' }}>~{m.vram_gb}GB</strong></span>
                      </div>
                    )}
                  </div>

                  {isSelected && (
                    <div className="mt-1 flex-shrink-0 w-4 h-4 rounded-full flex items-center justify-center" style={{ background: 'var(--accent)', color: '#0b0e14' }}>
                      <Check className="w-3 h-3 stroke-[3]" />
                    </div>
                  )}
                </button>
              )
            })}
          </div>

          {/* Footer note */}
          <div
            className="px-3.5 py-2 border-t text-[10px] flex items-center justify-between"
            style={{
              borderColor: 'var(--border-subtle)',
              background: 'rgba(0, 0, 0, 0.2)',
              color: 'var(--text-muted)',
              fontFamily: 'var(--font-mono)',
            }}
          >
            <span>Auto switches based on prompt</span>
            <span style={{ color: 'var(--accent)' }}>Ollama 7B / 14B Ready</span>
          </div>
        </div>
      )}
    </div>
  )
}
