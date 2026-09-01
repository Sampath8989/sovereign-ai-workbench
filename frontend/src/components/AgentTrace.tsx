import { ChevronDown, ChevronRight, Brain, Play, FileSearch, Search, CheckCircle, Sparkles } from 'lucide-react'
import { useState } from 'react'

interface Props {
  trace?: string[]
}

/** Map trace step text to type and display config */
function parseStep(text: string): { type: string; icon: React.ReactNode; color: string; nodeClass: string; label: string } {
  const lower = text.toLowerCase()
  if (lower.startsWith('planner')) {
    return {
      type: 'planner',
      icon: <Play className="w-3 h-3" />,
      color: '#a78bfa',
      nodeClass: 'step-node step-node-planner',
      label: 'Planner',
    }
  }
  if (lower.startsWith('executor')) {
    return {
      type: 'executor',
      icon: <FileSearch className="w-3 h-3" />,
      color: '#38bdf8',
      nodeClass: 'step-node step-node-executor',
      label: 'Executor',
    }
  }
  if (lower.startsWith('retriever')) {
    return {
      type: 'retriever',
      icon: <Search className="w-3 h-3" />,
      color: '#fbbf24',
      nodeClass: 'step-node step-node-retriever',
      label: 'Retriever',
    }
  }
  if (lower.startsWith('verifier')) {
    return {
      type: 'verifier',
      icon: <CheckCircle className="w-3 h-3" />,
      color: 'var(--accent)',
      nodeClass: 'step-node step-node-verifier',
      label: 'Verifier',
    }
  }
  if (lower.startsWith('synthesizer')) {
    return {
      type: 'synthesizer',
      icon: <Sparkles className="w-3 h-3" />,
      color: '#f472b6',
      nodeClass: 'step-node step-node-synthesizer',
      label: 'Synthesizer',
    }
  }
  // Generic 'Step N:' fallback
  if (lower.startsWith('step')) {
    return {
      type: 'generic',
      icon: <span className="text-[10px] font-bold">•</span>,
      color: 'var(--text-muted)',
      nodeClass: 'step-node',
      label: '',
    }
  }
  return {
    type: 'unknown',
    icon: <span className="text-[10px] font-bold">•</span>,
    color: 'var(--text-muted)',
    nodeClass: 'step-node',
    label: '',
  }
}

export default function AgentTrace({ trace }: Props) {
  const [expanded, setExpanded] = useState(false)
  const steps = trace && trace.length > 0 ? trace : null

  return (
    <div className="card">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-2 w-full text-left"
      >
        <Brain className="w-4 h-4" style={{ color: '#a78bfa' }} />
        <span className="text-xs font-semibold tracking-wide uppercase" style={{ color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>
          Agent Trace
        </span>
        {steps && (
          <span
            className="ml-1 text-[10px] px-1.5 py-0.5 rounded"
            style={{ background: 'var(--bg-elevated)', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}
          >
            {steps.length}
          </span>
        )}
        <span className="ml-auto" style={{ color: 'var(--text-muted)' }}>
          {expanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
        </span>
      </button>

      {expanded && (
        <div className="mt-3">
          {steps ? (
            <div className="relative">
              {steps.map((step, i) => {
                const parsed = parseStep(step)
                const isLast = i === steps.length - 1
                return (
                  <div key={i} className="flex items-start gap-2.5 relative">
                    {/* Connector line */}
                    {!isLast && (
                      <div
                        className="absolute left-[13px] top-7 w-px h-full"
                        style={{ background: 'var(--border-subtle)' }}
                      />
                    )}
                    {/* Node */}
                    <div className={parsed.nodeClass} style={{ position: 'relative', zIndex: 1 }}>
                      {parsed.icon}
                    </div>
                    {/* Content */}
                    <div className="pb-3 min-w-0">
                      {parsed.label && (
                        <span
                          className="text-[10px] font-semibold uppercase tracking-wider"
                          style={{ color: parsed.color, fontFamily: 'var(--font-mono)' }}
                        >
                          {parsed.label}
                        </span>
                      )}
                      <p className="text-[11px] leading-relaxed mt-0.5" style={{ color: 'var(--text-muted)' }}>
                        {parsed.label ? step.replace(/^[^:]+:\s*/, '') : step}
                      </p>
                    </div>
                  </div>
                )
              })}
            </div>
          ) : (
            <p className="text-[11px] italic" style={{ color: 'var(--text-muted)' }}>
              Send a message to see agent reasoning.
            </p>
          )}
        </div>
      )}
    </div>
  )
}
