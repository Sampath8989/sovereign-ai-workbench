import { User, Crown } from 'lucide-react'

interface Props {
  role: 'engineer' | 'manager'
  onChange: (role: 'engineer' | 'manager') => void
}

export default function RoleSwitcher({ role, onChange }: Props) {
  return (
    <div className="flex items-center gap-1 p-0.5 rounded-lg" style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-subtle)' }}>
      <button
        onClick={() => onChange('engineer')}
        className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-[11px] font-medium transition-all"
        style={{
          background: role === 'engineer' ? 'var(--accent)' : 'transparent',
          color: role === 'engineer' ? '#0b0e14' : 'var(--text-muted)',
          fontFamily: 'var(--font-mono)',
        }}
      >
        <User className="w-3 h-3" />
        Engineer
      </button>
      <button
        onClick={() => onChange('manager')}
        className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-[11px] font-medium transition-all"
        style={{
          background: role === 'manager' ? 'var(--accent)' : 'transparent',
          color: role === 'manager' ? '#0b0e14' : 'var(--text-muted)',
          fontFamily: 'var(--font-mono)',
        }}
      >
        <Crown className="w-3 h-3" />
        Manager
      </button>
    </div>
  )
}
