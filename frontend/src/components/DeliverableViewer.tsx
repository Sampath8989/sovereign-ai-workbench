import { Download, FileText, FileSpreadsheet, Presentation } from 'lucide-react'
import { getDownloadUrl } from '../hooks/useApi'

interface Props {
  response?: string
  deliverables?: string[]
}

interface Deliverable {
  filename: string
  type: 'docx' | 'xlsx' | 'pptx' | 'other'
}

function extractDeliverables(text?: string, explicit?: string[]): Deliverable[] {
  const seen = new Set<string>()
  const files: Deliverable[] = []

  // Add explicit deliverables
  if (explicit) {
    for (const fn of explicit) {
      if (fn && !seen.has(fn)) {
        seen.add(fn)
        const ext = fn.split('.').pop()?.toLowerCase() || ''
        files.push({ filename: fn, type: (['docx', 'xlsx', 'pptx'].includes(ext) ? ext : 'other') as Deliverable['type'] })
      }
    }
  }

  // Add regex matches from text
  if (text) {
    const patterns = [
      /[\w/.-]+\.(docx|xlsx|pptx)/gi,
      /outputs\/([\w.-]+\.(docx|xlsx|pptx))/gi,
    ]
    for (const pattern of patterns) {
      let match
      while ((match = pattern.exec(text)) !== null) {
        const filename = match[0].split('/').pop() || match[0]
        if (filename.startsWith('...') || seen.has(filename)) continue
        seen.add(filename)

        const ext = filename.split('.').pop()?.toLowerCase() || ''
        files.push({
          filename,
          type: (['docx', 'xlsx', 'pptx'].includes(ext) ? ext : 'other') as Deliverable['type'],
        })
      }
    }
  }
  return files
}

const iconMap = {
  docx: <FileText className="w-4 h-4" style={{ color: '#38bdf8' }} />,
  xlsx: <FileSpreadsheet className="w-4 h-4" style={{ color: 'var(--accent)' }} />,
  pptx: <Presentation className="w-4 h-4" style={{ color: '#fb923c' }} />,
  other: <FileText className="w-4 h-4" style={{ color: 'var(--text-muted)' }} />,
}

const labelMap: Record<string, string> = {
  docx: 'Word Document',
  xlsx: 'Spreadsheet',
  pptx: 'Presentation',
  other: 'File',
}

export default function DeliverableViewer({ response, deliverables }: Props) {
  const items = extractDeliverables(response, deliverables)

  if (items.length === 0) return null

  return (
    <div style={{ borderTop: '1px solid var(--border-subtle)', background: 'var(--bg-surface)', padding: '0.75rem 1.25rem' }}>
      <div className="flex items-center gap-2 mb-2">
        <Download className="w-3.5 h-3.5" style={{ color: 'var(--accent)' }} />
        <span className="text-[11px] font-semibold tracking-wide uppercase" style={{ color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>
          Deliverables
        </span>
      </div>
      <div className="flex flex-wrap gap-2">
        {items.map((d) => (
          <a
            key={d.filename}
            href={getDownloadUrl(d.filename)}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 px-3 py-2 rounded-lg transition-all group"
            style={{
              background: 'var(--bg-elevated)',
              border: '1px solid var(--border-subtle)',
            }}
            onMouseEnter={(e) => { e.currentTarget.style.borderColor = 'rgba(0,229,160,0.3)'; e.currentTarget.style.background = 'var(--bg-hover)' }}
            onMouseLeave={(e) => { e.currentTarget.style.borderColor = 'var(--border-subtle)'; e.currentTarget.style.background = 'var(--bg-elevated)' }}
          >
            {iconMap[d.type]}
            <div>
              <p className="text-[11px] font-medium group-hover:text-[var(--accent)]" style={{ color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
                {d.filename}
              </p>
              <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                {labelMap[d.type]}
              </p>
            </div>
            <Download className="w-3 h-3 ml-1 opacity-0 group-hover:opacity-100 transition-opacity" style={{ color: 'var(--text-muted)' }} />
          </a>
        ))}
      </div>
    </div>
  )
}
