import { useState, useRef, useEffect } from 'react'
import { Send, Loader2, AlertCircle, User, Bot, Shield, Paperclip, CheckCircle, Cpu, Download, FileText, FileSpreadsheet, Presentation } from 'lucide-react'
import { uploadFile, getDownloadUrl, type ChatResponse } from '../hooks/useApi'

interface Props {
  onSend: (prompt: string) => void
  response: ChatResponse | null
  loading: boolean
  error: string | null
  role: string
}

interface Message {
  role: 'user' | 'assistant'
  content: string
  model_used?: string
  deliverables?: string[]
}

function extractDeliverables(content: string, explicit?: string[]): string[] {
  const set = new Set<string>(explicit || [])
  const patterns = [
    /[\w/.-]+\.(docx|xlsx|pptx)/gi,
    /outputs\/([\w.-]+\.(docx|xlsx|pptx))/gi,
  ]
  for (const p of patterns) {
    let match
    while ((match = p.exec(content)) !== null) {
      const fn = match[0].split('/').pop() || match[0]
      if (fn && !fn.startsWith('...')) {
        set.add(fn)
      }
    }
  }
  return Array.from(set)
}

function getFileIcon(filename: string) {
  const ext = filename.split('.').pop()?.toLowerCase()
  if (ext === 'xlsx') return <FileSpreadsheet className="w-3.5 h-3.5 text-accent" />
  if (ext === 'pptx') return <Presentation className="w-3.5 h-3.5 text-orange-400" />
  return <FileText className="w-3.5 h-3.5 text-sky-400" />
}

function SkeletonBubble() {
  return (
    <div className="flex gap-3 justify-start">
      <div className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0" style={{ background: 'var(--accent-muted)', border: '1px solid rgba(0,229,160,0.25)', boxShadow: '0 0 10px rgba(0,229,160,0.1)' }}>
        <Bot className="w-4 h-4" style={{ color: 'var(--accent)' }} />
      </div>
      <div className="msg-assistant px-4 py-3 max-w-[70%]">
        <div className="space-y-2">
          <div className="skeleton h-3 rounded" style={{ width: '85%' }} />
          <div className="skeleton h-3 rounded" style={{ width: '60%' }} />
          <div className="skeleton h-3 rounded" style={{ width: '75%' }} />
        </div>
      </div>
    </div>
  )
}

export default function ChatCanvas({ onSend, response, loading, error, role }: Props) {
  const [input, setInput] = useState('')
  const [messages, setMessages] = useState<Message[]>([])
  const [uploading, setUploading] = useState(false)
  const [uploadStatus, setUploadStatus] = useState<string | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Append assistant response when it arrives
  useEffect(() => {
    if (response?.response) {
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: response.response,
          model_used: response.model_used,
          deliverables: response.deliverables,
        },
      ])
    }
  }, [response])

  // Auto-scroll
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const trimmed = input.trim()
    if (!trimmed || loading) return
    setMessages((prev) => [...prev, { role: 'user', content: trimmed }])
    onSend(trimmed)
    setInput('')
  }

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    setUploading(true)
    setUploadStatus(`Uploading ${file.name}...`)
    try {
      const res = await uploadFile(file)
      setUploadStatus(`Uploaded ${file.name} to sandbox`)
      // Pre-populate input with reference to the uploaded file if empty
      if (!input.trim()) {
        setInput(`Analyze uploaded file: ${res.filename}`)
      }
      setTimeout(() => setUploadStatus(null), 4000)
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Upload failed'
      setUploadStatus(`Upload error: ${msg}`)
      setTimeout(() => setUploadStatus(null), 5000)
    } finally {
      setUploading(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  return (
    <div className="flex flex-col h-full" style={{ background: 'var(--bg-base)' }}>
      {/* Messages area */}
      <div className="flex-1 overflow-y-auto px-5 py-4 space-y-4">
        {messages.length === 0 && (
          <div className="flex items-center justify-center h-full">
            <div className="text-center max-w-sm">
              <div
                className="w-16 h-16 rounded-2xl mx-auto mb-4 flex items-center justify-center"
                style={{
                  background: 'rgba(0,229,160,0.08)',
                  border: '1px solid rgba(0,229,160,0.2)',
                  boxShadow: '0 0 24px rgba(0,229,160,0.15)',
                }}
              >
                <Shield className="w-8 h-8" style={{ color: 'var(--accent)' }} />
              </div>
              <p className="text-base font-semibold mb-1 tracking-tight" style={{ color: 'var(--text-primary)' }}>
                Sovereign AI Workbench
              </p>
              <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
                Air-gapped local inference · Role: <span style={{ color: 'var(--accent)', fontFamily: 'var(--font-mono)' }}>{role}</span>
              </p>
              <p className="text-[12px] leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
                Ask questions, generate deliverables, or upload local sandbox files for triage and extraction.
              </p>
            </div>
          </div>
        )}

        {messages.map((msg, i) => {
          const files = msg.role === 'assistant' ? extractDeliverables(msg.content, msg.deliverables) : []
          return (
            <div
              key={i}
              className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              {msg.role === 'assistant' && (
                <div
                  className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5"
                  style={{
                    background: 'rgba(0,229,160,0.12)',
                    border: '1px solid rgba(0,229,160,0.25)',
                    boxShadow: '0 0 10px rgba(0,229,160,0.1)',
                  }}
                >
                  <Bot className="w-4 h-4" style={{ color: 'var(--accent)' }} />
                </div>
              )}
              <div className="flex flex-col gap-1.5 max-w-[78%]">
                <div
                  className={`px-4 py-3 text-[13px] leading-relaxed ${
                    msg.role === 'user' ? 'msg-user' : 'msg-assistant'
                  }`}
                >
                  <pre className="whitespace-pre-wrap" style={{ fontFamily: 'var(--font-body)' }}>{msg.content}</pre>

                  {/* Render inline download buttons for deliverables */}
                  {files.length > 0 && (
                    <div className="flex flex-wrap gap-2 mt-3 pt-2.5" style={{ borderTop: '1px solid rgba(255, 255, 255, 0.08)' }}>
                      {files.map((fn) => (
                        <a
                          key={fn}
                          href={getDownloadUrl(fn)}
                          target="_blank"
                          rel="noopener noreferrer"
                          download={fn}
                          className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all group"
                          style={{
                            background: 'rgba(0, 229, 160, 0.12)',
                            border: '1px solid rgba(0, 229, 160, 0.3)',
                            color: 'var(--accent)',
                            fontFamily: 'var(--font-mono)',
                            boxShadow: '0 0 12px rgba(0, 229, 160, 0.15)',
                          }}
                        >
                          {getFileIcon(fn)}
                          <span>Download {fn}</span>
                          <Download className="w-3 h-3 ml-0.5 opacity-70 group-hover:opacity-100 transition-opacity" />
                        </a>
                      ))}
                    </div>
                  )}
                </div>
                {msg.role === 'assistant' && msg.model_used && (
                  <div
                    className="flex items-center gap-1.5 px-2 text-[11px]"
                    style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}
                  >
                    <Cpu className="w-3 h-3" style={{ color: 'var(--accent)' }} />
                    <span>Generated by <span style={{ color: 'var(--text-secondary)' }}>{msg.model_used}</span></span>
                  </div>
                )}
              </div>
              {msg.role === 'user' && (
                <div
                  className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5"
                  style={{
                    background: 'rgba(56,189,248,0.12)',
                    border: '1px solid rgba(56,189,248,0.25)',
                  }}
                >
                  <User className="w-4 h-4" style={{ color: '#38bdf8' }} />
                </div>
              )}
            </div>
          )
        })}

        {loading && <SkeletonBubble />}

        {error && (
          <div className="flex gap-3 justify-start">
            <div
              className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0"
              style={{ background: 'rgba(239,68,68,0.12)', border: '1px solid rgba(239,68,68,0.25)' }}
            >
              <AlertCircle className="w-4 h-4" style={{ color: '#f87171' }} />
            </div>
            <div
              className="px-4 py-3 rounded-xl flex items-center gap-2"
              style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)' }}
            >
              <span className="text-[13px]" style={{ color: '#f87171' }}>{error}</span>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Upload toast */}
      {uploadStatus && (
        <div
          className="mx-5 mb-2 px-3 py-1.5 rounded-lg flex items-center gap-2 text-xs"
          style={{
            background: 'rgba(0, 229, 160, 0.08)',
            border: '1px solid rgba(0, 229, 160, 0.2)',
            color: 'var(--accent)',
            fontFamily: 'var(--font-mono)',
          }}
        >
          {uploading ? (
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
          ) : (
            <CheckCircle className="w-3.5 h-3.5" />
          )}
          <span>{uploadStatus}</span>
        </div>
      )}

      {/* Input area */}
      <form
        onSubmit={handleSubmit}
        className="px-5 py-3"
        style={{
          borderTop: '1px solid var(--border-subtle)',
          background: 'var(--bg-surface)',
          backdropFilter: 'blur(16px)',
        }}
      >
        <div className="flex items-center gap-2">
          {/* File upload attachment button */}
          <input
            ref={fileInputRef}
            type="file"
            onChange={handleFileUpload}
            className="hidden"
            id="chat-file-upload"
          />
          <label
            htmlFor="chat-file-upload"
            className="p-2.5 rounded-lg cursor-pointer transition-all flex items-center justify-center"
            style={{
              background: 'rgba(255, 255, 255, 0.05)',
              border: '1px solid var(--border-subtle)',
              color: uploading ? 'var(--accent)' : 'var(--text-secondary)',
            }}
            title="Attach file to sandbox"
          >
            {uploading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Paperclip className="w-4 h-4 hover:text-accent transition-colors" />
            )}
          </label>

          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={`Ask the workbench (${role})...`}
            disabled={loading}
            className="flex-1 rounded-lg px-4 py-2.5 text-[13px] placeholder-sm focus:outline-none focus:ring-1 disabled:opacity-40"
            style={{
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid var(--border-default)',
              color: 'var(--text-primary)',
              fontFamily: 'var(--font-body)',
              backdropFilter: 'blur(8px)',
            }}
            onFocus={(e) => {
              e.currentTarget.style.borderColor = 'var(--accent)'
              e.currentTarget.style.boxShadow = '0 0 0 1px var(--accent-glow)'
            }}
            onBlur={(e) => {
              e.currentTarget.style.borderColor = 'var(--border-default)'
              e.currentTarget.style.boxShadow = 'none'
            }}
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="btn-primary flex items-center gap-2 px-4 py-2.5"
          >
            {loading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Send className="w-4 h-4" />
            )}
          </button>
        </div>
      </form>
    </div>
  )
}
