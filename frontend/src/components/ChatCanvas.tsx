import { useState, useRef, useEffect } from 'react'
import { Send, Loader2, AlertCircle, User, Bot, Shield, Paperclip, CheckCircle, Cpu, Download, FileText, FileSpreadsheet, Presentation, Upload, X } from 'lucide-react'
import { uploadFile, getDownloadUrl, type ChatResponse, type SessionMessage } from '../hooks/useApi'

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  model_used?: string
  deliverables?: string[]
}

interface Props {
  onSend: (prompt: string) => void
  response: ChatResponse | null
  loading: boolean
  error: string | null
  role: string
  sessionId?: string | null
  initialMessages?: ChatMessage[]
}

export interface ParsedResponse {
  reasoning: string | null
  answer: string
}

/**
 * Separate reasoning traces (DeepSeek R1-style <think> blocks) from final answer
 * so chain-of-thought can be folded/collapsed in the UI layer.
 */
export function parseReasoningAndAnswer(content: string): ParsedResponse {
  if (!content) return { reasoning: null, answer: '' }

  let reasoning: string | null = null
  let text = content

  // Match complete <think>...</think> or <thinking>...</thinking>
  const thinkMatch = text.match(/<(?:think|thinking)>([\s\S]*?)<\/(?:think|thinking)>/i)
  if (thinkMatch) {
    reasoning = thinkMatch[1].trim()
    text = text.replace(/<(?:think|thinking)>[\s\S]*?<\/(?:think|thinking)>/i, '')
  } else {
    // Handle unclosed <think>... (e.g. streaming or truncated)
    const openMatch = text.match(/<(?:think|thinking)>([\s\S]*)$/i)
    if (openMatch) {
      reasoning = openMatch[1].trim()
      text = text.replace(/<(?:think|thinking)>[\s\S]*$/i, '')
    }
  }

  // Strip internal scratchpad keys and orchestrator artifacts
  let cleaned = text
  cleaned = cleaned.replace(/\bstep_\d+_(?:result|tool|action)\b/gi, '')
  cleaned = cleaned.replace(/\bStep\s+step_\d+_(?:result|tool|action)\s*:?/gi, '')
  cleaned = cleaned.replace(/\b(?:User request|Execution results|Retrieved sources)\s*:/gi, '')
  cleaned = cleaned.replace(/<\/?(?:think|thinking)>/gi, '')
  cleaned = cleaned.replace(/[ \t]{2,}/g, ' ')
  cleaned = cleaned.replace(/\n{3,}/g, '\n\n')

  return {
    reasoning: reasoning && reasoning.length > 0 ? reasoning : null,
    answer: cleaned.trim(),
  }
}

export function cleanResponseText(content: string): string {
  return parseReasoningAndAnswer(content).answer
}

function extractDeliverables(content: string, explicit?: string[]): string[] {
  const set = new Set<string>()
  const seenLower = new Set<string>()
  const add = (fn: string) => {
    const lower = fn.toLowerCase()
    if (!seenLower.has(lower)) {
      seenLower.add(lower)
      set.add(fn)
    }
  }
  ;(explicit || []).forEach(add)
  const patterns = [
    /[\w/.-]+\.(docx|xlsx|pptx)/gi,
    /outputs\/([\w.-]+\.(docx|xlsx|pptx))/gi,
  ]
  for (const p of patterns) {
    let match
    while ((match = p.exec(content)) !== null) {
      const fn = match[0].split('/').pop() || match[0]
      if (fn && !fn.startsWith('...')) {
        add(fn)
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

function formatModelDisplayName(modelName: string): string {
  if (!modelName) return 'Local Model'
  if (modelName === 'MockLLM') return 'MockLLM (Simulated Engine)'
  const clean = modelName.replace(/\.gguf$/, '')
  if (clean.includes('deepseek')) return 'DeepSeek R1 7B · Reasoning Engine'
  if (clean.includes('phi4')) return 'Phi-4 14B · Deep Synthesis'
  if (clean.includes('coder-7b')) return 'Qwen 2.5 Coder 7B · Code Engine'
  if (clean.includes('llava')) return 'LLaVA 7B · Vision Multimodal'
  if (clean.includes('7b-instruct')) return 'Qwen 2.5 7B Instruct · Q&A'
  if (clean.includes('7b')) return 'Qwen 2.5 7B · General Chat'
  if (clean.includes('4b')) return 'Qwen 1.5 4B · Local Inference'
  if (clean.includes('coder-3b')) return 'Qwen 2.5 Coder 3B · Fast Code'
  if (clean.includes('0.5b')) return 'Qwen 2.5 0.5B · Emergency Fallback'
  return clean
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

export default function ChatCanvas({
  onSend,
  response,
  loading,
  error,
  role,
  sessionId,
  initialMessages,
}: Props) {
  const [input, setInput] = useState('')
  const [messages, setMessages] = useState<ChatMessage[]>(initialMessages || [])
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [uploadStatus, setUploadStatus] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [uploadedFiles, setUploadedFiles] = useState<{ name: string; size: number }[]>([])

  const bottomRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Sync messages when session changes or initialMessages are loaded
  useEffect(() => {
    if (initialMessages !== undefined) {
      setMessages(initialMessages)
    }
  }, [initialMessages, sessionId])

  // Auto-resize textarea dynamically up to 160px for multi-line input
  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    const newHeight = Math.min(Math.max(el.scrollHeight, 40), 160)
    el.style.height = `${newHeight}px`
  }, [input])

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

  const submit = () => {
    const trimmed = input.trim()
    if (!trimmed || loading) return
    setMessages((prev) => [...prev, { role: 'user', content: trimmed }])
    onSend(trimmed)
    setInput('')
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    submit()
  }

  // Multi-line input: Enter submits, Shift+Enter inserts a line break.
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  const processUpload = async (file: File) => {
    setUploading(true)
    setUploadProgress(0)
    setUploadStatus(`Uploading ${file.name}...`)
    try {
      const res = await uploadFile(file, (percent) => setUploadProgress(percent))
      setUploadStatus(`Uploaded ${file.name} to sandbox`)
      setUploadedFiles((prev) => [
        ...prev.filter((f) => f.name !== file.name),
        { name: file.name, size: file.size },
      ])
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

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      processUpload(file)
    }
  }

  return (
    <div
      className="flex flex-col h-full relative"
      style={{ background: 'var(--bg-base)' }}
      onDragOver={(e) => {
        e.preventDefault()
        e.stopPropagation()
        setIsDragging(true)
      }}
      onDragLeave={(e) => {
        e.preventDefault()
        e.stopPropagation()
        setIsDragging(false)
      }}
      onDrop={(e) => {
        e.preventDefault()
        e.stopPropagation()
        setIsDragging(false)
        const file = e.dataTransfer.files?.[0]
        if (file) {
          processUpload(file)
        }
      }}
    >
      {/* Drag & drop overlay */}
      {isDragging && (
        <div
          className="absolute inset-0 z-50 flex flex-col items-center justify-center gap-3"
          style={{
            background: 'rgba(10, 14, 23, 0.88)',
            backdropFilter: 'blur(8px)',
            border: '2px dashed var(--accent)',
          }}
        >
          <div
            className="w-16 h-16 rounded-2xl flex items-center justify-center animate-bounce"
            style={{
              background: 'rgba(0, 229, 160, 0.15)',
              border: '1px solid var(--accent)',
              boxShadow: '0 0 20px var(--accent-glow)',
            }}
          >
            <Upload className="w-8 h-8 text-accent" />
          </div>
          <p className="text-sm font-bold text-text-primary">
            Drop file to upload to sandbox
          </p>
          <p className="text-xs text-text-muted">
            Supports P&ID diagrams, equipment nameplates, PDFs, DOCX, XLSX, logs
          </p>
        </div>
      )}

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
                  {msg.role === 'user' ? (
                    <pre className="whitespace-pre-wrap" style={{ fontFamily: 'var(--font-body)' }}>{msg.content}</pre>
                  ) : (
                    (() => {
                      const { reasoning, answer } = parseReasoningAndAnswer(msg.content)
                      return (
                        <div>
                          {reasoning && (
                            <details
                              className="mb-3 rounded-lg overflow-hidden transition-all group"
                              style={{
                                background: 'rgba(255, 255, 255, 0.03)',
                                border: '1px solid rgba(255, 255, 255, 0.1)',
                              }}
                            >
                              <summary
                                className="px-3 py-2 text-xs font-medium cursor-pointer select-none flex items-center gap-2 transition-colors hover:bg-white/[0.04]"
                                style={{ color: 'var(--text-muted)' }}
                              >
                                <span className="w-1.5 h-1.5 rounded-full" style={{ background: 'var(--accent)' }} />
                                <span className="font-semibold" style={{ color: 'var(--text-secondary)' }}>Reasoning Process</span>
                                <span className="text-[10px] ml-auto opacity-60 group-open:rotate-180 transition-transform">▼</span>
                              </summary>
                              <div
                                className="px-3 py-2.5 text-[11px] leading-relaxed font-mono whitespace-pre-wrap"
                                style={{
                                  color: 'var(--text-secondary)',
                                  borderTop: '1px solid rgba(255, 255, 255, 0.06)',
                                  background: 'rgba(0, 0, 0, 0.15)',
                                }}
                              >
                                {reasoning}
                              </div>
                            </details>
                          )}
                          <pre className="whitespace-pre-wrap" style={{ fontFamily: 'var(--font-body)' }}>{answer}</pre>
                        </div>
                      )
                    })()
                  )}

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
                    <span>Generated by <span style={{ color: 'var(--text-secondary)' }}>{formatModelDisplayName(msg.model_used)}</span></span>
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

      {/* Upload toast + progress */}
      {uploadStatus && (
        <div
          className="mx-5 mb-2 px-3 py-2 rounded-lg flex flex-col gap-1.5 text-xs"
          style={{
            background: 'rgba(0, 229, 160, 0.08)',
            border: '1px solid rgba(0, 229, 160, 0.2)',
            color: 'var(--accent)',
            fontFamily: 'var(--font-mono)',
          }}
        >
          <div className="flex items-center gap-2">
            {uploading ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <CheckCircle className="w-3.5 h-3.5" />
            )}
            <span>{uploadStatus}</span>
            {uploading && <span className="ml-auto">{uploadProgress}%</span>}
          </div>
          {uploading && (
            <div
              className="h-1 rounded-full overflow-hidden"
              style={{ background: 'rgba(0, 229, 160, 0.15)' }}
            >
              <div
                className="h-full rounded-full transition-all duration-200"
                style={{ width: `${uploadProgress}%`, background: 'var(--accent)' }}
              />
            </div>
          )}
        </div>
      )}

      {/* Persistent uploaded sandbox files indicator tray */}
      {uploadedFiles.length > 0 && (
        <div
          className="mx-5 mb-2 px-3 py-2 rounded-lg flex flex-wrap items-center gap-2 text-xs"
          style={{
            background: 'rgba(0, 229, 160, 0.05)',
            border: '1px solid rgba(0, 229, 160, 0.18)',
          }}
        >
          <div className="flex items-center gap-1.5 text-[11px] font-semibold text-accent uppercase font-mono mr-1">
            <CheckCircle className="w-3.5 h-3.5" />
            <span>Sandbox Files ({uploadedFiles.length}):</span>
          </div>
          {uploadedFiles.map((file) => (
            <div
              key={file.name}
              className="flex items-center gap-1.5 px-2 py-1 rounded bg-white/[0.04] border border-white/10 text-xs"
            >
              {getFileIcon(file.name)}
              <span className="font-mono text-text-primary text-[11px] font-medium">{file.name}</span>
              <span className="text-[10px] text-text-muted">({(file.size / 1024).toFixed(0)} KB)</span>
              <button
                type="button"
                onClick={() => setUploadedFiles((prev) => prev.filter((f) => f.name !== file.name))}
                className="hover:text-red-400 text-text-muted transition-colors ml-1"
                title="Dismiss"
              >
                <X className="w-3 h-3" />
              </button>
            </div>
          ))}
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
        <div className="flex items-end gap-2">
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
            className="p-2.5 rounded-lg cursor-pointer transition-all flex items-center justify-center self-center"
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

          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={`Ask the workbench (${role})...`}
            disabled={loading}
            rows={1}
            className="flex-1 rounded-lg px-4 py-2.5 text-[13px] placeholder-sm focus:outline-none focus:ring-1 disabled:opacity-40 resize-none overflow-y-auto"
            style={{
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid var(--border-default)',
              color: 'var(--text-primary)',
              fontFamily: 'var(--font-body)',
              backdropFilter: 'blur(8px)',
              minHeight: '40px',
              maxHeight: '160px',
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
            className="btn-primary flex items-center gap-2 px-4 py-2.5 self-center"
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