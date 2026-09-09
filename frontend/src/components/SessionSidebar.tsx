import { useState, useEffect, useCallback, useMemo } from 'react'
import {
  MessageSquare,
  Plus,
  Trash2,
  ChevronLeft,
  ChevronRight,
  Clock,
  MessageCircle,
  FolderKanban,
  FolderPlus,
  Search,
  Folder,
  Layers,
  MoreVertical,
  Check,
  X
} from 'lucide-react'
import {
  fetchSessions,
  deleteSession,
  fetchProjects,
  createProject,
  deleteProject,
  assignSessionToProject,
  type SessionSummary,
  type ProjectSummary
} from '../hooks/useApi'

interface Props {
  activeSessionId: string | null
  onSelectSession: (sessionId: string | null) => void
  activeProjectId: string | null
  onSelectProject: (projectId: string | null) => void
  collapsed: boolean
  onToggleCollapse: () => void
}

function formatRelativeTime(timestamp: number): string {
  if (!timestamp) return ''
  const now = Date.now() / 1000
  const diffSec = Math.max(0, now - timestamp)
  if (diffSec < 60) return 'Just now'
  if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m ago`
  if (diffSec < 86400) return `${Math.floor(diffSec / 3600)}h ago`
  if (diffSec < 604800) return `${Math.floor(diffSec / 86400)}d ago`
  const d = new Date(timestamp * 1000)
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

export default function SessionSidebar({
  activeSessionId,
  onSelectSession,
  activeProjectId,
  onSelectProject,
  collapsed,
  onToggleCollapse,
}: Props) {
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [projects, setProjects] = useState<ProjectSummary[]>([])
  const [loading, setLoading] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [deletingSessionId, setDeletingSessionId] = useState<string | null>(null)
  const [deletingProjectId, setDeletingProjectId] = useState<string | null>(null)

  // Project creation state
  const [showNewProjectInput, setShowNewProjectInput] = useState(false)
  const [newProjectName, setNewProjectName] = useState('')

  // Assign session modal / menu
  const [movingSessionId, setMovingSessionId] = useState<string | null>(null)

  const loadProjects = useCallback(async () => {
    try {
      const projs = await fetchProjects()
      setProjects(projs)
    } catch (err) {
      console.error('Failed to load projects:', err)
    }
  }, [])

  const loadSessions = useCallback(async () => {
    try {
      setLoading(true)
      const list = await fetchSessions(100, 0, activeProjectId || undefined)
      setSessions(list)
    } catch (err) {
      console.error('Failed to load sessions:', err)
    } finally {
      setLoading(false)
    }
  }, [activeProjectId])

  useEffect(() => {
    loadProjects()
  }, [loadProjects])

  useEffect(() => {
    loadSessions()
  }, [loadSessions, activeSessionId])

  const handleNewChat = () => {
    onSelectSession(null)
  }

  const handleCreateProject = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!newProjectName.trim()) return
    try {
      const created = await createProject(newProjectName.trim())
      setNewProjectName('')
      setShowNewProjectInput(false)
      await loadProjects()
      onSelectProject(created.id)
    } catch (err) {
      console.error('Failed to create project:', err)
    }
  }

  const handleDeleteProject = async (e: React.MouseEvent, pid: string) => {
    e.stopPropagation()
    if (deletingProjectId) return
    try {
      setDeletingProjectId(pid)
      await deleteProject(pid)
      if (activeProjectId === pid) {
        onSelectProject(null)
      }
      await loadProjects()
      await loadSessions()
    } catch (err) {
      console.error('Failed to delete project:', err)
    } finally {
      setDeletingProjectId(null)
    }
  }

  const handleDeleteSession = async (e: React.MouseEvent, sid: string) => {
    e.stopPropagation()
    if (deletingSessionId) return
    try {
      setDeletingSessionId(sid)
      await deleteSession(sid)
      setSessions((prev) => prev.filter((s) => s.id !== sid))
      if (activeSessionId === sid) {
        onSelectSession(null)
      }
      await loadProjects()
    } catch (err) {
      console.error('Failed to delete session:', err)
    } finally {
      setDeletingSessionId(null)
    }
  }

  const handleAssignProject = async (sessionId: string, targetProjectId: string | null) => {
    try {
      await assignSessionToProject(sessionId, targetProjectId)
      setMovingSessionId(null)
      await loadProjects()
      await loadSessions()
    } catch (err) {
      console.error('Failed to assign session to project:', err)
    }
  }

  // Filter sessions by search
  const filteredSessions = useMemo(() => {
    if (!searchQuery.trim()) return sessions
    const q = searchQuery.toLowerCase()
    return sessions.filter((s) => s.title.toLowerCase().includes(q))
  }, [sessions, searchQuery])

  // Total session count
  const totalCount = useMemo(() => {
    return projects.reduce((acc, p) => acc + p.session_count, 0)
  }, [projects])

  const activeProject = useMemo(() => {
    return projects.find((p) => p.id === activeProjectId) || null
  }, [projects, activeProjectId])

  if (collapsed) {
    return (
      <div
        className="h-full flex flex-col items-center py-3.5 px-1.5 transition-all duration-300 select-none"
        style={{
          width: '54px',
          background: 'var(--bg-surface)',
          borderRight: '1px solid var(--border-subtle)',
        }}
      >
        <button
          onClick={onToggleCollapse}
          className="p-2 rounded-lg hover:bg-white/5 text-text-muted hover:text-text-primary transition-colors mb-3"
          title="Expand workspace sidebar"
        >
          <ChevronRight className="w-4 h-4" />
        </button>

        <button
          onClick={handleNewChat}
          className="p-2 rounded-lg transition-all"
          style={{
            background: 'rgba(0, 229, 160, 0.12)',
            border: '1px solid rgba(0, 229, 160, 0.25)',
            color: 'var(--accent)',
          }}
          title={activeProject ? `New Chat in ${activeProject.name}` : 'New Conversation'}
        >
          <Plus className="w-4 h-4" />
        </button>

        <div className="w-6 h-px my-3" style={{ background: 'var(--border-subtle)' }} />

        {/* Collapsed Projects shortcut */}
        <button
          onClick={() => onSelectProject(null)}
          className="w-8 h-8 rounded-lg flex items-center justify-center transition-all mb-1.5"
          style={{
            background: activeProjectId === null ? 'rgba(0, 229, 160, 0.15)' : 'rgba(255, 255, 255, 0.03)',
            border: activeProjectId === null ? '1px solid var(--accent)' : '1px solid var(--border-subtle)',
            color: activeProjectId === null ? 'var(--accent)' : 'var(--text-muted)',
          }}
          title="All Conversations"
        >
          <Layers className="w-3.5 h-3.5" />
        </button>

        <div className="flex-1 overflow-y-auto mt-2 w-full flex flex-col items-center gap-1.5">
          {filteredSessions.slice(0, 8).map((s) => {
            const isActive = s.id === activeSessionId
            return (
              <button
                key={s.id}
                onClick={() => onSelectSession(s.id)}
                className="w-8 h-8 rounded-lg flex items-center justify-center transition-all relative"
                style={{
                  background: isActive ? 'rgba(0, 229, 160, 0.15)' : 'rgba(255, 255, 255, 0.03)',
                  border: isActive ? '1px solid var(--accent)' : '1px solid var(--border-subtle)',
                  color: isActive ? 'var(--accent)' : 'var(--text-muted)',
                }}
                title={s.title}
              >
                <MessageSquare className="w-3.5 h-3.5" />
              </button>
            )
          })}
        </div>
      </div>
    )
  }

  return (
    <div
      className="h-full flex flex-col transition-all duration-300 select-none text-xs"
      style={{
        width: '280px',
        background: 'var(--bg-surface)',
        borderRight: '1px solid var(--border-subtle)',
      }}
    >
      {/* Header bar */}
      <div
        className="flex items-center justify-between px-3.5 py-3"
        style={{ borderBottom: '1px solid var(--border-subtle)' }}
      >
        <div className="flex items-center gap-2">
          <MessageCircle className="w-4 h-4 text-accent" />
          <span
            className="font-semibold uppercase tracking-wider"
            style={{ color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}
          >
            Workspace
          </span>
          <span
            className="text-[10px] px-1.5 py-0.5 rounded-full font-medium"
            style={{
              background: 'rgba(255, 255, 255, 0.06)',
              color: 'var(--text-muted)',
              fontFamily: 'var(--font-mono)',
            }}
          >
            {sessions.length}
          </span>
        </div>

        <button
          onClick={onToggleCollapse}
          className="p-1 rounded-md hover:bg-white/5 text-text-muted hover:text-text-primary transition-colors"
          title="Collapse sidebar"
        >
          <ChevronLeft className="w-4 h-4" />
        </button>
      </div>

      {/* New Conversation Button */}
      <div className="p-3 pb-2">
        <button
          onClick={handleNewChat}
          className="w-full flex items-center justify-center gap-2 py-2 px-3 rounded-lg font-semibold transition-all duration-200"
          style={{
            background: 'rgba(0, 229, 160, 0.1)',
            border: '1px solid rgba(0, 229, 160, 0.25)',
            color: 'var(--accent)',
            boxShadow: '0 2px 8px rgba(0, 229, 160, 0.08)',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = 'rgba(0, 229, 160, 0.18)'
            e.currentTarget.style.borderColor = 'var(--accent)'
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = 'rgba(0, 229, 160, 0.1)'
            e.currentTarget.style.borderColor = 'rgba(0, 229, 160, 0.25)'
          }}
        >
          <Plus className="w-3.5 h-3.5" />
          <span>{activeProject ? `New in ${activeProject.name}` : 'New Conversation'}</span>
        </button>
      </div>

      {/* Projects Section */}
      <div className="px-3 pt-2 pb-1">
        <div className="flex items-center justify-between mb-1.5">
          <div className="flex items-center gap-1.5">
            <FolderKanban className="w-3.5 h-3.5 text-accent/80" />
            <span
              className="text-[10px] font-bold uppercase tracking-wider text-text-muted"
              style={{ fontFamily: 'var(--font-mono)' }}
            >
              Projects
            </span>
          </div>
          <button
            onClick={() => setShowNewProjectInput((v) => !v)}
            className="flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] text-text-muted hover:text-accent hover:bg-white/5 transition-all"
            title="Create new project"
          >
            <FolderPlus className="w-3 h-3" />
            <span>+ Project</span>
          </button>
        </div>

        {/* Inline Create Project Form */}
        {showNewProjectInput && (
          <form onSubmit={handleCreateProject} className="mb-2 p-2 rounded-lg bg-white/5 border border-white/10">
            <input
              type="text"
              placeholder="Project name..."
              value={newProjectName}
              onChange={(e) => setNewProjectName(e.target.value)}
              autoFocus
              className="w-full bg-black/40 border border-white/15 rounded px-2 py-1 text-xs text-text-primary focus:outline-none focus:border-accent"
            />
            <div className="flex items-center justify-end gap-1.5 mt-2">
              <button
                type="button"
                onClick={() => {
                  setShowNewProjectInput(false)
                  setNewProjectName('')
                }}
                className="px-2 py-0.5 rounded text-[11px] text-text-muted hover:text-text-primary"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={!newProjectName.trim()}
                className="px-2.5 py-0.5 rounded text-[11px] font-medium bg-accent text-bg-base hover:bg-accent-hover transition-colors disabled:opacity-40"
              >
                Create
              </button>
            </div>
          </form>
        )}

        {/* Project List / Grouping Selector */}
        <div className="space-y-0.5 max-h-36 overflow-y-auto pr-0.5">
          {/* All Chats Option */}
          <div
            onClick={() => onSelectProject(null)}
            className="flex items-center justify-between px-2 py-1.5 rounded cursor-pointer transition-colors"
            style={{
              background: activeProjectId === null ? 'rgba(0, 229, 160, 0.12)' : 'transparent',
              color: activeProjectId === null ? 'var(--accent)' : 'var(--text-secondary)',
              border: activeProjectId === null ? '1px solid rgba(0, 229, 160, 0.3)' : '1px solid transparent',
            }}
          >
            <div className="flex items-center gap-2 truncate">
              <Layers className="w-3 h-3 flex-shrink-0" />
              <span className="truncate font-medium">All Conversations</span>
            </div>
          </div>

          {/* Standalone Option */}
          <div
            onClick={() => onSelectProject('standalone')}
            className="flex items-center justify-between px-2 py-1.5 rounded cursor-pointer transition-colors"
            style={{
              background: activeProjectId === 'standalone' ? 'rgba(0, 229, 160, 0.12)' : 'transparent',
              color: activeProjectId === 'standalone' ? 'var(--accent)' : 'var(--text-secondary)',
              border: activeProjectId === 'standalone' ? '1px solid rgba(0, 229, 160, 0.3)' : '1px solid transparent',
            }}
          >
            <div className="flex items-center gap-2 truncate">
              <Folder className="w-3 h-3 flex-shrink-0 text-text-muted" />
              <span className="truncate">Standalone (No Project)</span>
            </div>
          </div>

          {/* User Projects */}
          {projects.map((proj) => {
            const isSelected = activeProjectId === proj.id
            return (
              <div
                key={proj.id}
                onClick={() => onSelectProject(proj.id)}
                className="group flex items-center justify-between px-2 py-1.5 rounded cursor-pointer transition-colors relative"
                style={{
                  background: isSelected ? 'rgba(0, 229, 160, 0.12)' : 'transparent',
                  color: isSelected ? 'var(--accent)' : 'var(--text-secondary)',
                  border: isSelected ? '1px solid rgba(0, 229, 160, 0.3)' : '1px solid transparent',
                }}
              >
                <div className="flex items-center gap-2 truncate flex-1 min-w-0">
                  <Folder className="w-3 h-3 flex-shrink-0" style={{ color: isSelected ? 'var(--accent)' : '#a78bfa' }} />
                  <span className="truncate font-medium">{proj.name}</span>
                </div>

                <div className="flex items-center gap-1 flex-shrink-0 ml-1">
                  <span
                    className="text-[10px] px-1.5 py-0.2 rounded-full font-mono"
                    style={{
                      background: 'rgba(255, 255, 255, 0.06)',
                      color: isSelected ? 'var(--accent)' : 'var(--text-muted)',
                    }}
                    title={`${proj.session_count} chats in this project`}
                  >
                    {proj.session_count}
                  </span>
                  <button
                    onClick={(e) => handleDeleteProject(e, proj.id)}
                    disabled={deletingProjectId === proj.id}
                    className="opacity-0 group-hover:opacity-100 p-0.5 rounded hover:bg-red-500/10 text-text-muted hover:text-red-400 transition-opacity"
                    title="Delete project (conversations become standalone)"
                  >
                    <Trash2 className="w-2.5 h-2.5" />
                  </button>
                </div>
              </div>
            )
          })}
        </div>
      </div>

      <div className="px-3 py-1.5">
        <div className="w-full h-px" style={{ background: 'var(--border-subtle)' }} />
      </div>

      {/* Search Input */}
      <div className="px-3 pb-2">
        <div className="relative">
          <Search className="w-3 h-3 absolute left-2.5 top-1/2 -translate-y-1/2 text-text-muted" />
          <input
            type="text"
            placeholder={activeProject ? `Search in ${activeProject.name}...` : 'Search conversations...'}
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-7 pr-2 py-1 rounded-md text-[11px] text-text-primary placeholder:text-text-muted/60 focus:outline-none"
            style={{
              background: 'rgba(255, 255, 255, 0.04)',
              border: '1px solid var(--border-subtle)',
            }}
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery('')}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-text-muted hover:text-text-primary"
            >
              <X className="w-2.5 h-2.5" />
            </button>
          )}
        </div>
      </div>

      {/* Active Filter Pill */}
      {activeProject && (
        <div className="px-3 pb-1 flex items-center justify-between">
          <div className="flex items-center gap-1.5 text-[11px] text-accent truncate">
            <span className="truncate">Filtered: <strong>{activeProject.name}</strong></span>
          </div>
          <button
            onClick={() => onSelectProject(null)}
            className="text-[10px] text-text-muted hover:text-text-primary underline ml-1"
          >
            Clear
          </button>
        </div>
      )}

      {/* Sessions list */}
      <div className="flex-1 overflow-y-auto px-2 pb-3 space-y-1">
        {filteredSessions.length === 0 && !loading ? (
          <div className="px-3 py-6 text-center">
            <MessageSquare className="w-5 h-5 mx-auto mb-1.5 text-text-muted/40" />
            <p className="text-[11px] text-text-muted">
              {searchQuery
                ? 'No matching conversations'
                : activeProject
                ? `No chats in ${activeProject.name} yet`
                : 'No previous conversations'}
            </p>
            <p className="text-[10px] text-text-muted/60 mt-0.5">
              Start chatting to persist locally
            </p>
          </div>
        ) : (
          filteredSessions.map((sess) => {
            const isActive = sess.id === activeSessionId
            const isMoving = movingSessionId === sess.id
            const sessionProject = projects.find((p) => p.id === sess.project_id)

            return (
              <div key={sess.id} className="relative">
                <div
                  onClick={() => onSelectSession(sess.id)}
                  className="group relative flex items-start gap-2 px-2.5 py-2 rounded-lg cursor-pointer transition-all duration-150"
                  style={{
                    background: isActive ? 'rgba(0, 229, 160, 0.08)' : 'transparent',
                    border: isActive ? '1px solid rgba(0, 229, 160, 0.3)' : '1px solid transparent',
                  }}
                  onMouseEnter={(e) => {
                    if (!isActive) e.currentTarget.style.background = 'rgba(255, 255, 255, 0.03)'
                  }}
                  onMouseLeave={(e) => {
                    if (!isActive) e.currentTarget.style.background = 'transparent'
                  }}
                >
                  <MessageSquare
                    className="w-3.5 h-3.5 mt-0.5 flex-shrink-0"
                    style={{ color: isActive ? 'var(--accent)' : 'var(--text-muted)' }}
                  />

                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between gap-1">
                      <p
                        className="text-xs font-medium truncate"
                        style={{
                          color: isActive ? 'var(--text-primary)' : 'var(--text-secondary)',
                          fontWeight: isActive ? 600 : 400,
                        }}
                      >
                        {sess.title}
                      </p>
                    </div>

                    <div className="flex items-center gap-1.5 mt-1 flex-wrap">
                      <span
                        className="text-[10px] flex items-center gap-1"
                        style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}
                      >
                        <Clock className="w-2.5 h-2.5" />
                        {formatRelativeTime(sess.updated_at)}
                      </span>
                      <span
                        className="text-[10px]"
                        style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}
                      >
                        · {sess.message_count} msg{sess.message_count !== 1 ? 's' : ''}
                      </span>
                      {sessionProject && activeProjectId === null && (
                        <span
                          className="text-[9px] px-1 py-0.2 rounded bg-purple-500/10 text-purple-300 font-mono"
                          title={`Part of ${sessionProject.name}`}
                        >
                          {sessionProject.name}
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Actions: Move to project + Delete */}
                  <div className="opacity-0 group-hover:opacity-100 flex items-center gap-0.5 flex-shrink-0 self-center transition-opacity">
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        setMovingSessionId(isMoving ? null : sess.id)
                      }}
                      className="p-1 rounded hover:bg-white/10 text-text-muted hover:text-accent transition-colors"
                      title="Move to project..."
                    >
                      <Folder className="w-3 h-3" />
                    </button>
                    <button
                      onClick={(e) => handleDeleteSession(e, sess.id)}
                      disabled={deletingSessionId === sess.id}
                      className="p-1 rounded hover:bg-red-500/10 text-text-muted hover:text-red-400 transition-colors"
                      title="Delete conversation"
                    >
                      <Trash2 className="w-3 h-3" />
                    </button>
                  </div>
                </div>

                {/* Move to Project Dropdown Popover */}
                {isMoving && (
                  <div
                    className="absolute right-2 top-full mt-1 z-50 p-1.5 rounded-lg shadow-xl"
                    style={{
                      background: 'var(--bg-card)',
                      border: '1px solid var(--border-subtle)',
                      width: '200px',
                    }}
                    onClick={(e) => e.stopPropagation()}
                  >
                    <div className="flex items-center justify-between pb-1 px-1 border-b border-white/10 mb-1">
                      <span className="text-[10px] font-bold text-text-muted uppercase">Move to Project</span>
                      <button
                        onClick={() => setMovingSessionId(null)}
                        className="text-text-muted hover:text-text-primary"
                      >
                        <X className="w-3 h-3" />
                      </button>
                    </div>

                    {/* Standalone option */}
                    <button
                      onClick={() => handleAssignProject(sess.id, null)}
                      className="w-full flex items-center justify-between px-2 py-1 rounded text-[11px] text-left hover:bg-white/5 text-text-secondary hover:text-text-primary transition-colors"
                    >
                      <span>Standalone (None)</span>
                      {!sess.project_id && <Check className="w-3 h-3 text-accent" />}
                    </button>

                    {/* Project options */}
                    {projects.map((p) => (
                      <button
                        key={p.id}
                        onClick={() => handleAssignProject(sess.id, p.id)}
                        className="w-full flex items-center justify-between px-2 py-1 rounded text-[11px] text-left hover:bg-white/5 text-text-secondary hover:text-text-primary transition-colors"
                      >
                        <span className="truncate">{p.name}</span>
                        {sess.project_id === p.id && <Check className="w-3 h-3 text-accent" />}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}
