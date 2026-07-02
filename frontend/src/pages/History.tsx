import { useEffect, useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import {
  ArrowLeft, Clock, MessageSquare, CheckCircle2,
  XCircle, Loader2, BarChart3, Trash2, Search, X,
} from 'lucide-react'
import { api } from '../api'
import { pushToast } from '../hooks/useToast'
import type { JobSummary } from '../api'

// ── Helpers ───────────────────────────────────────────────────────────────────

function timeAgo(dateStr?: string | null): string {
  if (!dateStr) return '—'
  const diff = Date.now() - new Date(dateStr).getTime()
  const mins = Math.floor(diff / 60_000)
  if (mins < 1)   return 'just now'
  if (mins < 60)  return `${mins}m ago`
  const hrs = Math.floor(mins / 60)
  if (hrs  < 24)  return `${hrs}h ago`
  const days = Math.floor(hrs / 24)
  if (days < 7)   return `${days}d ago`
  return new Date(dateStr).toLocaleDateString('en-US', {
    month: 'short', day: 'numeric', year: 'numeric'
  })
}

function StatusBadge({ status }: { status: string }) {
  const cfg: Record<string, { icon: React.ReactNode; cls: string; label: string }> = {
    done:    { icon: <CheckCircle2 size={12} />, cls: 'text-pos bg-pos/10 border-pos/25',            label: 'Done'    },
    failed:  { icon: <XCircle     size={12} />, cls: 'text-neg bg-neg/10 border-neg/25',            label: 'Failed'  },
    pending: { icon: <Clock       size={12} />, cls: 'text-gray-400 bg-gray-800 border-gray-700',   label: 'Pending' },
  }
  const s = cfg[status] ?? {
    icon:  <Loader2 size={12} className="animate-spin" />,
    cls:   'text-amber bg-amber/10 border-amber/25',
    label: status,
  }
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full
                      text-xs font-mono border ${s.cls}`}>
      {s.icon} {s.label}
    </span>
  )
}

function shortUrl(url: string): string {
  try {
    const u = new URL(url)
    const v = u.searchParams.get('v')
    return v ? `youtube.com/watch?v=${v}` : url.replace('https://', '').slice(0, 40)
  } catch { return url.slice(0, 40) }
}

// ── Delete confirmation inline ────────────────────────────────────────────────

function DeleteButton({ jobId, onDeleted }: { jobId: string; onDeleted: () => void }) {
  const [confirming, setConfirming] = useState(false)
  const [deleting,   setDeleting]   = useState(false)

  async function handleDelete() {
    setDeleting(true)
    try {
      await api.deleteJob(jobId)
      pushToast('Job deleted', 'success')
      onDeleted()
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Delete failed'
      pushToast(msg, 'error')
    } finally {
      setDeleting(false)
      setConfirming(false)
    }
  }

  if (confirming) {
    return (
      <div className="flex items-center gap-2 flex-shrink-0">
        <span className="text-xs font-mono text-gray-500">Delete?</span>
        <button
          onClick={handleDelete}
          disabled={deleting}
          className="text-xs font-mono text-neg hover:text-neg/80
                     border border-neg/30 px-2 py-1 rounded-lg transition-colors"
        >
          {deleting ? <Loader2 size={11} className="animate-spin" /> : 'Yes'}
        </button>
        <button
          onClick={() => setConfirming(false)}
          className="text-xs font-mono text-gray-500 hover:text-white
                     border border-base-border px-2 py-1 rounded-lg transition-colors"
        >
          No
        </button>
      </div>
    )
  }

  return (
    <button
      onClick={() => setConfirming(true)}
      title="Delete job"
      className="flex-shrink-0 text-gray-700 hover:text-neg transition-colors p-1.5
                 rounded-lg hover:bg-neg/5"
    >
      <Trash2 size={14} />
    </button>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

const STATUS_FILTERS = ['All', 'Done', 'Failed', 'In Progress']

function matchesStatus(job: JobSummary, filter: string): boolean {
  if (filter === 'All') return true
  if (filter === 'Done')   return job.status === 'done'
  if (filter === 'Failed') return job.status === 'failed'
  return !['done', 'failed'].includes(job.status)
}

export default function History() {
  const navigate = useNavigate()
  const [jobs,         setJobs]         = useState<JobSummary[]>([])
  const [loading,      setLoading]      = useState(true)
  const [error,        setError]        = useState('')
  const [search,       setSearch]       = useState('')
  const [statusFilter, setStatusFilter] = useState('All')

  useEffect(() => {
    api.jobs()
      .then(d => { setJobs(d.jobs); setLoading(false) })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [])

  function removeJob(id: string) {
    setJobs(prev => prev.filter(j => j.id !== id))
  }

  // Search + status filter
  const filtered = jobs.filter(job => {
    if (!matchesStatus(job, statusFilter)) return false
    if (!search) return true
    const q = search.toLowerCase()
    return (
      (job.video_title ?? '').toLowerCase().includes(q) ||
      job.youtube_url.toLowerCase().includes(q)
    )
  })

  return (
    <div className="min-h-screen px-4 py-6 max-w-3xl mx-auto">

      {/* Header */}
      <div className="flex items-center gap-3 mb-6">
        <button onClick={() => navigate('/')}
          className="text-gray-500 hover:text-white transition-colors">
          <ArrowLeft size={18} />
        </button>
        <div>
          <h1 className="font-display text-xl font-bold" style={{ color: 'var(--color-text)' }}>
            Analysis History
          </h1>
          <p className="text-xs font-mono text-gray-600 mt-0.5">
            {loading ? '—' : `${jobs.length} job${jobs.length !== 1 ? 's' : ''} total`}
          </p>
        </div>
      </div>

      {/* Search + status filter */}
      {!loading && jobs.length > 0 && (
        <div className="flex flex-col sm:flex-row gap-2 mb-4">
          {/* Search */}
          <div className="relative flex-1">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-600" />
            <input
              className="input-field pl-8 py-2 text-xs w-full"
              placeholder="Search by title or URL…"
              value={search}
              onChange={e => setSearch(e.target.value)}
            />
            {search && (
              <button onClick={() => setSearch('')}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-600
                           hover:text-gray-400 transition-colors">
                <X size={13} />
              </button>
            )}
          </div>

          {/* Status pills */}
          <div className="flex gap-1.5 flex-shrink-0">
            {STATUS_FILTERS.map(f => (
              <button key={f} onClick={() => setStatusFilter(f)}
                className={`text-xs font-mono px-3 py-2 rounded-lg border transition-all
                  ${statusFilter === f
                    ? 'bg-amber text-gray-950 border-amber font-medium'
                    : 'border-base-border text-gray-500 hover:text-gray-300'}`}>
                {f}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* States */}
      {loading && (
        <div className="flex items-center justify-center py-20 gap-2">
          <Loader2 size={18} className="animate-spin text-amber" />
          <span className="font-mono text-sm text-gray-500">Loading…</span>
        </div>
      )}

      {error && (
        <div className="card border-neg/20 bg-neg/5 text-neg font-mono text-sm">{error}</div>
      )}

      {!loading && !error && jobs.length === 0 && (
        <div className="text-center py-20 space-y-3">
          <BarChart3 size={32} className="text-gray-700 mx-auto" />
          <p className="text-gray-500 font-mono text-sm">No analyses yet.</p>
          <Link to="/" className="btn-primary inline-flex items-center gap-2 text-sm">
            Analyze a video
          </Link>
        </div>
      )}

      {!loading && !error && jobs.length > 0 && filtered.length === 0 && (
        <div className="text-center py-16">
          <p className="text-gray-500 font-mono text-sm">No jobs match your filters.</p>
          <button onClick={() => { setSearch(''); setStatusFilter('All') }}
            className="mt-2 text-xs font-mono text-amber hover:underline">
            Clear filters
          </button>
        </div>
      )}

      {/* Job list */}
      {!loading && filtered.length > 0 && (
        <div className="space-y-3">
          {filtered.map(job => (
            <div key={job.id} className="card flex items-start gap-4">

              {/* Status icon */}
              <div className="mt-0.5 flex-shrink-0">
                {job.status === 'done'   && <CheckCircle2 size={18} className="text-pos" />}
                {job.status === 'failed' && <XCircle      size={18} className="text-neg" />}
                {!['done','failed'].includes(job.status) &&
                  <Loader2 size={18} className="text-amber animate-spin" />}
              </div>

              {/* Info */}
              <div className="flex-1 min-w-0">
                <p className="font-body text-sm font-medium truncate"
                   style={{ color: 'var(--color-text)' }}>
                  {job.video_title ?? shortUrl(job.youtube_url)}
                </p>
                {job.video_title && (
                  <p className="font-mono text-xs text-gray-600 truncate mt-0.5">
                    {shortUrl(job.youtube_url)}
                  </p>
                )}
                <div className="flex flex-wrap items-center gap-3 mt-2">
                  <StatusBadge status={job.status} />
                  {job.comment_count > 0 && (
                    <span className="flex items-center gap-1 text-xs font-mono text-gray-500">
                      <MessageSquare size={11} /> {job.comment_count}
                    </span>
                  )}
                  <span className="flex items-center gap-1 text-xs font-mono text-gray-600">
                    <Clock size={11} /> {timeAgo(job.created_at)}
                  </span>
                </div>
              </div>

              {/* Actions */}
              <div className="flex items-center gap-2 flex-shrink-0 mt-0.5">
                {job.status === 'done' && (
                  <Link to={`/dashboard/${job.id}`}
                    className="text-xs font-mono text-amber hover:text-amber-glow
                               border border-amber/30 hover:border-amber/60
                               px-3 py-1.5 rounded-lg transition-all">
                    View →
                  </Link>
                )}
                {!['done','failed'].includes(job.status) && (
                  <Link to={`/progress/${job.id}`}
                    className="text-xs font-mono text-gray-400 hover:text-white
                               border border-base-border px-3 py-1.5 rounded-lg transition-all">
                    Progress →
                  </Link>
                )}
                <DeleteButton jobId={job.id} onDeleted={() => removeJob(job.id)} />
              </div>

            </div>
          ))}
        </div>
      )}

      {/* Filtered count */}
      {!loading && jobs.length > 0 && (
        <p className="text-xs font-mono text-gray-700 mt-4 text-center">
          {filtered.length === jobs.length
            ? `${jobs.length} jobs`
            : `${filtered.length} of ${jobs.length} jobs`}
        </p>
      )}
    </div>
  )
}
