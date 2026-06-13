import { useEffect, useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { ArrowLeft, Clock, MessageSquare, CheckCircle2, XCircle, Loader2, BarChart3 } from 'lucide-react'
import { api } from '../api'
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
  return new Date(dateStr).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
}

function StatusBadge({ status }: { status: string }) {
  const cfg: Record<string, { icon: React.ReactNode; cls: string; label: string }> = {
    done:    { icon: <CheckCircle2 size={12} />, cls: 'text-pos  bg-pos/10  border-pos/25',  label: 'Done'       },
    failed:  { icon: <XCircle     size={12} />, cls: 'text-neg  bg-neg/10  border-neg/25',  label: 'Failed'     },
    pending: { icon: <Clock       size={12} />, cls: 'text-gray-400 bg-gray-800 border-gray-700', label: 'Pending' },
  }
  const s = cfg[status] ?? { icon: <Loader2 size={12} className="animate-spin" />,
    cls: 'text-amber bg-amber/10 border-amber/25', label: status }
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-mono border ${s.cls}`}>
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

// ── Component ─────────────────────────────────────────────────────────────────

export default function History() {
  const navigate = useNavigate()
  const [jobs,    setJobs]    = useState<JobSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error,   setError]   = useState('')

  useEffect(() => {
    api.jobs()
      .then(d => { setJobs(d.jobs); setLoading(false) })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [])

  return (
    <div className="min-h-screen px-4 py-6 max-w-3xl mx-auto">

      {/* Header */}
      <div className="flex items-center gap-3 mb-6">
        <button onClick={() => navigate('/')} className="text-gray-500 hover:text-white transition-colors">
          <ArrowLeft size={18} />
        </button>
        <div>
          <h1 className="font-display text-xl font-bold text-white">Analysis History</h1>
          <p className="text-xs font-mono text-gray-600 mt-0.5">
            {loading ? '—' : `${jobs.length} job${jobs.length !== 1 ? 's' : ''} total`}
          </p>
        </div>
      </div>

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

      {/* Job list */}
      {!loading && jobs.length > 0 && (
        <div className="space-y-3">
          {jobs.map(job => (
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
                {/* Title */}
                <p className="font-body text-sm font-medium text-white truncate">
                  {job.video_title ?? shortUrl(job.youtube_url)}
                </p>

                {/* URL if title exists */}
                {job.video_title && (
                  <p className="font-mono text-xs text-gray-600 truncate mt-0.5">
                    {shortUrl(job.youtube_url)}
                  </p>
                )}

                {/* Meta row */}
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

              {/* Action */}
              <div className="flex-shrink-0 mt-0.5">
                {job.status === 'done' && (
                  <Link
                    to={`/dashboard/${job.id}`}
                    className="text-xs font-mono text-amber hover:text-amber-glow
                               border border-amber/30 hover:border-amber/60
                               px-3 py-1.5 rounded-lg transition-all"
                  >
                    View →
                  </Link>
                )}
                {['pending','fetching','preprocessing','analyzing',
                  'toxicity','building_topics','building_rag'].includes(job.status) && (
                  <Link
                    to={`/progress/${job.id}`}
                    className="text-xs font-mono text-gray-400 hover:text-white
                               border border-base-border px-3 py-1.5 rounded-lg transition-all"
                  >
                    Progress →
                  </Link>
                )}
              </div>

            </div>
          ))}
        </div>
      )}
    </div>
  )
}
