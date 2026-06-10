import { useEffect, useRef } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { Clapperboard, CheckCircle2, XCircle, Loader2 } from 'lucide-react'
import { api } from '../api'
import { useStore } from '../store'

// Each pipeline stage in display order
const STAGES: { key: string; label: string; progress: number }[] = [
  { key: 'fetching',        label: 'Fetching comments',          progress: 20  },
  { key: 'preprocessing',  label: 'Neplish preprocessing',       progress: 35  },
  { key: 'analyzing',      label: 'Sentiment analysis',          progress: 55  },
  { key: 'toxicity',       label: 'Toxicity detection',          progress: 70  },
  { key: 'building_topics',label: 'Topic modeling (BERTopic)',    progress: 85  },
  { key: 'building_rag',   label: 'Building search index',       progress: 98  },
  { key: 'done',           label: 'Analysis complete',           progress: 100 },
]

function stageIndex(status: string) {
  return STAGES.findIndex(s => s.key === status)
}

export default function Progress() {
  const { jobId }   = useParams<{ jobId: string }>()
  const navigate    = useNavigate()
  const setStatus   = useStore(s => s.setStatus)
  const status      = useStore(s => s.status)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    if (!jobId) return

    async function poll() {
      try {
        const s = await api.status(jobId!)
        setStatus(s)
        if (s.status === 'done') {
          clearInterval(intervalRef.current!)
          navigate(`/dashboard/${jobId}`)
        }
        if (s.status === 'failed') {
          clearInterval(intervalRef.current!)
        }
      } catch {
        // backend temporarily unavailable — keep polling
      }
    }

    poll()
    intervalRef.current = setInterval(poll, 3000)
    return () => clearInterval(intervalRef.current!)
  }, [jobId])

  const current  = status?.status ?? 'pending'
  const progress = status?.progress ?? 0
  const curIdx   = stageIndex(current)
  const failed   = current === 'failed'

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-4 py-16">

      {/* Header */}
      <div className="animate-fade-up mb-10 text-center">
        <div className="flex items-center justify-center gap-2 mb-2">
          <Clapperboard size={18} className="text-amber" />
          <span className="font-display text-xl font-bold text-white">
            Vox<span className="text-amber">Tube</span>
          </span>
        </div>
        {status?.video_title && (
          <p className="text-gray-400 text-sm font-body mt-1 max-w-sm truncate">
            {status.video_title}
          </p>
        )}
      </div>

      {/* Progress card */}
      <div className="animate-fade-up w-full max-w-md card space-y-6">

        {/* Progress bar */}
        <div>
          <div className="flex justify-between items-center mb-2">
            <span className="label">Progress</span>
            <span className="font-mono text-sm text-amber">{progress}%</span>
          </div>
          <div className="h-1.5 bg-base rounded-full overflow-hidden">
            <div
              className="h-full bg-amber rounded-full transition-all duration-700 ease-out"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>

        {/* Stage list */}
        <div className="space-y-1">
          {STAGES.filter(s => s.key !== 'done').map((stage, idx) => {
            const done    = curIdx > idx || current === 'done'
            const active  = curIdx === idx && !failed
            const pending = curIdx < idx && !failed

            return (
              <div
                key={stage.key}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all duration-300
                  ${active  ? 'bg-amber/8 border border-amber/20' : ''}
                  ${done    ? 'opacity-60' : ''}
                  ${pending ? 'opacity-30' : ''}
                `}
              >
                {/* Icon */}
                <div className="w-5 flex-shrink-0 flex items-center justify-center">
                  {done    && <CheckCircle2 size={16} className="text-pos" />}
                  {active  && <Loader2 size={16} className="text-amber animate-spin" />}
                  {pending && <span className="w-3 h-3 rounded-full border border-gray-700" />}
                  {failed && idx === curIdx && <XCircle size={16} className="text-neg" />}
                </div>

                {/* Label */}
                <span className={`text-sm font-body ${active ? 'text-white' : 'text-gray-400'}`}>
                  {stage.label}
                </span>

                {/* Comment count badge (shows after fetch) */}
                {stage.key === 'fetching' && done && status?.comment_count ? (
                  <span className="ml-auto font-mono text-xs text-amber">
                    {status.comment_count} comments
                  </span>
                ) : null}
              </div>
            )
          })}
        </div>

        {/* Error message */}
        {failed && status?.error_message && (
          <div className="bg-neg/5 border border-neg/20 rounded-lg px-4 py-3">
            <p className="text-neg text-xs font-mono">{status.error_message}</p>
            <button
              onClick={() => navigate('/')}
              className="mt-2 text-xs text-gray-400 hover:text-white underline font-mono"
            >
              ← Try again
            </button>
          </div>
        )}
      </div>

      {/* Job ID */}
      <p className="mt-6 text-xs font-mono text-gray-700">
        job: {jobId}
      </p>
    </div>
  )
}
