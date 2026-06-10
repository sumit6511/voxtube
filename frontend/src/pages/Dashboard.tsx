import { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { ArrowLeft, BarChart3, List, AlertTriangle, MessageSquare } from 'lucide-react'
import { api } from '../api'
import { useStore } from '../store'
import type { ResultsResponse } from '../api'
import SentimentChart    from '../components/SentimentChart'
import TopicsChart       from '../components/TopicsChart'
import CommentsList      from '../components/CommentsList'
import ToxicityPanel     from '../components/ToxicityPanel'
import ChatPanel         from '../components/ChatPanel'
import WordCloud         from '../components/WordCloud'
import LanguageChart     from '../components/LanguageChart'
import SentimentTimeline from '../components/SentimentTimeline'

type Tab = 'overview' | 'comments' | 'toxicity' | 'chat'

const TABS: { id: Tab; label: string; Icon: React.ElementType }[] = [
  { id: 'overview', label: 'Overview',  Icon: BarChart3      },
  { id: 'comments', label: 'Comments',  Icon: List           },
  { id: 'toxicity', label: 'Toxicity',  Icon: AlertTriangle  },
  { id: 'chat',     label: 'Chat',      Icon: MessageSquare  },
]

export default function Dashboard() {
  const { jobId }     = useParams<{ jobId: string }>()
  const navigate      = useNavigate()
  const storeResults  = useStore(s => s.results)
  const setResults    = useStore(s => s.setResults)

  const [results, setLocal] = useState<ResultsResponse | null>(
    storeResults?.job_id === jobId ? storeResults : null
  )
  const [loading, setLoading] = useState(!results)
  const [error,   setError]   = useState('')
  const [tab,     setTab]     = useState<Tab>('overview')

  useEffect(() => {
    if (!jobId || results) return
    api.results(jobId)
      .then(r => { setResults(r); setLocal(r); setLoading(false) })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [jobId])

  if (loading) return <Spinner />
  if (error)   return <ErrorScreen msg={error} onBack={() => navigate('/')} />
  if (!results) return null

  const { sentiment_summary: ss, total_comments, topics, comments } = results
  const toxicCount = comments.filter(c => c.is_toxic).length
  const posPercent = total_comments
    ? Math.round((ss.positive / total_comments) * 100)
    : 0

  return (
    <div className="min-h-screen px-4 py-6 max-w-5xl mx-auto">

      {/* ── Header ─────────────────────────────────────────────────── */}
      <div className="flex items-start gap-3 mb-6">
        <button
          onClick={() => navigate('/')}
          className="mt-0.5 text-gray-500 hover:text-white transition-colors flex-shrink-0"
        >
          <ArrowLeft size={18} />
        </button>
        <div className="min-w-0">
          <h1 className="font-display text-xl font-bold text-white leading-tight truncate">
            {results.video_title ?? 'Analysis Results'}
          </h1>
          <p className="text-xs font-mono text-gray-700 mt-0.5 truncate">{jobId}</p>
        </div>
      </div>

      {/* ── Stats row ──────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6">
        {[
          { label: 'Comments', value: total_comments, color: 'text-white'  },
          { label: 'Positive', value: `${posPercent}%`, color: 'text-pos'  },
          { label: 'Toxic',    value: toxicCount,      color: 'text-tox'   },
          { label: 'Topics',   value: topics.length,   color: 'text-amber' },
        ].map(stat => (
          <div key={stat.label} className="card text-center py-4">
            <div className={`font-display text-2xl font-bold ${stat.color}`}>
              {stat.value}
            </div>
            <div className="label mt-1">{stat.label}</div>
          </div>
        ))}
      </div>

      {/* ── Tab bar ────────────────────────────────────────────────── */}
      <div className="flex gap-0 mb-6 border-b border-base-border overflow-x-auto justify-center">
        {TABS.map(({ id, label, Icon }) => (
          <button
            key={id}
            onClick={() => setTab(id)}
            className={`flex items-center gap-2 px-4 py-2.5 text-sm font-body
                        font-medium border-b-2 -mb-px whitespace-nowrap transition-all
              ${tab === id
                ? 'border-amber text-white'
                : 'border-transparent text-gray-500 hover:text-gray-300'}`}
          >
            <Icon size={14} />
            {label}
          </button>
        ))}
      </div>

      {/* ── Tab content ────────────────────────────────────────────── */}
      {tab === 'overview' && (
        <div className="space-y-4">

          {/* Row 1: Sentiment + Language */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="card"><SentimentChart data={ss} /></div>
            <div className="card"><LanguageChart  comments={comments} /></div>
          </div>

          {/* Row 2: Sentiment timeline */}
          <div className="card">
            <SentimentTimeline comments={comments} />
          </div>

          {/* Row 3: Topics */}
          <div className="card">
            {topics.length > 0
              ? <TopicsChart topics={topics} />
              : (
                <div className="h-48 flex items-center justify-center">
                  <p className="text-gray-600 font-mono text-sm">No topics discovered</p>
                </div>
              )
            }
          </div>

          {/* Row 4: Word cloud */}
          <div className="card">
            <WordCloud comments={comments} />
          </div>

        </div>
      )}

      {tab === 'comments' && (
        <CommentsList comments={comments} topics={topics} />
      )}

      {tab === 'toxicity' && (
        <ToxicityPanel comments={comments} />
      )}

      {tab === 'chat' && (
        <ChatPanel jobId={jobId!} />
      )}

    </div>
  )
}

function Spinner() {
  return (
    <div className="min-h-screen flex items-center justify-center">
      <div className="text-center space-y-3">
        <div className="w-8 h-8 border-2 border-amber border-t-transparent rounded-full
                        animate-spin mx-auto" />
        <p className="text-gray-500 font-mono text-sm">Loading results…</p>
      </div>
    </div>
  )
}

function ErrorScreen({ msg, onBack }: { msg: string; onBack: () => void }) {
  return (
    <div className="min-h-screen flex items-center justify-center px-4">
      <div className="card max-w-md w-full text-center space-y-4">
        <p className="text-neg font-mono text-sm">{msg}</p>
        <button onClick={onBack} className="btn-primary">← Go back</button>
      </div>
    </div>
  )
}
