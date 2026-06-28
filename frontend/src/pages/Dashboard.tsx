import { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { ArrowLeft, BarChart3, List, AlertTriangle, MessageSquare,
         FileSpreadsheet, FileText, Tag } from 'lucide-react'
import ThemeToggle from '../components/ThemeToggle'
import { api, BASE } from '../api'
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
import ChartCard         from '../components/ChartCard'
import EntitiesPanel     from '../components/EntitiesPanel'

type Tab = 'overview' | 'comments' | 'toxicity' | 'entities' | 'chat'

const TABS: { id: Tab; label: string; Icon: React.ElementType }[] = [
  { id: 'overview',  label: 'Overview',  Icon: BarChart3      },
  { id: 'comments',  label: 'Comments',  Icon: List           },
  { id: 'toxicity',  label: 'Toxicity',  Icon: AlertTriangle  },
  { id: 'entities',  label: 'Entities',  Icon: Tag            },
  { id: 'chat',      label: 'Chat',      Icon: MessageSquare  },
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
      <div className="flex items-start gap-3 mb-5">
        <button
          onClick={() => navigate('/')}
          className="mt-0.5 text-gray-500 hover:text-white transition-colors flex-shrink-0"
        >
          <ArrowLeft size={18} />
        </button>

        {/* Thumbnail */}
        {results.video_id && (
          <a
            href={results.youtube_url ?? `https://youtube.com/watch?v=${results.video_id}`}
            target="_blank" rel="noopener noreferrer"
            className="flex-shrink-0 rounded-lg overflow-hidden border border-base-border
                       hover:border-amber/40 transition-colors"
          >
            <img
              src={`https://img.youtube.com/vi/${results.video_id}/mqdefault.jpg`}
              alt="thumbnail"
              className="w-24 h-[54px] object-cover block"
              onError={e => { (e.target as HTMLImageElement).style.display = 'none' }}
            />
          </a>
        )}

        {/* Title + meta */}
        <div className="min-w-0 flex-1">
          <h1 className="font-display text-xl font-bold text-white leading-tight truncate">
            {results.video_title ?? 'Analysis Results'}
          </h1>
          <div className="flex flex-wrap items-center gap-3 mt-1">
            {results.channel_title && (
              <span className="text-xs font-mono text-gray-500">{results.channel_title}</span>
            )}
            {results.view_count != null && (
              <span className="text-xs font-mono text-gray-600">
                {results.view_count.toLocaleString()} views
              </span>
            )}
            {results.like_count != null && (
              <span className="text-xs font-mono text-gray-600">
                {results.like_count.toLocaleString()} likes
              </span>
            )}
          </div>
          <p className="text-xs font-mono text-gray-800 mt-0.5 truncate">{jobId}</p>
        </div>

        {/* Theme + Export buttons */}
        <ThemeToggle />
        <a
          href={`${BASE}/export/${jobId}/excel`}
          download
          className="flex-shrink-0 flex items-center gap-1.5 text-xs font-mono text-gray-400
                     hover:text-amber border border-base-border hover:border-amber/40
                     px-3 py-2 rounded-lg transition-all whitespace-nowrap"
        >
          <FileSpreadsheet size={14} />
          Excel
        </a>
        <a
          href={`${BASE}/export/${jobId}/pdf`}
          download
          className="flex-shrink-0 flex items-center gap-1.5 text-xs font-mono text-gray-400
                     hover:text-amber border border-base-border hover:border-amber/40
                     px-3 py-2 rounded-lg transition-all whitespace-nowrap"
        >
          <FileText size={14} />
          PDF
        </a>
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
            <ChartCard title="Sentiment distribution" filename="sentiment-distribution">
              <SentimentChart data={ss} />
            </ChartCard>
            <ChartCard title="Language breakdown" filename="language-breakdown">
              <LanguageChart comments={comments} />
            </ChartCard>
          </div>

          {/* Row 2: Sentiment timeline */}
          <ChartCard title="Sentiment over time" filename="sentiment-timeline">
            <SentimentTimeline comments={comments} />
          </ChartCard>

          {/* Row 3: Topics */}
          <ChartCard title="Per-topic sentiment" filename="topics-sentiment">
            {topics.length > 0
              ? <TopicsChart topics={topics} />
              : (
                <div className="h-48 flex items-center justify-center">
                  <p className="text-gray-600 font-mono text-sm">No topics discovered</p>
                </div>
              )
            }
          </ChartCard>

          {/* Row 4: Word cloud */}
          <ChartCard title="Word cloud" filename="word-cloud">
            <WordCloud comments={comments} />
          </ChartCard>

        </div>
      )}

      {tab === 'comments' && (
        <CommentsList comments={comments} topics={topics} />
      )}

      {tab === 'toxicity' && (
        <ToxicityPanel comments={comments} />
      )}

      {tab === 'entities' && (
        <EntitiesPanel jobId={jobId!} />
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
