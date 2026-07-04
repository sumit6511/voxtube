import { useState, useMemo } from 'react'
import {
  ScatterChart, Scatter, XAxis, YAxis, ZAxis,
  Tooltip, Legend, ResponsiveContainer, Cell,
} from 'recharts'
import { Loader2, AlertTriangle, X } from 'lucide-react'
import { api } from '../api'
import type { ScatterPoint, Comment, Topic } from '../api'

interface Props {
  jobId: string
  comments: Comment[]
  topics: Topic[]
}

const SENT_COLOR: Record<string, string> = {
  positive: '#10B981',
  neutral:  '#6B7280',
  negative: '#F43F5E',
}

type ColorBy = 'sentiment' | 'language' | 'toxicity'

const LANG_COLOR: Record<string, string> = {
  nepali:  '#10B981',
  english: '#378ADD',
  neplish: '#F59E0B',
}

const SENT_PILL: Record<string, string> = {
  positive: 'pill-pos',
  negative: 'pill-neg',
  neutral:  'pill-neu',
}

function pointColor(p: ScatterPoint, colorBy: ColorBy): string {
  if (colorBy === 'sentiment') return SENT_COLOR[p.sentiment] ?? '#6B7280'
  if (colorBy === 'language')  return LANG_COLOR[p.lang]      ?? '#6B7280'
  return p.is_toxic ? '#EF4444' : '#6B7280'
}

function CustomTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null
  const p: ScatterPoint = payload[0]?.payload
  if (!p) return null
  return (
    <div className="bg-base-surface border border-base-border rounded-xl p-3
                    max-w-[240px] shadow-xl">
      <p className="text-xs font-body text-gray-300 leading-relaxed mb-2">
        {p.text}{p.text.length >= 70 ? '…' : ''}
      </p>
      <div className="flex flex-wrap gap-1.5">
        <span className="text-xs font-mono px-1.5 py-0.5 rounded"
              style={{ color: SENT_COLOR[p.sentiment], background: `${SENT_COLOR[p.sentiment]}18` }}>
          {p.sentiment}
        </span>
        <span className="text-xs font-mono text-gray-500">{p.lang}</span>
        {p.is_toxic === 1 && (
          <span className="text-xs font-mono text-tox bg-tox/10 px-1.5 py-0.5 rounded">
            🚨 toxic
          </span>
        )}
      </div>
      <p className="text-xs font-mono text-gray-700 mt-2">Click for the full comment</p>
    </div>
  )
}

// ── Detail modal shown when a point is clicked ────────────────────────────────
function CommentDetailModal({
  point, comments, topics, onClose,
}: {
  point: ScatterPoint
  comments: Comment[]
  topics: Topic[]
  onClose: () => void
}) {
  // The scatter payload only carries a 70-char text snippet (to keep the
  // plot's payload small) — look up the matching Comment from data already
  // in memory to show the full, untruncated text and richer annotations.
  const full = comments.find(c => c.id === point.id)
  const topicLabel = full?.topic_id != null && full.topic_id !== -1
    ? topics.find(t => t.topic_id === full.topic_id)?.label.split(' | ')[0]
    : undefined

  const sentiment = full?.sentiment_label ?? point.sentiment
  const isToxic   = full ? !!full.is_toxic : point.is_toxic === 1

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: 'rgba(0,0,0,0.6)' }}
      onClick={onClose}
    >
      <div
        className="card max-w-md w-full space-y-3"
        style={{ backgroundColor: 'var(--color-surface)' }}
        onClick={e => e.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-3">
          <p className="label">Comment detail</p>
          <button onClick={onClose} className="text-gray-500 hover:text-white transition-colors flex-shrink-0">
            <X size={16} />
          </button>
        </div>

        <p className="text-sm leading-relaxed" style={{ color: 'var(--color-text)' }}>
          {full?.original_text ?? point.text}
        </p>

        {!full && (
          <p className="text-xs font-mono text-gray-700">
            (Showing truncated text — full comment unavailable)
          </p>
        )}

        <div className="flex flex-wrap gap-2 items-center pt-1">
          {sentiment && (
            <span className={SENT_PILL[sentiment] ?? 'pill-neu'}>{sentiment}</span>
          )}
          {full?.sentiment_score != null && (
            <span className="font-mono text-xs text-gray-600">
              XLM {(full.sentiment_score * 100).toFixed(0)}%
            </span>
          )}
          {full?.vader_compound != null && (
            <span className="font-mono text-xs text-gray-600">
              VADER {full.vader_compound > 0 ? '+' : ''}{full.vader_compound.toFixed(3)}
            </span>
          )}
          <span className="font-mono text-xs text-gray-700">{full?.lang ?? point.lang}</span>
          {topicLabel && (
            <span className="font-mono text-xs text-amber/70 bg-amber/5
                             border border-amber/15 px-2 py-0.5 rounded-full">
              {topicLabel}
            </span>
          )}
          {isToxic && (
            <span className="font-mono text-xs text-tox bg-tox/5
                             border border-tox/20 px-2 py-0.5 rounded-full">
              🚨 toxic
            </span>
          )}
        </div>
      </div>
    </div>
  )
}

export default function ScatterPlot({ jobId, comments, topics }: Props) {
  const [points,  setPoints]  = useState<ScatterPoint[] | null>(null)
  const [method,  setMethod]  = useState('')
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState('')
  const [colorBy, setColorBy] = useState<ColorBy>('sentiment')
  const [selectedPoint, setSelectedPoint] = useState<ScatterPoint | null>(null)

  function run() {
    setLoading(true); setError('')
    api.umap(jobId)
      .then(d => { setPoints(d.points); setMethod(d.method); setLoading(false) })
      .catch(e => { setError(e.message); setLoading(false) })
  }

  // Group points by color for Scatter series (Recharts needs one <Scatter> per color)
  const coloredPoints = useMemo(() => {
    if (!points) return []
    return points.map(p => ({ ...p, _color: pointColor(p, colorBy) }))
  }, [points, colorBy])

  // Recharts hands back the merged element props on click — the source
  // datum lives at .payload, but fall back to the object itself just in case.
  function handlePointClick(clicked: any) {
    const p = clicked?.payload ?? clicked
    if (p?.id) setSelectedPoint(p)
  }

  if (!points && !loading && !error) return (
    <div className="card flex flex-col items-center justify-center py-14 gap-4">
      <div className="w-16 h-16 rounded-full bg-amber/10 border border-amber/20
                      flex items-center justify-center">
        <span className="text-2xl">✦</span>
      </div>
      <div className="text-center">
        <p className="text-sm font-body text-gray-400">
          Project comment embeddings into 2D space
        </p>
        <p className="text-xs font-mono text-gray-700 mt-1">
          Uses UMAP (or PCA fallback) · coloured by sentiment, language, or toxicity
        </p>
      </div>
      <button onClick={run} className="btn-primary text-sm px-5 py-2.5">
        Generate Scatter Plot
      </button>
    </div>
  )

  if (loading) return (
    <div className="card flex items-center justify-center py-16 gap-3">
      <Loader2 size={20} className="animate-spin text-amber" />
      <p className="font-mono text-sm text-gray-500">
        Computing 2D projection… (first run may take ~30s)
      </p>
    </div>
  )

  if (error) return (
    <div className="card flex items-center gap-3 py-6">
      <AlertTriangle size={18} className="text-neg flex-shrink-0" />
      <p className="text-neg font-mono text-sm">{error}</p>
      <button onClick={run} className="ml-auto text-xs font-mono text-gray-500
                                       hover:text-white border border-base-border
                                       px-3 py-1.5 rounded-lg transition-colors">
        Retry
      </button>
    </div>
  )

  if (!points) return null

  return (
    <div className="space-y-4">

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex gap-1.5">
          {(['sentiment', 'language', 'toxicity'] as ColorBy[]).map(opt => (
            <button key={opt} onClick={() => setColorBy(opt)}
              className={`text-xs font-mono px-3 py-1.5 rounded-lg border capitalize transition-all
                ${colorBy === opt
                  ? 'bg-amber text-gray-950 border-amber font-medium'
                  : 'border-base-border text-gray-500 hover:text-gray-300'}`}>
              {opt}
            </button>
          ))}
        </div>
        <p className="ml-auto font-mono text-xs text-gray-600">
          {points.length} comments · {method.toUpperCase()}
        </p>
        <button onClick={run}
          className="text-xs font-mono text-gray-500 hover:text-white
                     border border-base-border px-3 py-1.5 rounded-lg transition-colors">
          Recompute
        </button>
      </div>

      {/* Chart */}
      <div className="card overflow-hidden p-2">
        <ResponsiveContainer width="100%" height={440}>
          <ScatterChart margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
            <XAxis
              type="number" dataKey="x" domain={[-1, 1]}
              tick={{ fontSize: 10, fill: '#6B7280', fontFamily: 'IBM Plex Mono' }}
              axisLine={false} tickLine={false} tickCount={5}
            />
            <YAxis
              type="number" dataKey="y" domain={[-1, 1]}
              tick={{ fontSize: 10, fill: '#6B7280', fontFamily: 'IBM Plex Mono' }}
              axisLine={false} tickLine={false} tickCount={5}
            />
            <ZAxis range={[18, 18]} />
            <Tooltip content={<CustomTooltip />} cursor={false} />

            <Scatter data={coloredPoints} isAnimationActive={false} onClick={handlePointClick}>
              {coloredPoints.map((p, i) => (
                <Cell
                  key={i}
                  fill={p._color}
                  fillOpacity={colorBy === 'toxicity' && p.is_toxic === 0 ? 0.25 : 0.75}
                  stroke={p.is_toxic === 1 && colorBy !== 'toxicity' ? '#EF4444' : 'none'}
                  strokeWidth={1.5}
                  style={{ cursor: 'pointer' }}
                />
              ))}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-3">
        {colorBy === 'sentiment' && Object.entries(SENT_COLOR).map(([k, v]) => (
          <span key={k} className="flex items-center gap-1.5 text-xs font-mono text-gray-400">
            <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ background: v }} />
            {k}
          </span>
        ))}
        {colorBy === 'language' && Object.entries(LANG_COLOR).map(([k, v]) => (
          <span key={k} className="flex items-center gap-1.5 text-xs font-mono text-gray-400">
            <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ background: v }} />
            {k}
          </span>
        ))}
        {colorBy === 'toxicity' && (
          <>
            <span className="flex items-center gap-1.5 text-xs font-mono text-gray-400">
              <span className="w-2.5 h-2.5 rounded-full inline-block bg-tox" /> Toxic
            </span>
            <span className="flex items-center gap-1.5 text-xs font-mono text-gray-400">
              <span className="w-2.5 h-2.5 rounded-full inline-block bg-gray-600" /> Clean
            </span>
          </>
        )}
        {colorBy !== 'toxicity' && (
          <span className="flex items-center gap-1.5 text-xs font-mono text-gray-600 ml-auto">
            Toxic comments have a red outline
          </span>
        )}
      </div>

      <p className="text-xs font-mono text-gray-700">
        Each dot = one comment · hover for a preview, click for the full comment · proximity = semantic similarity
      </p>

      {selectedPoint && (
        <CommentDetailModal
          point={selectedPoint}
          comments={comments}
          topics={topics}
          onClose={() => setSelectedPoint(null)}
        />
      )}
    </div>
  )
}
