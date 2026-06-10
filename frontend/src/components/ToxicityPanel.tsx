import type { Comment } from '../api'

interface Props { comments: Comment[] }

const CATEGORIES = [
  { key: 'toxic',         label: 'Toxic',         color: '#EF4444' },
  { key: 'severe_toxic',  label: 'Severe toxic',  color: '#DC2626' },
  { key: 'obscene',       label: 'Obscene',       color: '#F97316' },
  { key: 'threat',        label: 'Threat',        color: '#8B5CF6' },
  { key: 'insult',        label: 'Insult',        color: '#EC4899' },
  { key: 'identity_hate', label: 'Identity hate', color: '#F43F5E' },
]

function safeParseScores(json?: string | null): Record<string, number> {
  try { return JSON.parse(json ?? '{}') }
  catch { return {} }
}

export default function ToxicityPanel({ comments }: Props) {
  const total = comments.length
  if (total === 0) return (
    <p className="text-gray-600 font-mono text-sm">No comments to analyze.</p>
  )

  const toxicCount = comments.filter(c => c.is_toxic).length
  const toxicPct   = Math.round((toxicCount / total) * 100)

  // Mean score per category across ALL comments
  const avgScores: Record<string, number> = {}
  for (const cat of CATEGORIES) {
    const sum = comments.reduce((acc, c) => {
      const scores = safeParseScores(c.toxicity_json)
      return acc + (scores[cat.key] ?? 0)
    }, 0)
    avgScores[cat.key] = sum / total
  }

  // Max avg for relative bar scaling
  const maxAvg = Math.max(...Object.values(avgScores), 0.01)

  return (
    <div className="space-y-4 max-w-xl mx-auto">

      {/* ── Overall stat card ──────────────────────────────────────── */}
      <div className="card flex items-center justify-between">
        <div>
          <p className="label">Flagged as toxic</p>
          <p className="font-display text-4xl font-bold text-tox mt-1">{toxicPct}%</p>
        </div>
        <div className="text-right">
          <p className="font-mono text-sm text-gray-400">{toxicCount} comments</p>
          <p className="font-mono text-xs text-gray-600">out of {total}</p>
        </div>
      </div>

      {/* ── Per-category breakdown ─────────────────────────────────── */}
      <div className="card space-y-4">
        <p className="label">Average score per category</p>
        <p className="text-xs text-gray-600 font-body -mt-2">
          Mean ToxicBERT confidence across all {total} comments (0 = clean, 1 = certain)
        </p>

        {CATEGORIES.map(cat => {
          const avg  = avgScores[cat.key]
          const pct  = (avg / maxAvg) * 100   // relative to highest category

          return (
            <div key={cat.key}>
              <div className="flex justify-between items-center mb-1.5">
                <span className="text-sm font-body text-gray-300">{cat.label}</span>
                <span className="font-mono text-xs" style={{ color: cat.color }}>
                  {avg.toFixed(4)}
                </span>
              </div>
              <div className="h-1.5 bg-base rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-700 ease-out"
                  style={{ width: `${pct}%`, background: cat.color }}
                />
              </div>
            </div>
          )
        })}
      </div>

      {/* ── Note ───────────────────────────────────────────────────── */}
      <p className="text-xs text-gray-700 font-mono">
        Model: unitary/toxic-bert · threshold: 0.5
      </p>
    </div>
  )
}
