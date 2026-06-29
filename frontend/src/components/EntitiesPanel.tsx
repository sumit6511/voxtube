import { useState } from 'react'
import { Loader2, Search, AlertTriangle } from 'lucide-react'
import { api } from '../api'
import type { Entity, NerResponse } from '../api'

interface Props { jobId: string }

const CATEGORY_CONFIG: Record<string, { color: string; bg: string; emoji: string }> = {
  Person:        { color: 'text-pos',    bg: 'bg-pos/10',    emoji: '👤' },
  Organization:  { color: 'text-amber',  bg: 'bg-amber/10',  emoji: '🏢' },
  Location:      { color: 'text-[#378ADD]', bg: 'bg-blue-500/10', emoji: '📍' },
  Miscellaneous: { color: 'text-gray-400', bg: 'bg-gray-700/30',  emoji: '🔖' },
}

const SENT_COLOR: Record<string, string> = {
  positive: 'text-pos',
  negative: 'text-neg',
  neutral:  'text-gray-500',
}

function EntityBubble({ entity }: { entity: Entity }) {
  const cfg = CATEGORY_CONFIG[entity.category] ?? CATEGORY_CONFIG.Miscellaneous
  return (
    <div className={`flex items-center gap-2 px-3 py-2 rounded-lg border
                     border-base-border ${cfg.bg} group`}>
      <span className="text-base">{cfg.emoji}</span>
      <div className="min-w-0">
        <p className={`text-sm font-body font-medium ${cfg.color} truncate`}>
          {entity.text}
        </p>
        <p className="text-xs font-mono text-gray-600">
          {entity.category} ·{' '}
          <span className={SENT_COLOR[entity.sentiment] ?? 'text-gray-500'}>
            {entity.sentiment}
          </span>
        </p>
      </div>
      <span className="ml-auto font-mono text-xs text-gray-600 flex-shrink-0">
        ×{entity.count}
      </span>
    </div>
  )
}

export default function EntitiesPanel({ jobId }: Props) {
  const [data,       setData]       = useState<NerResponse | null>(null)
  const [loading,    setLoading]    = useState(false)
  const [error,      setError]      = useState('')
  const [filter,     setFilter]     = useState<string>('All')

  function run() {
    setLoading(true)
    setError('')
    api.ner(jobId)
      .then(d => { setData(d); setLoading(false) })
      .catch(e => { setError(e.message); setLoading(false) })
  }

  // Not yet run
  if (!data && !loading && !error) {
    return (
      <div className="card flex flex-col items-center justify-center py-12 gap-4">
        <Search size={28} className="text-gray-700" />
        <div className="text-center">
          <p className="text-sm font-body text-gray-400">
            Extract people, organisations, and locations mentioned in comments
          </p>
          <p className="text-xs font-mono text-gray-700 mt-1">
            Uses dslim/bert-base-NER · runs on Latin-script comments only
          </p>
        </div>
        <button onClick={run} className="btn-primary text-sm px-5 py-2.5">
          Run Entity Extraction
        </button>
      </div>
    )
  }

  if (loading) return (
    <div className="card flex items-center justify-center py-16 gap-3">
      <Loader2 size={20} className="animate-spin text-amber" />
      <p className="font-mono text-sm text-gray-500">
        Extracting entities… (first run downloads model ~400MB)
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

  if (!data) return null

  const { entities, total_processed, total_skipped, model_available } = data
  const categories = ['All', ...Array.from(new Set(entities.map(e => e.category)))]
  const filtered   = filter === 'All' ? entities : entities.filter(e => e.category === filter)

  return (
    <div className="space-y-4">

      {/* Stats row */}
      <div className="flex flex-wrap gap-4 items-center">
        <div className="card py-2 px-4 flex gap-6">
          <div className="text-center">
            <p className="font-display text-xl font-bold text-white">{entities.length}</p>
            <p className="label">entities</p>
          </div>
          <div className="text-center">
            <p className="font-display text-xl font-bold text-amber">{total_processed}</p>
            <p className="label">processed</p>
          </div>
          {total_skipped > 0 && (
            <div className="text-center">
              <p className="font-display text-xl font-bold text-gray-500">{total_skipped}</p>
              <p className="label">skipped (Devanagari)</p>
            </div>
          )}
        </div>

        {!model_available && (
          <div className="flex items-center gap-2 text-xs font-mono text-amber/80
                          bg-amber/5 border border-amber/20 rounded-lg px-3 py-2">
            <AlertTriangle size={12} />
            Model unavailable — install torch and internet access required
          </div>
        )}

        <button onClick={run}
          className="ml-auto text-xs font-mono text-gray-500 hover:text-white
                     border border-base-border px-3 py-1.5 rounded-lg transition-colors">
          Re-run
        </button>
      </div>

      {/* Category filter */}
      <div className="flex gap-2 flex-wrap">
        {categories.map(cat => {
          const cfg = CATEGORY_CONFIG[cat]
          return (
            <button key={cat} onClick={() => setFilter(cat)}
              className={`text-xs font-mono px-3 py-1.5 rounded-lg border transition-all
                ${filter === cat
                  ? 'bg-amber text-gray-950 border-amber font-medium'
                  : 'border-base-border text-gray-500 hover:text-gray-300'}`}>
              {cfg?.emoji} {cat}
            </button>
          )
        })}
      </div>

      {/* Entity grid */}
      {filtered.length === 0 ? (
        <p className="text-gray-600 font-mono text-sm py-8 text-center">
          No entities found with ≥2 mentions.
        </p>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
          {filtered.map((e, i) => <EntityBubble key={i} entity={e} />)}
        </div>
      )}

      <p className="text-xs font-mono text-gray-700">
        Model: dslim/bert-base-NER · CoNLL-2003 · confidence threshold 0.80 · min 2 mentions
      </p>
    </div>
  )
}
