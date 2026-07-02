import { useState, useMemo } from 'react'
import { ArrowUpDown } from 'lucide-react'
import type { Comment, Topic } from '../api'
import Dropdown from './Dropdown'

interface Props { comments: Comment[]; topics: Topic[] }

const PAGE_SIZE = 20

const SENT_PILL: Record<string, string> = {
  positive: 'pill-pos',
  negative: 'pill-neg',
  neutral:  'pill-neu',
}

type SortKey = 'default' | 'xlm_desc' | 'xlm_asc' | 'vader_desc' | 'vader_asc' | 'date_desc' | 'date_asc'

const SORT_OPTIONS: { value: SortKey; label: string }[] = [
  { value: 'default',   label: 'Default'            },
  { value: 'xlm_desc',  label: 'XLM score ↓'       },
  { value: 'xlm_asc',   label: 'XLM score ↑'       },
  { value: 'vader_desc',label: 'VADER score ↓'      },
  { value: 'vader_asc', label: 'VADER score ↑'      },
  { value: 'date_desc', label: 'Date (newest)'      },
  { value: 'date_asc',  label: 'Date (oldest)'      },
]

function sortComments(comments: Comment[], key: SortKey): Comment[] {
  if (key === 'default') return comments
  return [...comments].sort((a, b) => {
    switch (key) {
      case 'xlm_desc':   return (b.sentiment_score ?? 0) - (a.sentiment_score ?? 0)
      case 'xlm_asc':    return (a.sentiment_score ?? 0) - (b.sentiment_score ?? 0)
      case 'vader_desc': return (b.vader_compound ?? 0) - (a.vader_compound ?? 0)
      case 'vader_asc':  return (a.vader_compound ?? 0) - (b.vader_compound ?? 0)
      case 'date_desc':  return new Date(b.published_at ?? 0).getTime() - new Date(a.published_at ?? 0).getTime()
      case 'date_asc':   return new Date(a.published_at ?? 0).getTime() - new Date(b.published_at ?? 0).getTime()
      default:           return 0
    }
  })
}

export default function CommentsList({ comments, topics }: Props) {
  const [page,        setPage]        = useState(0)
  const [sentFilter,  setSentFilter]  = useState('all')
  const [toxicOnly,   setToxicOnly]   = useState(false)
  const [search,      setSearch]      = useState('')
  const [sortKey,     setSortKey]     = useState<SortKey>('default')

  const topicLabel = Object.fromEntries(
    topics.map(t => [t.topic_id, t.label.split(' | ')[0]])
  )

  const filtered = useMemo(() => {
    const f = comments.filter(c => {
      if (sentFilter !== 'all' && c.sentiment_label !== sentFilter) return false
      if (toxicOnly && !c.is_toxic) return false
      if (search && !c.original_text.toLowerCase().includes(search.toLowerCase())) return false
      return true
    })
    return sortComments(f, sortKey)
  }, [comments, sentFilter, toxicOnly, search, sortKey])

  const totalPages = Math.ceil(filtered.length / PAGE_SIZE)
  const visible    = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)

  function resetPage() { setPage(0) }

  return (
    <div className="space-y-4">

      {/* ── Row 1: search + sort ─────────────────────────────────────── */}
      <div className="flex flex-wrap gap-2">
        <input
          className="input-field flex-1 min-w-[160px] py-2 text-xs"
          placeholder="Search comments…"
          value={search}
          onChange={e => { setSearch(e.target.value); resetPage() }}
        />

        {/* Sort dropdown */}
        <Dropdown
          value={sortKey}
          onChange={v => { setSortKey(v as SortKey); resetPage() }}
          options={SORT_OPTIONS}
          icon={<ArrowUpDown size={12} className="text-gray-500 flex-shrink-0" />}
        />
      </div>

      {/* ── Row 2: sentiment + toxic filters ────────────────────────── */}
      <div className="flex flex-wrap gap-2">
        {['all', 'positive', 'neutral', 'negative'].map(s => (
          <button
            key={s}
            onClick={() => { setSentFilter(s); resetPage() }}
            className={`px-3 py-1.5 rounded-lg text-xs font-mono capitalize transition-all
              ${sentFilter === s
                ? 'bg-amber text-gray-950 font-medium'
                : 'border border-base-border text-gray-500 hover:text-gray-300'}`}
          >
            {s}
          </button>
        ))}

        <button
          onClick={() => { setToxicOnly(v => !v); resetPage() }}
          className={`px-3 py-1.5 rounded-lg text-xs font-mono border transition-all
            ${toxicOnly
              ? 'border-tox/60 text-tox bg-tox/8'
              : 'border-base-border text-gray-500 hover:text-gray-300'}`}
        >
          🚨 toxic only
        </button>
      </div>

      {/* Count */}
      <p className="label">{filtered.length} comments</p>

      {/* ── Comment rows ─────────────────────────────────────────────── */}
      <div className="space-y-2">
        {visible.length === 0 && (
          <p className="text-gray-600 font-mono text-sm py-8 text-center">
            No comments match the current filters.
          </p>
        )}

        {visible.map(c => (
          <div key={c.id} className="card py-3 space-y-2">
            <p className="text-sm leading-relaxed" style={{ color: 'var(--color-text)' }}>
              {c.original_text}
            </p>

            <div className="flex flex-wrap gap-2 items-center">
              {c.sentiment_label && (
                <span className={SENT_PILL[c.sentiment_label] ?? 'pill-neu'}>
                  {c.sentiment_label}
                </span>
              )}

              {c.sentiment_score != null && (
                <span className="font-mono text-xs text-gray-600">
                  XLM {(c.sentiment_score * 100).toFixed(0)}%
                </span>
              )}

              {c.vader_compound != null && (
                <span className="font-mono text-xs text-gray-600">
                  VADER {c.vader_compound > 0 ? '+' : ''}{c.vader_compound.toFixed(3)}
                </span>
              )}

              {c.topic_id != null && c.topic_id !== -1 && topicLabel[c.topic_id] && (
                <span className="font-mono text-xs text-amber/70 bg-amber/5
                                 border border-amber/15 px-2 py-0.5 rounded-full">
                  {topicLabel[c.topic_id]}
                </span>
              )}

              {c.lang && (
                <span className="font-mono text-xs text-gray-700">
                  {c.lang}
                </span>
              )}

              {!!c.is_toxic && (
                <span className="font-mono text-xs text-tox bg-tox/5
                                 border border-tox/20 px-2 py-0.5 rounded-full">
                  🚨 toxic
                </span>
              )}

              {c.published_at && (
                <span className="font-mono text-xs text-gray-700 ml-auto">
                  {new Date(c.published_at).toLocaleDateString('en-US', {
                    month: 'short', day: 'numeric'
                  })}
                </span>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* ── Pagination ───────────────────────────────────────────────── */}
      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-3 pt-2">
          <button
            onClick={() => setPage(p => p - 1)} disabled={page === 0}
            className="px-4 py-1.5 text-sm font-mono border border-base-border rounded-lg
                       disabled:opacity-30 hover:border-amber/40 transition-colors">
            ← prev
          </button>
          <span className="text-xs font-mono text-gray-500">{page + 1} / {totalPages}</span>
          <button
            onClick={() => setPage(p => p + 1)} disabled={page >= totalPages - 1}
            className="px-4 py-1.5 text-sm font-mono border border-base-border rounded-lg
                       disabled:opacity-30 hover:border-amber/40 transition-colors">
            next →
          </button>
        </div>
      )}
    </div>
  )
}
