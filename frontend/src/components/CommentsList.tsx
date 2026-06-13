import { useState } from 'react'
import type { Comment, Topic } from '../api'

interface Props {
  comments: Comment[]
  topics:   Topic[]
}

const PAGE_SIZE = 20

const SENT_PILL: Record<string, string> = {
  positive: 'pill-pos',
  negative: 'pill-neg',
  neutral:  'pill-neu',
}

export default function CommentsList({ comments, topics }: Props) {
  const [page,        setPage]        = useState(0)
  const [sentFilter,  setSentFilter]  = useState('all')
  const [toxicOnly,   setToxicOnly]   = useState(false)
  const [repliesOnly, setRepliesOnly] = useState(false)
  const [search,      setSearch]      = useState('')

  const topicLabel = Object.fromEntries(
    topics.map(t => [t.topic_id, t.label.split(' | ')[0]])
  )

  const topLevelCount = comments.filter(c => !c.parent_id).length
  const replyCount    = comments.filter(c => !!c.parent_id).length

  const filtered = comments.filter(c => {
    if (sentFilter !== 'all' && c.sentiment_label !== sentFilter) return false
    if (toxicOnly   && !c.is_toxic)  return false
    if (repliesOnly && !c.parent_id) return false
    if (search && !c.original_text.toLowerCase().includes(search.toLowerCase())) return false
    return true
  })

  const totalPages = Math.ceil(filtered.length / PAGE_SIZE)
  const visible    = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)

  function resetPage() { setPage(0) }

  return (
    <div className="space-y-4">

      {/* ── Filters ─────────────────────────────────────────────────────── */}
      <div className="flex flex-wrap gap-2 items-center">

        <input
          className="input-field flex-1 min-w-[180px] py-2 text-xs"
          placeholder="Search comments…"
          value={search}
          onChange={e => { setSearch(e.target.value); resetPage() }}
        />

        {/* Sentiment pills */}
        <div className="flex gap-1.5">
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
        </div>

        {/* Toxic toggle */}
        <button
          onClick={() => { setToxicOnly(v => !v); resetPage() }}
          className={`px-3 py-1.5 rounded-lg text-xs font-mono border transition-all
            ${toxicOnly
              ? 'border-tox/60 text-tox bg-tox/8'
              : 'border-base-border text-gray-500 hover:text-gray-300'}`}
        >
          🚨 toxic only
        </button>

        {/* Replies toggle — only shown if there are replies */}
        {replyCount > 0 && (
          <button
            onClick={() => { setRepliesOnly(v => !v); resetPage() }}
            className={`px-3 py-1.5 rounded-lg text-xs font-mono border transition-all
              ${repliesOnly
                ? 'border-amber/60 text-amber bg-amber/8'
                : 'border-base-border text-gray-500 hover:text-gray-300'}`}
          >
            ↩ replies only
          </button>
        )}
      </div>

      {/* Count */}
      <div className="flex items-center gap-3">
        <p className="label">{filtered.length} comments</p>
        {replyCount > 0 && (
          <p className="text-xs font-mono text-gray-600">
            {topLevelCount} top-level · {replyCount} replies
          </p>
        )}
      </div>

      {/* ── Comment rows ─────────────────────────────────────────────────── */}
      <div className="space-y-2">
        {visible.length === 0 && (
          <p className="text-gray-600 font-mono text-sm py-8 text-center">
            No comments match the current filters.
          </p>
        )}

        {visible.map(c => (
          <div
            key={c.id}
            className={`card py-3 space-y-2 ${
              c.parent_id ? 'ml-4 border-l-2 border-l-amber/30' : ''
            }`}
          >
            {/* Comment text */}
            <p className="text-sm text-gray-200 leading-relaxed">{c.original_text}</p>

            {/* Meta badges */}
            <div className="flex flex-wrap gap-2 items-center">

              {/* Reply badge */}
              {c.parent_id && (
                <span className="font-mono text-xs text-amber/70 bg-amber/5
                                 border border-amber/20 px-2 py-0.5 rounded-full">
                  ↩ reply
                </span>
              )}

              {c.sentiment_label && (
                <span className={SENT_PILL[c.sentiment_label] ?? 'pill-neu'}>
                  {c.sentiment_label}
                </span>
              )}

              {c.vader_compound !== undefined && c.vader_compound !== null && (
                <span className="font-mono text-xs text-gray-600">
                  VADER {c.vader_compound > 0 ? '+' : ''}{c.vader_compound.toFixed(3)}
                </span>
              )}

              {c.topic_id !== undefined && c.topic_id !== null && c.topic_id !== -1
                && topicLabel[c.topic_id] && (
                <span className="font-mono text-xs text-amber/70 bg-amber/5
                                 border border-amber/15 px-2 py-0.5 rounded-full">
                  {topicLabel[c.topic_id]}
                </span>
              )}

              {!!c.is_toxic && (
                <span className="font-mono text-xs text-tox bg-tox/5
                                 border border-tox/20 px-2 py-0.5 rounded-full">
                  🚨 toxic
                </span>
              )}

            </div>
          </div>
        ))}
      </div>

      {/* ── Pagination ───────────────────────────────────────────────────── */}
      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-3 pt-2">
          <button
            onClick={() => setPage(p => p - 1)}
            disabled={page === 0}
            className="px-4 py-1.5 text-sm font-mono border border-base-border rounded-lg
                       disabled:opacity-30 hover:border-amber/40 transition-colors"
          >
            ← prev
          </button>

          <span className="text-xs font-mono text-gray-500">
            {page + 1} / {totalPages}
          </span>

          <button
            onClick={() => setPage(p => p + 1)}
            disabled={page >= totalPages - 1}
            className="px-4 py-1.5 text-sm font-mono border border-base-border rounded-lg
                       disabled:opacity-30 hover:border-amber/40 transition-colors"
          >
            next →
          </button>
        </div>
      )}
    </div>
  )
}
