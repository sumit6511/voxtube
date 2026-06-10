import { useMemo } from 'react'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer,
} from 'recharts'
import type { Comment } from '../api'

interface Props { comments: Comment[] }
type Granularity = 'day' | 'week' | 'month'

interface TimePoint {
  label:    string
  positive: number
  neutral:  number
  negative: number
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function getKey(d: Date, g: Granularity): string {
  if (g === 'day')   return d.toISOString().slice(0, 10)
  if (g === 'month') return d.toISOString().slice(0, 7)
  // week → Monday of that week
  const day  = d.getDay()
  const diff = d.getDate() - day + (day === 0 ? -6 : 1)
  const mon  = new Date(d); mon.setDate(diff)
  return mon.toISOString().slice(0, 10)
}

function formatLabel(key: string, g: Granularity): string {
  const d = new Date(key + (key.length === 7 ? '-01' : ''))
  if (g === 'month')
    return d.toLocaleDateString('en-US', { month: 'short', year: '2-digit' })
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
}

function buildTimeline(comments: Comment[]): {
  points: TimePoint[]; granularity: Granularity; withDates: number
} {
  const dated = comments.filter(c => !!c.published_at)

  if (dated.length === 0)
    return { points: [], granularity: 'day', withDates: 0 }

  const ts  = dated.map(c => new Date(c.published_at!).getTime())
  const span = (Math.max(...ts) - Math.min(...ts)) / 86_400_000   // days

  const g: Granularity =
    span <= 14  ? 'day'   :
    span <= 120 ? 'week'  : 'month'

  const groups: Record<string, TimePoint> = {}

  for (const c of dated) {
    const key = getKey(new Date(c.published_at!), g)
    if (!groups[key])
      groups[key] = { label: formatLabel(key, g), positive: 0, neutral: 0, negative: 0 }

    const s = c.sentiment_label ?? 'neutral'
    if (s === 'positive' || s === 'negative') groups[key][s]++
    else groups[key].neutral++
  }

  const points = Object.entries(groups)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([, pt]) => pt)

  return { points, granularity: g, withDates: dated.length }
}

// ── Tooltip ───────────────────────────────────────────────────────────────────

const TOOLTIP_STYLE = {
  background: '#13161E', border: '1px solid #1E2330',
  borderRadius: '8px', fontSize: '12px', fontFamily: 'IBM Plex Mono',
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function SentimentTimeline({ comments }: Props) {
  const { points, granularity, withDates } = useMemo(
    () => buildTimeline(comments), [comments]
  )

  if (withDates === 0) {
    return (
      <div>
        <p className="label mb-3">Sentiment over time</p>
        <div className="h-44 flex items-center justify-center">
          <p className="text-gray-600 font-mono text-sm text-center px-4">
            No timestamp data — re-analyze the video to enable this chart.
          </p>
        </div>
      </div>
    )
  }

  const periodLabel =
    granularity === 'day' ? 'daily' : granularity === 'week' ? 'weekly' : 'monthly'

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <p className="label">Sentiment over time</p>
        <p className="font-mono text-xs text-gray-600">
          {periodLabel} · {points.length} periods · {withDates} comments
        </p>
      </div>

      <ResponsiveContainer width="100%" height={220}>
        <AreaChart data={points} margin={{ top: 4, right: 8, left: -22, bottom: 0 }}>

          <defs>
            <linearGradient id="posGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%"  stopColor="#10B981" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#10B981" stopOpacity={0}   />
            </linearGradient>
            <linearGradient id="neuGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%"  stopColor="#6B7280" stopOpacity={0.2} />
              <stop offset="95%" stopColor="#6B7280" stopOpacity={0}   />
            </linearGradient>
            <linearGradient id="negGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%"  stopColor="#F43F5E" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#F43F5E" stopOpacity={0}   />
            </linearGradient>
          </defs>

          <CartesianGrid strokeDasharray="3 3" stroke="#1E2330" vertical={false} />

          <XAxis
            dataKey="label"
            tick={{ fill: '#6B7280', fontSize: 11, fontFamily: 'IBM Plex Mono' }}
            axisLine={false} tickLine={false}
            interval="preserveStartEnd"
          />

          <YAxis
            tick={{ fill: '#6B7280', fontSize: 11 }}
            axisLine={false} tickLine={false}
            allowDecimals={false}
          />

          <Tooltip
            contentStyle={TOOLTIP_STYLE}
            cursor={{ stroke: '#2D3446' }}
          />

          <Legend
            iconType="circle" iconSize={8}
            formatter={v => (
              <span style={{ fontSize: '12px', color: '#9CA3AF', fontFamily: 'IBM Plex Mono' }}>
                {v}
              </span>
            )}
          />

          <Area
            type="monotone" dataKey="positive" name="Positive"
            stroke="#10B981" strokeWidth={2} fill="url(#posGrad)"
            dot={false} activeDot={{ r: 4, fill: '#10B981' }}
          />
          <Area
            type="monotone" dataKey="neutral" name="Neutral"
            stroke="#6B7280" strokeWidth={2} fill="url(#neuGrad)"
            dot={false} activeDot={{ r: 4, fill: '#6B7280' }}
          />
          <Area
            type="monotone" dataKey="negative" name="Negative"
            stroke="#F43F5E" strokeWidth={2} fill="url(#negGrad)"
            dot={false} activeDot={{ r: 4, fill: '#F43F5E' }}
          />

        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}
