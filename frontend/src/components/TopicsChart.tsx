import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer,
} from 'recharts'
import type { Topic } from '../api'

interface Props { topics: Topic[] }

const TOOLTIP_STYLE = {
  background: '#13161E', border: '1px solid #1E2330',
  borderRadius: '8px', fontSize: '12px', fontFamily: 'IBM Plex Mono',
}

export default function TopicsChart({ topics }: Props) {
  const data = [...topics]
    .sort((a, b) => b.comment_count - a.comment_count)
    .slice(0, 8)
    .map(t => ({
      name:     t.label.split(' | ')[0],   // first keyword only for axis label
      positive: t.positive_count,
      neutral:  t.neutral_count,
      negative: t.negative_count,
    }))

  return (
    <div>
      <p className="label mb-3">Per-topic sentiment</p>

      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} margin={{ top: 0, right: 0, left: -22, bottom: 40 }}>

          <CartesianGrid strokeDasharray="3 3" stroke="#1E2330" vertical={false} />

          <XAxis
            dataKey="name"
            tick={{ fill: '#6B7280', fontSize: 11, fontFamily: 'IBM Plex Mono' }}
            axisLine={false} tickLine={false}
            interval={0} angle={-25} textAnchor="end"
          />

          <YAxis
            tick={{ fill: '#6B7280', fontSize: 11 }}
            axisLine={false} tickLine={false}
            allowDecimals={false}
          />

          <Tooltip
            contentStyle={TOOLTIP_STYLE}
            cursor={{ fill: 'rgba(255,255,255,0.03)' }}
          />

          <Legend
            iconType="circle" iconSize={8}
            formatter={v => (
              <span style={{ fontSize: '12px', color: '#9CA3AF', fontFamily: 'IBM Plex Mono' }}>
                {v}
              </span>
            )}
          />

          <Bar dataKey="positive" stackId="s" fill="#10B981" name="Positive" />
          <Bar dataKey="neutral"  stackId="s" fill="#6B7280" name="Neutral"  />
          <Bar dataKey="negative" stackId="s" fill="#F43F5E" name="Negative"
               radius={[4, 4, 0, 0]} />

        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
