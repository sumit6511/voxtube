import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from 'recharts'

interface Props {
  data: { positive: number; neutral: number; negative: number }
}

const SLICES = [
  { key: 'positive', label: 'Positive', color: '#10B981' },
  { key: 'neutral',  label: 'Neutral',  color: '#6B7280' },
  { key: 'negative', label: 'Negative', color: '#F43F5E' },
]

const TOOLTIP_STYLE = {
  background: '#13161E', border: '1px solid #1E2330',
  borderRadius: '8px', fontSize: '12px', fontFamily: 'IBM Plex Mono',
}

export default function SentimentChart({ data }: Props) {
  const total = data.positive + data.neutral + data.negative

  const chartData = SLICES
    .map(s => ({ name: s.label, value: data[s.key as keyof typeof data], color: s.color }))
    .filter(d => d.value > 0)

  return (
    <div>
      <p className="label mb-3">Sentiment distribution</p>

      <ResponsiveContainer width="100%" height={200}>
        <PieChart>
          <Pie
            data={chartData}
            cx="50%" cy="50%"
            innerRadius={52} outerRadius={78}
            paddingAngle={3}
            dataKey="value"
          >
            {chartData.map((entry, i) => (
              <Cell key={i} fill={entry.color} strokeWidth={0} />
            ))}
          </Pie>

          <Tooltip
            contentStyle={TOOLTIP_STYLE}
            formatter={(value) => {
              const n = Number(value)
              return [`${n}  (${total ? Math.round((n / total) * 100) : 0}%)`, '']
            }}
          />

          <Legend
            iconType="circle"
            iconSize={8}
            formatter={v => (
              <span style={{ fontSize: '12px', color: '#9CA3AF', fontFamily: 'IBM Plex Mono' }}>
                {v}
              </span>
            )}
          />
        </PieChart>
      </ResponsiveContainer>

      {/* Raw counts row */}
      <div className="grid grid-cols-3 gap-2 mt-1">
        {SLICES.map(s => (
          <div key={s.key} className="text-center">
            <div className="font-mono text-xl font-medium" style={{ color: s.color }}>
              {data[s.key as keyof typeof data]}
            </div>
            <div className="text-xs text-gray-600">{s.label}</div>
          </div>
        ))}
      </div>
    </div>
  )
}
