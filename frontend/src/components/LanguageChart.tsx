import { useMemo } from 'react'
import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import type { Comment } from '../api'

interface Props { comments: Comment[] }

const LANG_CONFIG: Record<string, { label: string; color: string; desc: string }> = {
  nepali:  { label: 'Nepali',  color: '#10B981', desc: 'Devanagari script'      },
  english: { label: 'English', color: '#378ADD', desc: 'Latin script'           },
  neplish: { label: 'Neplish', color: '#F59E0B', desc: 'Code-mixed / Romanized' },
}

const TOOLTIP_STYLE = {
  background: '#13161E', border: '1px solid #1E2330',
  borderRadius: '8px', fontSize: '12px', fontFamily: 'IBM Plex Mono',
}

export default function LanguageChart({ comments }: Props) {
  const counts = useMemo(() => {
    const c = { nepali: 0, english: 0, neplish: 0 }
    for (const comment of comments) {
      const lang = (comment.lang ?? 'neplish') as keyof typeof c
      if (lang in c) c[lang]++
      else c.neplish++
    }
    return c
  }, [comments])

  const total = comments.length
  const chartData = Object.entries(LANG_CONFIG)
    .map(([key, cfg]) => ({
      name:  cfg.label,
      value: counts[key as keyof typeof counts],
      color: cfg.color,
      desc:  cfg.desc,
    }))
    .filter(d => d.value > 0)

  return (
    <div>
      <p className="label mb-3">Language breakdown</p>

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
        {Object.entries(LANG_CONFIG).map(([key, cfg]) => {
          const count = counts[key as keyof typeof counts]
          const pct   = total ? Math.round((count / total) * 100) : 0
          return (
            <div key={key} className="text-center">
              <div className="font-mono text-xl font-medium" style={{ color: cfg.color }}>
                {count}
              </div>
              <div className="text-xs text-gray-400">{cfg.label}</div>
              <div className="text-xs text-gray-600 font-mono">{pct}%</div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
