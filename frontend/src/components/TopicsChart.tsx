import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import type { Topic } from '../api'
import { useMountedAfterTick } from '../hooks/useMountedAfterTick'

interface Props {
  topics: Topic[]
  onTopicClick?: (topicId: number) => void
}

const TOOLTIP_STYLE = {
  background: '#13161E', border: '1px solid #1E2330',
  borderRadius: '8px', fontSize: '12px', fontFamily: 'IBM Plex Mono',
}

const MAX_CHARS_PER_LINE = 12

// Wraps a topic title onto at most 2 lines, breaking on the nearest word
// boundary; a second line that still overflows is truncated with an ellipsis.
function wrapLabel(text: string): string[] {
  if (text.length <= MAX_CHARS_PER_LINE) return [text]

  const words = text.split(' ')
  let line1 = ''
  let i = 0
  for (; i < words.length; i++) {
    const next = line1 ? `${line1} ${words[i]}` : words[i]
    if (line1 && next.length > MAX_CHARS_PER_LINE) break
    line1 = next
  }
  if (i === 0) {
    // Single word longer than the line — hard-split it instead of stalling.
    return [text.slice(0, MAX_CHARS_PER_LINE), text.slice(MAX_CHARS_PER_LINE, MAX_CHARS_PER_LINE * 2 - 1) + '…']
  }

  let line2 = words.slice(i).join(' ')
  if (line2.length > MAX_CHARS_PER_LINE) line2 = `${line2.slice(0, MAX_CHARS_PER_LINE - 1)}…`
  return line2 ? [line1, line2] : [line1]
}

function TopicTick({ x, y, payload }: any) {
  const lines = wrapLabel(payload.value)
  return (
    <g transform={`translate(${x},${y})`}>
      <text textAnchor="middle" fill="#6B7280" fontSize={11} fontFamily="IBM Plex Mono">
        {lines.map((line, i) => <tspan key={i} x={0} dy={i === 0 ? 14 : 13}>{line}</tspan>)}
      </text>
    </g>
  )
}

export default function TopicsChart({ topics, onTopicClick }: Props) {
  const mounted = useMountedAfterTick()
  const data = [...topics]
    .sort((a, b) => b.comment_count - a.comment_count)
    .slice(0, 8)
    .map(t => ({
      name: t.label.split(' | ')[0],
      topic_id: t.topic_id,
      positive: t.positive_count, neutral: t.neutral_count, negative: t.negative_count,
    }))

  // Recharts passes the merged bar props (including .payload = the source
  // data row) to onClick — grab topic_id from there and bubble it up.
  function handleBarClick(barProps: any) {
    if (!onTopicClick) return
    const topicId = barProps?.payload?.topic_id ?? barProps?.topic_id
    if (topicId !== undefined) onTopicClick(topicId)
  }

  return (
    <div>
      {mounted && (
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={data} margin={{ top: 4, right: 0, left: -22, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1E2330" vertical={false} />
            <XAxis dataKey="name" tick={<TopicTick />} axisLine={false} tickLine={false} interval={0} />
            <YAxis tick={{ fill: '#6B7280', fontSize: 11 }} axisLine={false} tickLine={false} allowDecimals={false} />
            <Tooltip contentStyle={TOOLTIP_STYLE} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
            <Legend iconType="circle" iconSize={8} verticalAlign="top" align="right" height={28}
              formatter={v => (
                <span style={{ fontSize: '12px', color: '#9CA3AF', fontFamily: 'IBM Plex Mono' }}>{v}</span>
              )} />
            <Bar dataKey="positive" stackId="s" fill="#10B981" name="Positive"
                 onClick={handleBarClick} style={{ cursor: onTopicClick ? 'pointer' : 'default' }} />
            <Bar dataKey="neutral"  stackId="s" fill="#6B7280" name="Neutral"
                 onClick={handleBarClick} style={{ cursor: onTopicClick ? 'pointer' : 'default' }} />
            <Bar dataKey="negative" stackId="s" fill="#F43F5E" name="Negative" radius={[4, 4, 0, 0]}
                 onClick={handleBarClick} style={{ cursor: onTopicClick ? 'pointer' : 'default' }} />
          </BarChart>
        </ResponsiveContainer>
      )}
      {!mounted && <div style={{ height: 240 }} />}

      {onTopicClick && (
        <p className="text-xs font-mono text-gray-700 text-center -mt-1">
          Click a bar to view its comments →
        </p>
      )}
    </div>
  )
}
