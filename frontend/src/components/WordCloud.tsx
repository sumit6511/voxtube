import { useEffect, useRef, useState, useMemo } from 'react'
import cloud from 'd3-cloud'
import type { Comment } from '../api'

interface Props { comments: Comment[] }

// ── Stop words ────────────────────────────────────────────────────────────────
const STOP = new Set([
  'the','a','an','is','are','was','were','be','been','being',
  'have','has','had','do','does','did','will','would','could','should',
  'not','no','this','that','these','those','it','its','i','you','he',
  'she','we','they','me','him','her','us','them','my','your','his',
  'our','their','what','which','who','how','all','any','some','just',
  'but','and','or','for','in','on','at','to','of','as','if','so',
  'get','got','like','also','about','very','still','even','here','there',
  // romanized Nepali
  'cha','xa','yo','ko','ma','lai','le','ra','ani','tara','ho','haina',
  'ni','ta','ki','k','ek','ne','garne','pani','chai','aile','huncha',
  'thyo','hola','garcha','hunna','bhayo','bhi','bro','yar','yaar','sir',
  // Devanagari particles
  'यो','को','मा','लाई','ले','र','हो','छ','नि','पनि','त','कि',
])

function tokenize(text: string): string[] {
  return text.toLowerCase()
    .split(/[\s.,!?;:()\[\]"'`~@#%^&*+=<>/\\|{}_\-0-9]+/)
    .filter(w => w.length > 2 && !STOP.has(w))
}

function wordFrequencies(comments: Comment[]): [string, number][] {
  const freq: Record<string, number> = {}
  for (const c of comments) {
    for (const w of tokenize(c.clean_text || c.original_text || ''))
      freq[w] = (freq[w] ?? 0) + 1
  }
  return Object.entries(freq).sort((a, b) => b[1] - a[1]).slice(0, 80)
}

interface CloudInput {
  text: string
  size: number
  freq: number
  // d3-cloud adds these after layout:
  x?: number
  y?: number
  rotate?: number
}

function lerp(v: number, inMin: number, inMax: number, outMin: number, outMax: number) {
  if (inMax === inMin) return (outMin + outMax) / 2
  return outMin + ((v - inMin) / (inMax - inMin)) * (outMax - outMin)
}

// Warm + vibrant palette — similar feel to the reference image
const PALETTE = [
  '#F59E0B', // amber
  '#EF4444', // red
  '#10B981', // emerald
  '#F97316', // orange
  '#8B5CF6', // purple
  '#3B82F6', // blue
  '#EC4899', // pink
  '#84CC16', // lime
  '#06B6D4', // cyan
  '#A16207', // dark amber
]

function colorFor(text: string): string {
  let h = 0
  for (let i = 0; i < text.length; i++) h = text.charCodeAt(i) + ((h << 5) - h)
  return PALETTE[Math.abs(h) % PALETTE.length]
}

interface PlacedWord {
  text: string; size: number; x: number; y: number
  rotate: number; freq: number
}

// ── Component ─────────────────────────────────────────────────────────────────
export default function WordCloud({ comments }: Props) {
  const containerRef                    = useRef<HTMLDivElement>(null)
  const [placed, setPlaced]             = useState<PlacedWord[]>([])
  const [computing, setComputing]       = useState(true)
  const [dims, setDims]                 = useState({ w: 700, h: 380 })

  const frequencies = useMemo(() => wordFrequencies(comments), [comments])

  useEffect(() => {
    const el = containerRef.current
    if (!el || frequencies.length === 0) { setComputing(false); return }

    const w = el.clientWidth  || 700
    const h = 380
    setDims({ w, h })
    setComputing(true)

    const maxF = frequencies[0][1]
    const minF = frequencies[frequencies.length - 1][1]

    const words: CloudInput[] = frequencies.map(([text, freq]) => ({
      text,
      size: Math.round(lerp(freq, minF, maxF, 13, 64)),
      freq,
    }))

    cloud<CloudInput>()
      .size([w, h])
      .words(words)
      .padding(5)
      .rotate(() => {
        const r = Math.random()
        if (r < 0.55) return 0
        if (r < 0.75) return 90
        return -90
      })
      .font('DM Sans')
      .fontWeight((d) => (d.size! > 28 ? '700' : '500'))
      .fontSize(d => d.size!)
      .on('end', (result: CloudInput[]) => {
        setPlaced(result.map(d => ({
          text:   d.text!,
          size:   d.size!,
          x:      d.x ?? 0,
          y:      d.y ?? 0,
          rotate: d.rotate ?? 0,
          freq:   d.freq,
        })))
        setComputing(false)
      })
      .start()
  }, [frequencies])

  const missing = frequencies.length - placed.length

  return (
    <div ref={containerRef}>
      <div className="flex items-center justify-between mb-3">
        <p className="label">Word cloud</p>
        <p className="font-mono text-xs text-gray-600">
          {placed.length} words · {comments.length} comments
        </p>
      </div>

      {computing && (
        <div className="h-[380px] flex items-center justify-center">
          <div className="flex items-center gap-2 text-gray-500 font-mono text-sm">
            <div className="w-4 h-4 border-2 border-amber border-t-transparent rounded-full animate-spin" />
            Building word cloud…
          </div>
        </div>
      )}

      {!computing && placed.length === 0 && (
        <p className="h-[380px] flex items-center justify-center text-gray-600 font-mono text-sm">
          Not enough text to generate a word cloud.
        </p>
      )}

      {!computing && placed.length > 0 && (
        <>
          <svg
            width={dims.w}
            height={dims.h}
            style={{ overflow: 'visible' }}
          >
            <g transform={`translate(${dims.w / 2},${dims.h / 2})`}>
              {placed.map(w => (
                <text
                  key={w.text}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  transform={`translate(${w.x},${w.y}) rotate(${w.rotate})`}
                  style={{
                    fontSize:    w.size,
                    fontFamily:  'DM Sans, sans-serif',
                    fontWeight:  w.size > 28 ? 700 : 500,
                    fill:        colorFor(w.text),
                    cursor:      'default',
                    userSelect:  'none',
                  }}
                >
                  <title>{w.text} × {w.freq}</title>
                  {w.text}
                </text>
              ))}
            </g>
          </svg>

          {missing > 0 && (
            <p className="text-xs text-gray-700 font-mono mt-1 text-right">
              {missing} low-frequency words omitted (space)
            </p>
          )}
        </>
      )}
    </div>
  )
}
