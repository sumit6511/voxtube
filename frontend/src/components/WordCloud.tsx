import { useEffect, useRef, useState, useMemo } from 'react'
import cloud from 'd3-cloud'
import type { Comment } from '../api'

interface Props { comments: Comment[] }

const STOP = new Set([
  'the','a','an','is','are','was','were','be','been','being','have','has','had','do','does','did',
  'will','would','could','should','not','no','this','that','these','those','it','its','i','you','he',
  'she','we','they','me','him','her','us','them','my','your','his','our','their','what','which','who',
  'how','all','any','some','just','but','and','or','for','in','on','at','to','of','as','if','so',
  'get','got','like','also','about','very','still','even','here','there',
  'cha','xa','yo','ko','ma','lai','le','ra','ani','tara','ho','haina','ni','ta','ki','k','ek','ne',
  'garne','pani','chai','aile','huncha','thyo','hola','garcha','hunna','bhayo','bhi','bro','yar','yaar','sir',
  'यो','को','मा','लाई','ले','र','हो','छ','नि','पनि','त','कि',
])

function tokenize(text: string): string[] {
  return text.toLowerCase().split(/[\s.,!?;:()\[\]"'`~@#%^&*+=<>/\\|{}_\-0-9]+/)
    .filter(w => w.length > 2 && !STOP.has(w))
}

function wordFrequencies(comments: Comment[]): [string, number][] {
  const freq: Record<string, number> = {}
  for (const c of comments)
    for (const w of tokenize(c.clean_text || c.original_text || ''))
      freq[w] = (freq[w] ?? 0) + 1
  return Object.entries(freq).sort((a, b) => b[1] - a[1]).slice(0, 80)
}

interface CloudInput { text: string; size: number; freq: number; x?: number; y?: number; rotate?: number }

function lerp(v: number, inMin: number, inMax: number, outMin: number, outMax: number) {
  if (inMax === inMin) return (outMin + outMax) / 2
  return outMin + ((v - inMin) / (inMax - inMin)) * (outMax - outMin)
}

const PALETTE = ['#F59E0B','#EF4444','#10B981','#F97316','#8B5CF6','#3B82F6','#EC4899','#84CC16','#06B6D4','#A16207']

function colorFor(text: string): string {
  let h = 0
  for (let i = 0; i < text.length; i++) h = text.charCodeAt(i) + ((h << 5) - h)
  return PALETTE[Math.abs(h) % PALETTE.length]
}

interface PlacedWord { text: string; size: number; x: number; y: number; rotate: number; freq: number }

// d3-cloud measures each word on an offscreen canvas using this exact font to
// decide non-overlapping placement. DM Sans loads asynchronously (Google Fonts
// @import), so if layout runs before it's ready, the canvas falls back to a
// system font for measurement — the computed layout no longer matches what
// actually paints in the SVG, and words end up visually overlapping despite
// the algorithm having "succeeded". Force both weights to load first.
async function ensureFontReady() {
  try {
    await Promise.all([
      document.fonts.load('500 16px "DM Sans"'),
      document.fonts.load('700 16px "DM Sans"'),
    ])
    await document.fonts.ready
  } catch { /* Font Loading API unavailable — proceed with best-effort metrics */ }
}

export default function WordCloud({ comments }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [placed, setPlaced] = useState<PlacedWord[]>([])
  const [computing, setComputing] = useState(true)
  const [dims, setDims] = useState({ w: 700, h: 380 })

  const frequencies = useMemo(() => wordFrequencies(comments), [comments])

  useEffect(() => {
    const el = containerRef.current
    if (!el || frequencies.length === 0) { setComputing(false); return }

    const w = el.clientWidth || 700
    const h = 380
    setDims({ w, h })
    setComputing(true)

    const maxF = frequencies[0][1]
    const minF = frequencies[frequencies.length - 1][1]
    const words: CloudInput[] = frequencies.map(([text, freq]) => ({
      text, size: Math.round(lerp(freq, minF, maxF, 13, 64)), freq,
    }))

    let cancelled = false

    ensureFontReady().then(() => {
      if (cancelled) return
      cloud<CloudInput>()
        .size([w, h]).words(words).padding(6)
        // Continuous random tilt instead of a fixed 0/±90 choice — d3-cloud's
        // collision detection works on rasterized sprites, so arbitrary
        // angles are just as safe for non-overlap as axis-aligned ones.
        .rotate(() => Math.random() * 70 - 35)
        .font('DM Sans').fontWeight((d) => (d.size! > 28 ? '700' : '500')).fontSize(d => d.size!)
        .on('end', (result: CloudInput[]) => {
          if (cancelled) return
          setPlaced(result.map(d => ({
            text: d.text!, size: d.size!, x: d.x ?? 0, y: d.y ?? 0, rotate: d.rotate ?? 0, freq: d.freq,
          })))
          setComputing(false)
        })
        .start()
    })

    return () => { cancelled = true }
  }, [frequencies])

  const missing = frequencies.length - placed.length

  return (
    <div ref={containerRef}>
      <p className="font-mono text-xs text-gray-600 text-right mb-3">
        {placed.length} words · {comments.length} comments
      </p>

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
          <svg width={dims.w} height={dims.h} style={{ overflow: 'visible' }}>
            <g transform={`translate(${dims.w / 2},${dims.h / 2})`}>
              {placed.map(w => (
                <text key={w.text} textAnchor="middle" dominantBaseline="middle"
                  transform={`translate(${w.x},${w.y}) rotate(${w.rotate})`}
                  style={{
                    fontSize: w.size, fontFamily: 'DM Sans, sans-serif',
                    fontWeight: w.size > 28 ? 700 : 500, fill: colorFor(w.text),
                    cursor: 'default', userSelect: 'none',
                  }}>
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
