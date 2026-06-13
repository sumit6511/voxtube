import { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { Clapperboard, ArrowRight, Loader2 } from 'lucide-react'
import { api } from '../api'
import { useStore } from '../store'

const EXAMPLES = [
  'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
  'https://youtu.be/dQw4w9WgXcQ',
]

export default function Home() {
  const navigate  = useNavigate()
  const setJobId  = useStore(s => s.setJobId)

  const [url,     setUrl]     = useState('')
  const [max,     setMax]     = useState(200)
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState('')

  async function handleSubmit() {
    const trimmed = url.trim()
    if (!trimmed) { setError('Please enter a YouTube URL.'); return }
    setError('')
    setLoading(true)
    try {
      const { job_id } = await api.analyze(trimmed, max)
      setJobId(job_id)
      navigate(`/progress/${job_id}`)
    } catch (e: any) {
      setError(e.message ?? 'Something went wrong. Is the backend running?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-4 py-16">

      {/* Wordmark */}
      <div className="animate-fade-up mb-12 text-center" style={{ animationDelay: '0ms' }}>
        <div className="flex items-center justify-center gap-3 mb-3">
          <div className="w-10 h-10 rounded-lg bg-amber/10 border border-amber/30 flex items-center justify-center">
            <Clapperboard size={20} className="text-amber" />
          </div>
          <span className="font-display text-3xl font-extrabold tracking-tight text-white">
            Vox<span className="text-amber">Tube</span>
          </span>
        </div>
        <p className="text-gray-500 font-body text-sm max-w-xs leading-relaxed">
          Multidimensional sentiment analysis &amp; topic modeling for YouTube comments
        </p>
      </div>

      {/* Card */}
      <div
        className="animate-fade-up w-full max-w-lg card space-y-5"
        style={{ animationDelay: '80ms' }}
      >
        {/* URL input */}
        <div>
          <label className="label mb-2 block">YouTube URL</label>
          <input
            className="input-field"
            placeholder="https://www.youtube.com/watch?v=..."
            value={url}
            onChange={e => { setUrl(e.target.value); setError('') }}
            onKeyDown={e => e.key === 'Enter' && handleSubmit()}
            disabled={loading}
          />
          {/* Quick-fill examples */}
          <div className="mt-2 flex flex-wrap gap-2">
            {EXAMPLES.map(ex => (
              <button
                key={ex}
                onClick={() => setUrl(ex)}
                className="text-xs font-mono text-gray-600 hover:text-amber transition-colors truncate max-w-[240px]"
              >
                {ex}
              </button>
            ))}
          </div>
        </div>

        {/* Max comments slider */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="label">Max comments</label>
            <span className="font-mono text-sm text-amber">{max}</span>
          </div>
          <input
            type="range" min={50} max={500} step={50}
            value={max}
            onChange={e => setMax(Number(e.target.value))}
            disabled={loading}
            className="w-full accent-amber h-1 rounded-full cursor-pointer"
          />
          <div className="flex justify-between mt-1">
            <span className="text-xs text-gray-600 font-mono">50</span>
            <span className="text-xs text-gray-600 font-mono">500</span>
          </div>
        </div>

        {/* Error */}
        {error && (
          <p className="text-neg text-sm font-mono bg-neg/5 border border-neg/20 rounded-lg px-4 py-2">
            {error}
          </p>
        )}

        {/* Submit */}
        <button
          className="btn-primary w-full flex items-center justify-center gap-2"
          onClick={handleSubmit}
          disabled={loading}
        >
          {loading
            ? <><Loader2 size={16} className="animate-spin" /> Submitting…</>
            : <><span>Analyze Comments</span><ArrowRight size={16} /></>
          }
        </button>
      </div>

      {/* Feature tags */}
      <div
        className="animate-fade-up mt-8 flex flex-wrap justify-center gap-2"
        style={{ animationDelay: '160ms' }}
      >
        {['XLM-RoBERTa', 'VADER Baseline', 'BERTopic', 'ToxicBERT', 'RAG · Ollama'].map(tag => (
          <span key={tag} className="text-xs font-mono text-gray-600 border border-base-border rounded-full px-3 py-1">
            {tag}
          </span>
        ))}
      </div>

      {/* Secondary links */}
      <div
        className="animate-fade-up mt-4 flex items-center gap-3"
        style={{ animationDelay: '220ms' }}
      >
        <Link
          to="/history"
          className="flex items-center gap-1.5 text-xs font-mono text-gray-500
                     hover:text-amber border border-base-border hover:border-amber/40
                     px-4 py-2 rounded-full transition-all duration-200"
        >
          <span className="text-amber">◷</span>
          History
          <ArrowRight size={12} />
        </Link>
        <Link
          to="/evaluate"
          className="flex items-center gap-1.5 text-xs font-mono text-gray-500
                     hover:text-amber border border-base-border hover:border-amber/40
                     px-4 py-2 rounded-full transition-all duration-200"
        >
          <span className="text-amber">⬡</span>
          Model Evaluation
          <ArrowRight size={12} />
        </Link>
      </div>
    </div>
  )
}
