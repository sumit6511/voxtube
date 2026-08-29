import { useState, useRef, useEffect } from 'react'
import { Send, Loader2, ChevronDown, Cpu } from 'lucide-react'
import { api, type ChatTurn } from '../api'
import Dropdown from './Dropdown'

interface Source { id: string; text: string; score: number }
interface Message {
  role: 'user' | 'assistant'; text: string
  sources?: Source[]; error?: boolean; streaming?: boolean
}

const SUGGESTIONS = [
  'What do viewers think about this video overall?',
  'What topics are most discussed in the comments?',
  'Are there any toxic or hateful comments?',
  'What do people say about the music or editing?',
  'Which aspects received the most praise?',
]

function ModelSelector({ selected, onChange }: { selected: string; onChange: (m: string) => void }) {
  const [models, setModels] = useState<string[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.ollamaModels()
      .then(data => {
        setModels(data.models); setError(data.error)
        if (!selected && (data.models.length > 0 || data.default)) {
          // Prefer the backend's configured default (OLLAMA_MODEL, e.g.
          // "llama3.2") when it's actually pulled — Ollama's tag list isn't
          // sorted, so blindly taking models[0] could pick a different model
          // just because it happens to come first. Model names carry a tag
          // suffix ("llama3.2:latest"), so match on that too.
          const preferred = data.models.find(
            m => m === data.default || m.startsWith(`${data.default}:`)
          )
          onChange(preferred ?? data.models[0] ?? data.default)
        }
        setLoading(false)
      })
      .catch(() => { setError('Could not reach backend'); setLoading(false) })
  }, [])

  return (
    <div className="flex items-center gap-2 pb-3 border-b border-base-border">
      <Cpu size={13} className="text-gray-600 flex-shrink-0" />
      <span className="text-xs font-mono text-gray-600">Model</span>
      {loading && <Loader2 size={12} className="animate-spin text-gray-600" />}
      {!loading && models.length > 0 && (
        <Dropdown
          value={selected}
          onChange={onChange}
          options={models.map(m => ({ value: m, label: m }))}
          size="sm"
          valueClassName="text-amber"
        />
      )}
      {!loading && models.length === 0 && !error && (
        <span className="text-xs font-mono text-gray-600">
          No models pulled — run: <code className="text-amber">ollama pull llama3.2</code>
        </span>
      )}
      {!loading && error && <span className="text-xs font-mono text-neg/80">{error}</span>}
      {!loading && selected && !error && (
        <span className="ml-auto text-xs font-mono text-gray-700 hidden sm:block">{selected}</span>
      )}
    </div>
  )
}

function SourcesCitation({ sources }: { sources: Source[] }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="mt-3 border-t border-base-border pt-2">
      <button onClick={() => setOpen(v => !v)}
        className="flex items-center gap-1.5 text-xs font-mono text-gray-600 hover:text-gray-400 transition-colors">
        <ChevronDown size={12} className={`transition-transform duration-200 ${open ? 'rotate-180' : ''}`} />
        {sources.length} source{sources.length !== 1 ? 's' : ''} used
      </button>
      {open && (
        <div className="mt-2 space-y-1.5">
          {sources.map(s => (
            <div key={s.id} className="text-xs text-gray-500 font-body bg-base rounded-lg px-3 py-2 border border-base-border/50">
              <span className="font-mono text-amber/60 mr-2">{(s.score * 100).toFixed(0)}%</span>
              {s.text.length > 120 ? `${s.text.slice(0, 120)}…` : s.text}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default function ChatPanel({ jobId }: { jobId: string }) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [selectedModel, setSelectedModel] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)
  // Starts false; the effect below flips it true on mount. This is the
  // StrictMode-safe pattern: React's dev-mode mount→unmount→remount cycle
  // runs the cleanup (sets false) then re-runs the effect body (sets true
  // again). The previous version only ever set it to true via useRef's
  // initial value and never reset it inside the effect itself, so after
  // StrictMode's simulated unmount it was permanently stuck at false —
  // silently no-oping every state update guarded by it (onDone, onError),
  // which is why the chat bubble never stopped "streaming".
  const isMountedRef = useRef(false)

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, loading])
  useEffect(() => {
    isMountedRef.current = true
    return () => { isMountedRef.current = false }
  }, [])

  async function send(question: string) {
    const q = question.trim()
    if (!q || loading) return
    setInput('')

    // Snapshot prior turns as conversation context for follow-up questions —
    // taken before the new user/assistant messages are appended below.
    const history: ChatTurn[] = messages
      .filter(m => !m.error)
      .slice(-10)
      .map(m => ({ role: m.role, text: m.text }))

    // Index the assistant placeholder will land at (user msg, then assistant msg)
    const assistantIndex = messages.length + 1

    setMessages(prev => [
      ...prev,
      { role: 'user', text: q },
      { role: 'assistant', text: '', streaming: true },
    ])
    setLoading(true)

    function updateAssistant(patch: Partial<Message>) {
      if (!isMountedRef.current) return
      setMessages(prev => prev.map((m, i) => (i === assistantIndex ? { ...m, ...patch } : m)))
    }

    await api.chatStream(jobId, q, selectedModel || undefined, history, {
      onSources: (sources) => updateAssistant({ sources }),
      onToken:   (text) => {
        if (!isMountedRef.current) return
        setMessages(prev => prev.map((m, i) =>
          i === assistantIndex ? { ...m, text: m.text + text } : m))
      },
      onError:   (message) => {
        updateAssistant({ text: message, error: true, streaming: false })
        if (isMountedRef.current) setLoading(false)
      },
      onDone: () => {
        updateAssistant({ streaming: false })
        if (isMountedRef.current) setLoading(false)
      },
    })
  }

  return (
    <div className="max-w-2xl mx-auto flex flex-col gap-4">
      <ModelSelector selected={selectedModel} onChange={setSelectedModel} />

      <div className="space-y-3 min-h-[100px]">
        {messages.length === 0 && (
          <div>
            <p className="label mb-3">Try asking</p>
            <div className="flex flex-wrap gap-2">
              {SUGGESTIONS.map(s => (
                <button key={s} onClick={() => send(s)} disabled={loading}
                  className="text-xs font-body text-gray-400 border border-base-border hover:border-amber/40
                             hover:text-gray-200 px-3 py-2 rounded-lg transition-all text-left">
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[88%] rounded-xl px-4 py-3 text-sm font-body leading-relaxed
              ${msg.role === 'user' ? 'bg-amber/10 border border-amber/20 text-gray-200'
                : msg.error ? 'bg-neg/5 border border-neg/20 text-neg' : 'card text-gray-300'}`}>
              <p className="whitespace-pre-wrap">
                {msg.text}
                {msg.streaming && (
                  <span className="inline-block w-1.5 h-4 bg-amber/70 ml-0.5 -mb-0.5
                                   animate-pulse-dot align-middle" />
                )}
              </p>
              {msg.sources && msg.sources.length > 0 && <SourcesCitation sources={msg.sources} />}
            </div>
          </div>
        ))}

        <div ref={bottomRef} />
      </div>

      <div className="flex gap-2">
        <input className="input-field flex-1 py-2.5 text-sm" placeholder="Ask about the comments…"
          value={input} onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && !e.shiftKey && send(input)} disabled={loading} />
        <button onClick={() => send(input)} disabled={loading || !input.trim()}
          className="btn-primary px-4 flex items-center gap-2 flex-shrink-0">
          {loading ? <Loader2 size={15} className="animate-spin" /> : <Send size={15} />}
        </button>
      </div>

      <p className="text-xs font-mono text-gray-700">
        Powered by Ollama (local LLM, streaming) · Hybrid BM25 + FAISS retrieval
      </p>
    </div>
  )
}
