export const BASE = 'http://localhost:8000'

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail ?? 'Request failed')
  }
  return res.json()
}

export interface AnalyzeResponse { job_id: string }
export interface StatusResponse {
  job_id: string; status: string; progress: number
  comment_count: number; video_id?: string
  video_title?: string; error_message?: string
}
export interface Topic {
  topic_id: number; label: string; keywords: string[]
  comment_count: number; positive_count: number
  neutral_count: number; negative_count: number
}
export interface Comment {
  id: string; original_text: string; clean_text?: string
  sentiment_label?: string; sentiment_score?: number
  vader_label?: string; vader_compound?: number
  is_toxic: number; toxicity_json?: string
  topic_id?: number; lang?: string; published_at?: string
}
export interface ResultsResponse {
  job_id: string; video_id?: string; video_title?: string
  youtube_url?: string; channel_title?: string
  view_count?: number; like_count?: number
  total_comments: number
  sentiment_summary: { positive: number; neutral: number; negative: number }
  topics: Topic[]; comments: Comment[]
}
export interface ChatSource { id: string; text: string; score: number }
export interface MetricsResult {
  accuracy: number; precision: number; recall: number; f1: number
  confusion_matrix: number[][]
}
export interface EvaluationResponse {
  total_samples: number; label_distribution: Record<string, number>
  xlm_roberta: MetricsResult | null; vader: MetricsResult; note?: string | null
  // Third, optional comparison point — see sentiment.py for why this is
  // kept separate from the main XLM-RoBERTa vs VADER comparison.
  nepali_model: MetricsResult | null; nepali_model_note?: string | null
}
export interface JobSummary {
  id: string; youtube_url: string; video_title?: string | null
  status: string; progress: number; comment_count: number
  created_at?: string | null
}
export interface Entity {
  text: string; category: 'Person' | 'Organization' | 'Location' | 'Miscellaneous'
  count: number; sentiment: string
}
export interface NerResponse {
  entities: Entity[]; total_processed: number
  total_skipped: number; model_available: boolean
}
export interface ScatterPoint {
  id: string; x: number; y: number; sentiment: string
  lang: string; text: string; is_toxic: number
}
export interface UmapResponse {
  points: ScatterPoint[]; method: string; total: number
}

export const api = {
  analyze: (url: string, maxComments = 200) =>
    request<AnalyzeResponse>('/analyze', {
      method: 'POST',
      body: JSON.stringify({ url, max_comments: maxComments }),
    }),

  status: (jobId: string) =>
    request<StatusResponse>(`/status/${jobId}`),

  results: (jobId: string) =>
    request<ResultsResponse>(`/results/${jobId}`),

  chatStream: async (
    jobId: string,
    question: string,
    model: string | undefined,
    handlers: {
      onSources?: (sources: ChatSource[]) => void
      onToken?:   (text: string) => void
      onError?:   (message: string) => void
      onDone?:    () => void
    }
  ): Promise<void> => {
    let res: Response
    try {
      res = await fetch(`${BASE}/chat/${jobId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, model: model ?? null }),
      })
    } catch {
      handlers.onError?.('Could not reach the backend. Is it running?')
      return
    }

    if (!res.ok || !res.body) {
      const err = await res.json().catch(() => ({ detail: res.statusText }))
      handlers.onError?.(err.detail ?? 'Request failed')
      return
    }

    const reader  = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    let finished = false

    function processLine(line: string) {
      if (!line) return
      try {
        const evt = JSON.parse(line)
        if      (evt.type === 'sources') handlers.onSources?.(evt.sources)
        else if (evt.type === 'token')   handlers.onToken?.(evt.text)
        else if (evt.type === 'error')   { finished = true; handlers.onError?.(evt.message) }
        else if (evt.type === 'done')    { finished = true; handlers.onDone?.() }
      } catch { /* skip malformed line */ }
    }

    const IDLE_TIMEOUT_MS = 45_000
    let timedOut = false

    async function readWithTimeout() {
      let timer: ReturnType<typeof setTimeout>
      const timeout = new Promise<'timeout'>(resolve => {
        timer = setTimeout(() => resolve('timeout'), IDLE_TIMEOUT_MS)
      })
      try {
        return await Promise.race([reader.read(), timeout])
      } finally {
        clearTimeout(timer!)
      }
    }

    while (true) {
      const result = await readWithTimeout()
      if (result === 'timeout') {
        timedOut = true
        try { await reader.cancel() } catch { /* ignore */ }
        break
      }
      const { done, value } = result
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      let idx: number
      while ((idx = buffer.indexOf('\n')) >= 0) {
        processLine(buffer.slice(0, idx).trim())
        buffer = buffer.slice(idx + 1)
      }
    }

    if (!timedOut) {
      buffer += decoder.decode()
      processLine(buffer.trim())
    }
    if (!finished) handlers.onDone?.()
  },

  evaluate: () =>
    request<EvaluationResponse>('/evaluate'),

  jobs: () =>
    request<{ jobs: JobSummary[]; total: number }>('/jobs'),

  deleteJob: (jobId: string) =>
    request<{ deleted: string }>(`/jobs/${jobId}`, { method: 'DELETE' }),

  ollamaModels: () =>
    request<{ models: string[]; default: string; error: string | null }>('/ollama/models'),

  ner: (jobId: string) =>
    request<NerResponse>(`/ner/${jobId}`),

  umap: (jobId: string) =>
    request<UmapResponse>(`/umap/${jobId}`),
}
