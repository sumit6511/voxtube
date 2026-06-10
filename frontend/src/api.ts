const BASE = 'http://localhost:8000'

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

// ── Types ─────────────────────────────────────────────────────────────────────

export interface AnalyzeResponse  { job_id: string }
export interface StatusResponse   {
  job_id: string; status: string; progress: number
  comment_count: number; video_title?: string; error_message?: string
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
  topic_id?: number; lang?: string
}
export interface ResultsResponse {
  job_id: string; video_title?: string; total_comments: number
  sentiment_summary: { positive: number; neutral: number; negative: number }
  topics: Topic[]; comments: Comment[]
}
export interface ChatResponse {
  answer: string
  sources: { id: string; text: string; score: number }[]
}

// ── Calls ─────────────────────────────────────────────────────────────────────

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

  chat: (jobId: string, question: string) =>
    request<ChatResponse>(`/chat/${jobId}`, {
      method: 'POST',
      body: JSON.stringify({ question }),
    }),
}
