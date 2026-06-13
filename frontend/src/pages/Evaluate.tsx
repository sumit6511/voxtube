import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { ArrowLeft, AlertTriangle } from 'lucide-react'
import { api } from '../api'
import type { EvaluationResponse } from '../api'

const LABELS = ['Positive', 'Neutral', 'Negative']

function MetricBar({ label, value, highlight }: { label: string; value: number; highlight?: boolean }) {
  return (
    <div>
      <div className="flex justify-between mb-1">
        <span className="text-xs font-mono text-gray-400">{label}</span>
        <span className={`text-xs font-mono font-medium ${highlight ? 'text-amber' : 'text-gray-300'}`}>
          {(value * 100).toFixed(1)}%
        </span>
      </div>
      <div className="h-1.5 bg-base rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-700 ${highlight ? 'bg-amber' : 'bg-gray-600'}`}
          style={{ width: `${value * 100}%` }}
        />
      </div>
    </div>
  )
}

function ConfusionMatrix({ matrix, title }: { matrix: number[][]; title: string }) {
  const rowTotals = matrix.map(r => r.reduce((a, b) => a + b, 0))
  return (
    <div>
      <p className="label mb-2">{title}</p>
      <div className="text-xs font-mono text-gray-600 mb-1 pl-16">← Predicted</div>
      <div className="flex items-center gap-1">
        <div className="flex flex-col gap-1 mr-1" style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)' }}>
          <span className="text-xs font-mono text-gray-600 text-center">↑ Actual</span>
        </div>
        <div>
          {/* Header row */}
          <div className="flex gap-1 mb-1">
            <div className="w-16" />
            {LABELS.map(l => (
              <div key={l} className="w-16 text-center text-xs font-mono text-gray-500 truncate">{l}</div>
            ))}
          </div>
          {/* Matrix rows */}
          {matrix.map((row, i) => (
            <div key={i} className="flex gap-1 mb-1 items-center">
              <div className="w-16 text-right pr-2 text-xs font-mono text-gray-500 truncate">{LABELS[i]}</div>
              {row.map((val, j) => (
                <div
                  key={j}
                  className={`w-16 h-10 flex items-center justify-center rounded text-sm font-mono font-medium border
                    ${i === j ? 'border-pos/30 bg-pos/15 text-pos' : 'border-base-border bg-base text-gray-500'}`}
                >
                  {val}
                </div>
              ))}
              <div className="text-xs font-mono text-gray-700 ml-1">/{rowTotals[i]}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

export default function Evaluate() {
  const navigate = useNavigate()
  const [data,    setData]    = useState<EvaluationResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error,   setError]   = useState('')

  useEffect(() => {
    api.evaluate()
      .then(d => { setData(d); setLoading(false) })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [])

  if (loading) return (
    <div className="min-h-screen flex items-center justify-center">
      <div className="text-center space-y-3">
        <div className="w-8 h-8 border-2 border-amber border-t-transparent rounded-full animate-spin mx-auto" />
        <p className="text-gray-500 font-mono text-sm">Running evaluation on 150 samples…</p>
        <p className="text-gray-700 font-mono text-xs">First run loads models (~30s)</p>
      </div>
    </div>
  )

  if (error) return (
    <div className="min-h-screen flex items-center justify-center px-4">
      <div className="card max-w-md w-full text-center space-y-3">
        <AlertTriangle size={24} className="text-neg mx-auto" />
        <p className="text-neg font-mono text-sm">{error}</p>
        <button onClick={() => navigate('/')} className="btn-primary">← Go back</button>
      </div>
    </div>
  )

  if (!data) return null

  const { total_samples, label_distribution: dist, xlm_roberta: xlm, vader, note } = data

  const metrics = [
    { key: 'f1',        label: 'F1 Score (Weighted)' },
    { key: 'accuracy',  label: 'Accuracy'             },
    { key: 'precision', label: 'Precision (Weighted)' },
    { key: 'recall',    label: 'Recall (Weighted)'    },
  ] as const

  return (
    <div className="min-h-screen px-4 py-6 max-w-4xl mx-auto">

      {/* Header */}
      <div className="flex items-start gap-3 mb-6">
        <button onClick={() => navigate('/')} className="mt-0.5 text-gray-500 hover:text-white transition-colors">
          <ArrowLeft size={18} />
        </button>
        <div>
          <h1 className="font-display text-xl font-bold text-white">Model Evaluation</h1>
          <p className="text-xs font-mono text-gray-600 mt-0.5">
            XLM-RoBERTa vs VADER — Labeled Neplish Dataset
          </p>
        </div>
      </div>

      {/* Note */}
      {note && (
        <div className="card border-amber/20 bg-amber/5 mb-4 flex gap-2 items-start">
          <AlertTriangle size={14} className="text-amber mt-0.5 flex-shrink-0" />
          <p className="text-xs font-mono text-amber/80">{note}</p>
        </div>
      )}

      {/* Dataset overview */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6">
        {[
          { label: 'Total Samples', value: total_samples,        color: 'text-white'  },
          { label: 'Positive',      value: dist.positive ?? 0,   color: 'text-pos'    },
          { label: 'Neutral',       value: dist.neutral  ?? 0,   color: 'text-neu'    },
          { label: 'Negative',      value: dist.negative ?? 0,   color: 'text-neg'    },
        ].map(s => (
          <div key={s.label} className="card text-center py-3">
            <div className={`font-display text-2xl font-bold ${s.color}`}>{s.value}</div>
            <div className="label mt-1">{s.label}</div>
          </div>
        ))}
      </div>

      {/* Metrics comparison */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">

        {/* XLM-RoBERTa */}
        <div className="card space-y-3">
          <div className="flex items-center justify-between">
            <p className="label">XLM-RoBERTa</p>
            {xlm && (
              <span className="text-xs font-mono bg-pos/10 text-pos border border-pos/20 px-2 py-0.5 rounded-full">
                Primary model
              </span>
            )}
          </div>
          {xlm ? (
            <>
              {metrics.map(m => (
                <MetricBar key={m.key} label={m.label} value={xlm[m.key]} highlight />
              ))}
              <p className="text-xs font-mono text-amber pt-1">
                F1: {(xlm.f1 * 100).toFixed(1)}% · Acc: {(xlm.accuracy * 100).toFixed(1)}%
              </p>
            </>
          ) : (
            <div className="py-6 text-center">
              <p className="text-gray-600 font-mono text-sm">
                Install torch to see XLM-RoBERTa results.
              </p>
              <code className="text-xs text-amber mt-2 block">
                pip install torch --index-url https://download.pytorch.org/whl/cpu
              </code>
            </div>
          )}
        </div>

        {/* VADER */}
        <div className="card space-y-3">
          <div className="flex items-center justify-between">
            <p className="label">VADER</p>
            <span className="text-xs font-mono bg-gray-800 text-gray-400 border border-gray-700 px-2 py-0.5 rounded-full">
              Baseline
            </span>
          </div>
          {metrics.map(m => (
            <MetricBar key={m.key} label={m.label} value={vader[m.key]} />
          ))}
          <p className="text-xs font-mono text-gray-500 pt-1">
            F1: {(vader.f1 * 100).toFixed(1)}% · Acc: {(vader.accuracy * 100).toFixed(1)}%
          </p>
        </div>
      </div>

      {/* Improvement badge */}
      {xlm && (
        <div className="card mb-6 flex items-center justify-between">
          <div>
            <p className="text-sm font-body text-gray-300">
              XLM-RoBERTa improvement over VADER baseline
            </p>
            <p className="text-xs font-mono text-gray-600 mt-0.5">
              Weighted F1 score — higher is better
            </p>
          </div>
          <div className="text-right">
            <div className={`font-display text-3xl font-bold ${
              xlm.f1 >= vader.f1 ? 'text-pos' : 'text-neg'
            }`}>
              {xlm.f1 >= vader.f1 ? '+' : ''}{((xlm.f1 - vader.f1) * 100).toFixed(1)}%
            </div>
            <div className="text-xs font-mono text-gray-600">F1 delta</div>
          </div>
        </div>
      )}

      {/* Confusion Matrices */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {xlm && (
          <div className="card overflow-x-auto">
            <ConfusionMatrix matrix={xlm.confusion_matrix} title="XLM-RoBERTa — Confusion Matrix" />
          </div>
        )}
        <div className={`card overflow-x-auto ${!xlm ? 'md:col-span-2 max-w-md mx-auto w-full' : ''}`}>
          <ConfusionMatrix matrix={vader.confusion_matrix} title="VADER — Confusion Matrix" />
        </div>
      </div>

      {/* Academic note */}
      <div className="card mt-4 border-base-border/50">
        <p className="text-xs font-mono text-gray-600 leading-relaxed">
          <span className="text-gray-400">Dataset: </span>
          150 manually labeled Neplish YouTube comments (50 positive, 51 neutral, 49 negative) spanning
          Devanagari script, romanized Nepali, English, and code-mixed text. VADER assigns neutral (0.000)
          to all Devanagari-script comments regardless of sentiment, which inflates its neutral predictions
          and reduces F1 on this dataset. XLM-RoBERTa processes all three language types natively via its
          multilingual tokenizer, validating its selection as the primary model for Neplish content.
        </p>
      </div>

    </div>
  )
}
