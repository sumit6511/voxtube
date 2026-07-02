import { CheckCircle2, XCircle, Info, X } from 'lucide-react'
import { useToastState } from '../hooks/useToast'
import type { ToastType } from '../hooks/useToast'

const CONFIG: Record<ToastType, { icon: React.ReactNode; bar: string; text: string }> = {
  success: { icon: <CheckCircle2 size={15} />, bar: 'bg-pos',   text: 'text-pos'   },
  error:   { icon: <XCircle      size={15} />, bar: 'bg-neg',   text: 'text-neg'   },
  info:    { icon: <Info         size={15} />, bar: 'bg-amber', text: 'text-amber' },
}

export default function ToastContainer() {
  const { toasts, dismiss } = useToastState()

  if (toasts.length === 0) return null

  return (
    <div className="fixed bottom-5 right-5 z-50 flex flex-col gap-2 pointer-events-none">
      {toasts.map(t => {
        const cfg = CONFIG[t.type]
        return (
          <div
            key={t.id}
            className="pointer-events-auto flex items-center gap-3 min-w-[260px] max-w-sm
                       bg-base-surface border border-base-border rounded-xl shadow-xl
                       px-4 py-3 animate-fade-up"
          >
            {/* Coloured left accent */}
            <div className={`w-0.5 self-stretch rounded-full flex-shrink-0 ${cfg.bar}`} />

            {/* Icon */}
            <span className={cfg.text}>{cfg.icon}</span>

            {/* Message */}
            <p className="text-sm font-body flex-1" style={{ color: 'var(--color-text)' }}>
              {t.message}
            </p>

            {/* Dismiss */}
            <button
              onClick={() => dismiss(t.id)}
              className="text-gray-600 hover:text-gray-400 flex-shrink-0 transition-colors"
            >
              <X size={13} />
            </button>
          </div>
        )
      })}
    </div>
  )
}
