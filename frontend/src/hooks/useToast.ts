import { useState, useCallback } from 'react'

export type ToastType = 'success' | 'error' | 'info'

export interface Toast {
  id:      string
  message: string
  type:    ToastType
}

// Module-level singleton so any component can push toasts
// without needing prop drilling or Context.
type Listener = (toasts: Toast[]) => void
let _toasts:   Toast[]    = []
let _listener: Listener | null = null

function notify() {
  _listener?.([..._toasts])
}

export function pushToast(message: string, type: ToastType = 'info') {
  const id = Math.random().toString(36).slice(2)
  _toasts = [..._toasts, { id, message, type }]
  notify()
  // Auto-dismiss after 4 s
  setTimeout(() => {
    _toasts = _toasts.filter(t => t.id !== id)
    notify()
  }, 4000)
}

/** Hook used by the single <ToastContainer /> to receive updates. */
export function useToastState() {
  const [toasts, setToasts] = useState<Toast[]>([])
  _listener = setToasts
  const dismiss = useCallback((id: string) => {
    _toasts = _toasts.filter(t => t.id !== id)
    notify()
  }, [])
  return { toasts, dismiss }
}
