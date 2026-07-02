import { useState, useRef, useEffect } from 'react'
import { ChevronDown, Check } from 'lucide-react'

export interface DropdownOption {
  value: string
  label: string
}

interface Props {
  value:           string
  onChange:        (v: string) => void
  options:         DropdownOption[]
  icon?:           React.ReactNode
  size?:           'sm' | 'md'
  valueClassName?: string   // overrides the trigger's value text color (e.g. "text-amber")
  className?:      string   // extra classes on the outer wrapper (e.g. "flex-shrink-0")
}

const SIZE_PADDING: Record<'sm' | 'md', string> = {
  sm: 'px-2.5 py-1',
  md: 'px-3 py-2',
}

/**
 * Custom dropdown replacing native <select>.
 * Native <select> popups are rendered by the OS and cannot be styled
 * (no border-radius, no shadow, no theme-aware background) — this
 * component gives full control while keeping the same value/onChange API.
 */
export default function Dropdown({
  value, onChange, options, icon, size = 'md', valueClassName, className = '',
}: Props) {
  const [open, setOpen] = useState(false)
  const wrapRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) setOpen(false)
    }
    function handleKey(e: KeyboardEvent) {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', handleClick)
    document.addEventListener('keydown', handleKey)
    return () => {
      document.removeEventListener('mousedown', handleClick)
      document.removeEventListener('keydown', handleKey)
    }
  }, [])

  const selected = options.find(o => o.value === value)

  return (
    <div ref={wrapRef} className={`relative inline-block ${className}`}>
      {/* Trigger */}
      <button
        type="button"
        onClick={() => setOpen(v => !v)}
        className={`flex items-center gap-1.5 text-xs font-mono border border-base-border
                    rounded-lg cursor-pointer transition-colors hover:border-amber/40
                    focus:outline-none focus:ring-1 focus:ring-amber/30 ${SIZE_PADDING[size]}`}
        style={{ backgroundColor: 'var(--color-surface)' }}
      >
        {icon}
        <span
          className={`truncate max-w-[160px] ${valueClassName ?? ''}`}
          style={valueClassName ? undefined : { color: 'var(--color-text)' }}
        >
          {selected?.label ?? 'Select…'}
        </span>
        <ChevronDown
          size={12}
          className={`text-gray-500 flex-shrink-0 transition-transform duration-200 ${open ? 'rotate-180' : ''}`}
        />
      </button>

      {/* Panel */}
      {open && (
        <div
          className="absolute z-20 left-0 mt-1.5 min-w-full max-w-xs rounded-xl border
                     border-base-border shadow-xl overflow-hidden overflow-y-auto
                     max-h-64 py-1 animate-dropdown-in"
          style={{ backgroundColor: 'var(--color-surface)' }}
        >
          {options.map(o => (
            <button
              key={o.value}
              type="button"
              onClick={() => { onChange(o.value); setOpen(false) }}
              className={`w-full flex items-center justify-between gap-2 px-3 py-2
                          text-xs font-mono text-left transition-colors hover:bg-amber/10
                          ${o.value === value ? 'text-amber' : ''}`}
              style={o.value === value ? undefined : { color: 'var(--color-text)' }}
            >
              <span className="truncate">{o.label}</span>
              {o.value === value && <Check size={12} className="flex-shrink-0" />}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
