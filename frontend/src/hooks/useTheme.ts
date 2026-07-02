import { useState, useEffect } from 'react'

type Theme = 'dark' | 'light'

function applyTheme(theme: Theme) {
  const root = document.documentElement
  if (theme === 'dark') root.classList.add('dark')
  else root.classList.remove('dark')
}

export function useTheme() {
  const [theme, setThemeState] = useState<Theme>(() => {
    try { return (localStorage.getItem('vt-theme') as Theme) ?? 'dark' }
    catch { return 'dark' }
  })

  useEffect(() => {
    applyTheme(theme)
    try { localStorage.setItem('vt-theme', theme) } catch { /* ignore */ }
  }, [theme])

  function toggleTheme() {
    setThemeState(t => t === 'dark' ? 'light' : 'dark')
  }

  return { theme, toggleTheme, isDark: theme === 'dark' }
}
