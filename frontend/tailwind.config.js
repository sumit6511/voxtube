/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  darkMode: 'class',    // toggle via <html class="dark">
  theme: {
    extend: {
      fontFamily: {
        display: ['"Syne"', 'sans-serif'],
        body:    ['"DM Sans"', 'sans-serif'],
        mono:    ['"IBM Plex Mono"', 'monospace'],
      },
      colors: {
        base:    { DEFAULT: 'var(--color-base)', surface: 'var(--color-surface)', border: 'var(--color-border)' },
        amber:   { DEFAULT: '#F59E0B', dim: '#92630A', glow: '#FCD34D' },
        pos:     '#10B981',
        neu:     '#6B7280',
        neg:     '#F43F5E',
        tox:     '#EF4444',
      },
      animation: {
        'fade-up':   'fadeUp 0.5s ease forwards',
        'pulse-dot': 'pulseDot 1.4s ease-in-out infinite',
      },
      keyframes: {
        fadeUp:   { from: { opacity: 0, transform: 'translateY(16px)' }, to: { opacity: 1, transform: 'none' } },
        pulseDot: { '0%,100%': { opacity: 0.3 }, '50%': { opacity: 1 } },
      },
    },
  },
  plugins: [],
}
