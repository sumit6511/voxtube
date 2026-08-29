import { useState, useEffect } from 'react'

// React 18 StrictMode double-invokes the initial mount (mount → simulated
// unmount → remount) in dev, which races with recharts' requestAnimationFrame-
// driven enter animation (via react-smooth) — the enter animation intermittently
// gets cancelled mid-flight depending on frame timing, so it sometimes skips on
// refresh. Delaying the animated subtree's first real mount past this window
// fixes it — but a bare useEffect only guarantees "after commit", not "after
// the browser has actually painted", so StrictMode's doubled commit can still
// occasionally land after the flip. A double rAF guarantees at least one real
// paint has happened first, closing that gap.
export function useMountedAfterTick() {
  const [mounted, setMounted] = useState(false)
  useEffect(() => {
    let inner = 0
    const outer = requestAnimationFrame(() => { inner = requestAnimationFrame(() => setMounted(true)) })
    return () => { cancelAnimationFrame(outer); cancelAnimationFrame(inner) }
  }, [])
  return mounted
}
