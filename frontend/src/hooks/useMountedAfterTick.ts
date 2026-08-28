import { useState, useEffect } from 'react'

// React 18 StrictMode double-invokes the initial mount (mount → simulated
// unmount → remount) in dev, which races with recharts' requestAnimationFrame-
// driven enter animation (via react-smooth) — the enter animation intermittently
// gets cancelled mid-flight depending on frame timing, so it sometimes skips on
// refresh. Delaying the animated subtree's first real mount to the effect phase
// puts it on its own, un-doubled commit, so the animation reliably plays once.
export function useMountedAfterTick() {
  const [mounted, setMounted] = useState(false)
  useEffect(() => { setMounted(true) }, [])
  return mounted
}
