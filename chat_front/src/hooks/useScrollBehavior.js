import { useEffect, useRef, useState } from 'react'

export function useScrollBehavior(messages) {
  const containerRef = useRef(null)
  const autoScrollRef = useRef(false)
  const hasOverflowRef = useRef(false)
  const timerRef = useRef(null)
  const [isScrollable, setIsScrollable] = useState(false)
  const [hasOverflow, setHasOverflow] = useState(false)

  useEffect(() => {
    const container = containerRef.current
    if (!container) return
    const recalc = () => {
      const overflow = container.scrollHeight > container.clientHeight + 1
      hasOverflowRef.current = overflow
      setHasOverflow(overflow)
      if (!overflow) setIsScrollable(false)
    }
    recalc()
    const onScroll = () => {
      if (autoScrollRef.current || !hasOverflowRef.current) return
      setIsScrollable(true)
      if (timerRef.current) window.clearTimeout(timerRef.current)
      timerRef.current = window.setTimeout(() => setIsScrollable(false), 1200)
    }
    const onResize = () => recalc()
    container.addEventListener('scroll', onScroll)
    window.addEventListener('resize', onResize)
    return () => {
      container.removeEventListener('scroll', onScroll)
      window.removeEventListener('resize', onResize)
      if (timerRef.current) window.clearTimeout(timerRef.current)
    }
  }, [])

  useEffect(() => {
    const container = containerRef.current
    if (!container) return
    const id = requestAnimationFrame(() => {
      autoScrollRef.current = true
      container.scrollTop = container.scrollHeight
      setHasOverflow(container.scrollHeight > container.clientHeight + 1)
      requestAnimationFrame(() => { autoScrollRef.current = false })
    })
    return () => cancelAnimationFrame(id)
  }, [messages])

  return { containerRef, isScrollable, hasOverflow }
}
