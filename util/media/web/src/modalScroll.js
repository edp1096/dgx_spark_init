let lockCount = 0
let previousBodyOverflow = ''
let previousHTMLOverflow = ''

export function lockModalScroll() {
  if (typeof document === 'undefined') return () => {}

  if (lockCount === 0) {
    previousBodyOverflow = document.body.style.overflow
    previousHTMLOverflow = document.documentElement.style.overflow
    document.body.style.overflow = 'hidden'
    document.documentElement.style.overflow = 'hidden'
  }
  lockCount += 1

  let released = false
  return () => {
    if (released || typeof document === 'undefined') return
    released = true
    lockCount = Math.max(0, lockCount - 1)
    if (lockCount === 0) {
      document.body.style.overflow = previousBodyOverflow
      document.documentElement.style.overflow = previousHTMLOverflow
    }
  }
}
