export function durationFromFrames(frames, fps) {
    return Math.round(Math.max(0, (Number(frames) - 1) / Math.max(1, Number(fps))) * 1000) / 1000
  }

export function framesForDuration(seconds, fps) {
    const rawFrames = Math.max(0, Number(seconds) || 0) * Math.max(1, Number(fps) || 1)
    return Math.max(9, Math.round(rawFrames / 8) * 8 + 1)
  }

export function snapDimension(value, multiple, minimum, maximum = Number.MAX_SAFE_INTEGER) {
    const numeric = Number(value)
    const fallback = Math.max(minimum, Number.isFinite(numeric) ? numeric : minimum)
    return Math.max(minimum, Math.min(maximum, Math.round(fallback / multiple) * multiple))
  }

export function formatDuration(seconds) {
    const total = Math.max(0, Number(seconds) || 0)
    const hours = Math.floor(total / 3600)
    const minutes = Math.floor((total % 3600) / 60)
    const secs = Math.round((total % 60) * 10) / 10
    const secondText = Number.isInteger(secs) ? String(secs).padStart(2, '0') : secs.toFixed(1).padStart(4, '0')
    if (hours) return `${hours}:${String(minutes).padStart(2, '0')}:${secondText}`
    return `${minutes}:${secondText}`
  }

