export const resultListKeys = ['image', 'video', 'speech', 'recognition']

export function jobsForList(jobs, key) {
  return key === 'history' ? jobs : jobs.filter((job) => job.kind === key)
}

export function lastPage(total, pageSize) {
  return Math.max(1, Math.ceil(Math.max(0, total) / Math.max(1, pageSize)))
}

export function clampPage(page, total, pageSize) {
  return Math.min(Math.max(1, Number(page) || 1), lastPage(total, pageSize))
}

export function pageItems(items, page, pageSize) {
  const size = Math.max(1, Number(pageSize) || 1)
  const current = clampPage(page, items.length, size)
  const start = (current - 1) * size
  return items.slice(start, start + size)
}

export function normalizePageSize(value, allowedSizes, fallback) {
  const size = Number(value)
  return allowedSizes.includes(size) ? size : fallback
}

export function normalizeSortOrder(value) {
  return value === 'asc' ? 'asc' : 'desc'
}
