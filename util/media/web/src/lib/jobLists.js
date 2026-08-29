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

export function normalizeTagName(value) {
  return String(value || '').trim().replace(/\s+/g, ' ')
}

export function tagKey(value) {
  return normalizeTagName(value).toLocaleLowerCase()
}

export function normalizeTagSelection(tags) {
  const seen = new Set()
  return (Array.isArray(tags) ? tags : []).map(normalizeTagName).filter((tag) => {
    const key = tagKey(tag)
    if (!key || seen.has(key)) return false
    seen.add(key)
    return true
  })
}

export function filterJobsByTags(jobs, selectedTags, mode = 'or', excludedTags = [], untaggedOnly = false) {
  const selected = normalizeTagSelection(selectedTags).map(tagKey)
  const excluded = normalizeTagSelection(excludedTags).map(tagKey)
  return jobs.filter((job) => {
    const tags = new Set(normalizeTagSelection(job.tags || []).map(tagKey))
    if (untaggedOnly) return tags.size === 0
    const included = !selected.length || (mode === 'and' ? selected.every((tag) => tags.has(tag)) : selected.some((tag) => tags.has(tag)))
    return included && !excluded.some((tag) => tags.has(tag))
  })
}

export function tagCatalogForJobs(jobs) {
  const catalog = new Map()
  for (const job of jobs) {
    const counted = new Set()
    for (const raw of job.tags || []) {
      const name = normalizeTagName(raw)
      const key = tagKey(name)
      if (!key || counted.has(key)) continue
      counted.add(key)
      const current = catalog.get(key) || { name, count: 0 }
      current.count += 1
      catalog.set(key, current)
    }
  }
  return [...catalog.values()].sort((a, b) => {
    const left = a.name.toLocaleLowerCase()
    const right = b.name.toLocaleLowerCase()
    return left < right ? -1 : left > right ? 1 : 0
  })
}
