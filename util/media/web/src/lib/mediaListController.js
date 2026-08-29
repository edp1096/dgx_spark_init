import { orderedJobs } from './generationMetrics.js'
import { clampPage, filterJobsByTags, jobsForList, normalizePageSize, normalizeSortOrder, normalizeTagSelection, tagCatalogForJobs, tagKey } from './jobLists.js'

const generatedKinds = new Set(['image', 'video', 'speech', 'recognition'])

export class MediaListController {
  constructor({ storage, pageSizeOptions, imagePageSizeOptions, actions }) {
    this.storage = storage
    this.pageSizeOptions = pageSizeOptions
    this.imagePageSizeOptions = imagePageSizeOptions
    this.actions = actions
  }

  state() { return this.actions.getState() }
  jobs(key) { return jobsForList(this.state().jobs, key) }
  sizeOptions(key) { return generatedKinds.has(key) ? this.imagePageSizeOptions : this.pageSizeOptions }

  tagOptions(key) { return tagCatalogForJobs(this.jobs(key)) }

  selectedTags(key) {
    return normalizeTagSelection(this.state().tagFilters?.[key] || [])
  }

  excludedTags(key) {
    return normalizeTagSelection(this.state().tagExclusions?.[key] || [])
  }

  untaggedOnly(key) {
    return Boolean(this.state().tagUntaggedOnly?.[key])
  }

  filtered(key) {
    const state = this.state()
    return filterJobsByTags(this.jobs(key), this.selectedTags(key), state.tagMatchModes?.[key], this.excludedTags(key), this.untaggedOnly(key))
  }

  ordered(key) {
    const state = this.state()
    return orderedJobs(this.filtered(key), state.sortOrders[key])
  }

  pageItems(key) {
    const state = this.state()
    const jobs = this.ordered(key)
    const start = (state.pages[key] - 1) * state.pageSizes[key]
    return jobs.slice(start, start + state.pageSizes[key])
  }

  clampPages() {
    const state = this.state()
    const pages = { ...state.pages }
    const tagFilters = { ...(state.tagFilters || {}) }
    const tagExclusions = { ...(state.tagExclusions || {}) }
    const tagUntaggedOnly = { ...(state.tagUntaggedOnly || {}) }
    for (const key of Object.keys(pages)) {
      const available = new Map(this.tagOptions(key).map((tag) => [tagKey(tag.name), tag.name]))
      tagFilters[key] = this.selectedTags(key).filter((tag) => available.has(tagKey(tag))).map((tag) => available.get(tagKey(tag)))
      tagExclusions[key] = this.excludedTags(key).filter((tag) => available.has(tagKey(tag))).map((tag) => available.get(tagKey(tag)))
      tagUntaggedOnly[key] = Boolean(tagUntaggedOnly[key])
      pages[key] = clampPage(pages[key], filterJobsByTags(this.jobs(key), tagFilters[key], state.tagMatchModes?.[key], tagExclusions[key], tagUntaggedOnly[key]).length, state.pageSizes[key])
    }
    this.actions.patch({ pages, tagFilters, tagExclusions, tagUntaggedOnly })
  }

  setPage(key, page) {
    const state = this.state()
    this.actions.patch({ pages: { ...state.pages, [key]: clampPage(page, this.filtered(key).length, state.pageSizes[key]) } })
  }

  setPageSize(key, pageSize) {
    const state = this.state()
    const size = normalizePageSize(pageSize, this.sizeOptions(key), state.pageSizes[key])
    this.actions.patch({ pageSizes: { ...state.pageSizes, [key]: size }, pages: { ...state.pages, [key]: 1 } })
    this.storage.setItem(`media-${key}-page-size`, String(size))
  }

  setSortOrder(key, order) {
    const state = this.state()
    const next = normalizeSortOrder(order)
    this.actions.patch({ sortOrders: { ...state.sortOrders, [key]: next }, pages: { ...state.pages, [key]: 1 } })
    this.storage.setItem(`media-${key}-sort-order`, next)
  }

  setTagFilter(key, tags) {
    const state = this.state()
    const selected = normalizeTagSelection(tags)
    const selectedKeys = new Set(selected.map(tagKey))
    this.actions.patch({
      tagFilters: { ...(state.tagFilters || {}), [key]: selected },
      tagExclusions: { ...(state.tagExclusions || {}), [key]: this.excludedTags(key).filter((tag) => !selectedKeys.has(tagKey(tag))) },
      tagUntaggedOnly: { ...(state.tagUntaggedOnly || {}), [key]: selected.length ? false : this.untaggedOnly(key) },
      pages: { ...state.pages, [key]: 1 }
    })
  }

  setTagExclusions(key, tags) {
    const state = this.state()
    const excluded = normalizeTagSelection(tags)
    const excludedKeys = new Set(excluded.map(tagKey))
    this.actions.patch({
      tagFilters: { ...(state.tagFilters || {}), [key]: this.selectedTags(key).filter((tag) => !excludedKeys.has(tagKey(tag))) },
      tagExclusions: { ...(state.tagExclusions || {}), [key]: excluded },
      tagUntaggedOnly: { ...(state.tagUntaggedOnly || {}), [key]: excluded.length ? false : this.untaggedOnly(key) },
      pages: { ...state.pages, [key]: 1 }
    })
  }

  setTagUntaggedOnly(key, enabled) {
    const state = this.state()
    const next = Boolean(enabled)
    this.actions.patch({
      tagFilters: { ...(state.tagFilters || {}), [key]: next ? [] : this.selectedTags(key) },
      tagExclusions: { ...(state.tagExclusions || {}), [key]: next ? [] : this.excludedTags(key) },
      tagUntaggedOnly: { ...(state.tagUntaggedOnly || {}), [key]: next },
      pages: { ...state.pages, [key]: 1 }
    })
  }

  setTagMatchMode(key, mode) {
    const state = this.state()
    this.actions.patch({
      tagMatchModes: { ...(state.tagMatchModes || {}), [key]: mode === 'and' ? 'and' : 'or' },
      pages: { ...state.pages, [key]: 1 }
    })
  }

  showNewest(key) {
    const state = this.state()
    this.actions.patch({ pages: { ...state.pages, [key]: 1 } })
  }

  setView(key, view) {
    const state = this.state()
    this.actions.patch({ views: { ...state.views, [key]: view } })
    this.storage.setItem(`media-${key}-view`, view)
  }

  setMobilePane(key, pane) {
    const state = this.state()
    this.actions.patch({ mobilePanes: { ...state.mobilePanes, [key]: pane } })
  }
}
