import { orderedJobs } from './generationMetrics.js'
import { clampPage, jobsForList, normalizePageSize, normalizeSortOrder } from './jobLists.js'

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

  ordered(key) {
    const state = this.state()
    return orderedJobs(this.jobs(key), state.sortOrders[key])
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
    for (const key of Object.keys(pages)) pages[key] = clampPage(pages[key], this.jobs(key).length, state.pageSizes[key])
    this.actions.patch({ pages })
  }

  setPage(key, page) {
    const state = this.state()
    this.actions.patch({ pages: { ...state.pages, [key]: clampPage(page, this.jobs(key).length, state.pageSizes[key]) } })
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
