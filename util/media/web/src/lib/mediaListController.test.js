import test from 'node:test'
import assert from 'node:assert/strict'
import { MediaListController } from './mediaListController.js'

test('media lists own paging, sorting and persistent view preferences', () => {
  let state = {
    jobs: [{ id: '2', kind: 'image', created_at: '2026-01-02' }, { id: '1', kind: 'image', created_at: '2026-01-01' }],
    pages: { image: 1 }, pageSizes: { image: 1 }, sortOrders: { image: 'desc' },
    views: { image: 'gallery' }, mobilePanes: { image: 'create' }, tagFilters: { image: [] }, tagExclusions: { image: [] }, tagUntaggedOnly: { image: false }, tagMatchModes: { image: 'or' }
  }
  const saved = new Map()
  const controller = new MediaListController({
    storage: { setItem: (key, value) => saved.set(key, value) }, pageSizeOptions: [1, 2], imagePageSizeOptions: [1, 2],
    actions: { getState: () => state, patch: (patch) => state = { ...state, ...patch } }
  })
  assert.equal(controller.pageItems('image')[0].id, '2')
  controller.setPage('image', 2)
  assert.equal(controller.pageItems('image')[0].id, '1')
  controller.setSortOrder('image', 'asc')
  assert.equal(controller.pageItems('image')[0].id, '1')
  controller.setView('image', 'list')
  controller.setMobilePane('image', 'results')
  assert.equal(state.views.image, 'list')
  assert.equal(state.mobilePanes.image, 'results')
  assert.equal(saved.get('media-image-sort-order'), 'asc')
})

test('media lists filter multiple tags with OR and AND before paging', () => {
  let state = {
    jobs: [
      { id: 'both', kind: 'image', tags: ['인물', '야간'], created_at: '2026-01-03' },
      { id: 'portrait', kind: 'image', tags: ['인물'], created_at: '2026-01-02' },
      { id: 'night', kind: 'image', tags: ['야간'], created_at: '2026-01-01' }
    ],
    pages: { image: 2 }, pageSizes: { image: 1 }, sortOrders: { image: 'desc' },
    views: { image: 'gallery' }, mobilePanes: { image: 'results' }, tagFilters: { image: [] }, tagExclusions: { image: [] }, tagUntaggedOnly: { image: false }, tagMatchModes: { image: 'or' }
  }
  const controller = new MediaListController({
    storage: { setItem: () => {} }, pageSizeOptions: [1], imagePageSizeOptions: [1],
    actions: { getState: () => state, patch: (patch) => state = { ...state, ...patch } }
  })
  assert.deepEqual(controller.tagOptions('image'), [{ name: '야간', count: 2 }, { name: '인물', count: 2 }])
  controller.setTagFilter('image', ['인물', '야간'])
  assert.equal(state.pages.image, 1)
  assert.deepEqual(controller.ordered('image').map((job) => job.id), ['both', 'portrait', 'night'])
  controller.setTagMatchMode('image', 'and')
  assert.deepEqual(controller.ordered('image').map((job) => job.id), ['both'])
})

test('media lists keep include, exclude and untagged-only filters mutually exclusive', () => {
  let state = {
    jobs: [
      { id: 'both', kind: 'image', tags: ['인물', '야간'], created_at: '2026-01-03' },
      { id: 'portrait', kind: 'image', tags: ['인물'], created_at: '2026-01-02' },
      { id: 'empty', kind: 'image', tags: [], created_at: '2026-01-01' }
    ],
    pages: { image: 3 }, pageSizes: { image: 1 }, sortOrders: { image: 'desc' },
    views: { image: 'gallery' }, mobilePanes: { image: 'results' }, tagFilters: { image: [] },
    tagExclusions: { image: [] }, tagUntaggedOnly: { image: false }, tagMatchModes: { image: 'or' }
  }
  const controller = new MediaListController({
    storage: { setItem: () => {} }, pageSizeOptions: [1], imagePageSizeOptions: [1],
    actions: { getState: () => state, patch: (patch) => state = { ...state, ...patch } }
  })
  controller.setTagFilter('image', ['인물'])
  controller.setTagExclusions('image', ['야간'])
  assert.deepEqual(controller.ordered('image').map((job) => job.id), ['portrait'])
  assert.equal(state.pages.image, 1)
  controller.setTagUntaggedOnly('image', true)
  assert.deepEqual(state.tagFilters.image, [])
  assert.deepEqual(state.tagExclusions.image, [])
  assert.deepEqual(controller.ordered('image').map((job) => job.id), ['empty'])
  controller.setTagFilter('image', ['인물'])
  assert.equal(state.tagUntaggedOnly.image, false)
  assert.deepEqual(controller.ordered('image').map((job) => job.id), ['both', 'portrait'])
})
