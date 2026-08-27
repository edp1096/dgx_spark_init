import test from 'node:test'
import assert from 'node:assert/strict'
import { MediaListController } from './mediaListController.js'

test('media lists own paging, sorting and persistent view preferences', () => {
  let state = {
    jobs: [{ id: '2', kind: 'image', created_at: '2026-01-02' }, { id: '1', kind: 'image', created_at: '2026-01-01' }],
    pages: { image: 1 }, pageSizes: { image: 1 }, sortOrders: { image: 'desc' },
    views: { image: 'gallery' }, mobilePanes: { image: 'create' }
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
