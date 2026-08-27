import test from 'node:test'
import assert from 'node:assert/strict'
import { clampPage, jobsForList, normalizePageSize, normalizeSortOrder, pageItems } from './jobLists.js'

test('job lists filter kinds while history retains all jobs', () => {
  const jobs = [{ id: 'a', kind: 'image' }, { id: 'b', kind: 'video' }]
  assert.deepEqual(jobsForList(jobs, 'image').map((job) => job.id), ['a'])
  assert.equal(jobsForList(jobs, 'history'), jobs)
})

test('pagination clamps invalid pages and uses approved sizes', () => {
  const items = Array.from({ length: 9 }, (_, index) => index)
  assert.equal(clampPage(9, items.length, 4), 3)
  assert.deepEqual(pageItems(items, 3, 4), [8])
  assert.equal(normalizePageSize('8', [4, 8, 12], 4), 8)
  assert.equal(normalizePageSize(7, [4, 8, 12], 4), 4)
  assert.equal(normalizeSortOrder('asc'), 'asc')
  assert.equal(normalizeSortOrder('invalid'), 'desc')
})
