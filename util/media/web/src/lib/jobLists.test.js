import test from 'node:test'
import assert from 'node:assert/strict'
import { clampPage, filterJobsByTags, jobsForList, normalizePageSize, normalizeSortOrder, pageItems, tagCatalogForJobs } from './jobLists.js'

test('job lists filter kinds while history retains all jobs', () => {
  const jobs = [{ id: 'a', kind: 'image' }, { id: 'b', kind: 'video' }]
  assert.deepEqual(jobsForList(jobs, 'image').map((job) => job.id), ['a'])
  assert.equal(jobsForList(jobs, 'history'), jobs)
})

test('tag catalog de-duplicates case and tag filters support OR and AND', () => {
  const jobs = [
    { id: 'a', tags: ['Portrait', '야간'] },
    { id: 'b', tags: ['portrait'] },
    { id: 'c', tags: ['풍경'] }
  ]
  assert.deepEqual(tagCatalogForJobs(jobs), [
    { name: 'Portrait', count: 2 },
    { name: '야간', count: 1 },
    { name: '풍경', count: 1 }
  ])
  assert.deepEqual(filterJobsByTags(jobs, ['portrait', '풍경'], 'or').map((job) => job.id), ['a', 'b', 'c'])
  assert.deepEqual(filterJobsByTags(jobs, ['portrait', '야간'], 'and').map((job) => job.id), ['a'])
})

test('tag filters exclude one or more tags after include matching', () => {
  const jobs = [
    { id: 'a', tags: ['인물', '야간'] },
    { id: 'b', tags: ['인물', '실내'] },
    { id: 'c', tags: ['풍경'] },
    { id: 'd', tags: [] }
  ]
  assert.deepEqual(filterJobsByTags(jobs, [], 'or', ['야간']).map((job) => job.id), ['b', 'c', 'd'])
  assert.deepEqual(filterJobsByTags(jobs, ['인물'], 'or', ['야간', '실패']).map((job) => job.id), ['b'])
  assert.deepEqual(filterJobsByTags(jobs, [], 'or', ['야간', '실내']).map((job) => job.id), ['c', 'd'])
})

test('untagged-only filter returns only jobs without valid tags', () => {
  const jobs = [
    { id: 'tagged', tags: ['인물'] },
    { id: 'empty', tags: [] },
    { id: 'missing' },
    { id: 'blank', tags: ['  '] }
  ]
  assert.deepEqual(filterJobsByTags(jobs, ['인물'], 'or', [], true).map((job) => job.id), ['empty', 'missing', 'blank'])
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
