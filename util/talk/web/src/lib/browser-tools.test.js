import test from 'node:test';
import assert from 'node:assert/strict';
import { browserToolPreview } from './browser-tools.js';
test('review batch failures show the real error, not search result fields', () => {
 const text = browserToolPreview({ok:false,results:[{id:'item-1',ok:false,error:'리뷰 입력창을 확인하지 못했습니다.'}],not_attempted:['item-2']});
 assert.match(text,/리뷰 입력창/);assert.match(text,/미실행 1개/);assert.doesNotMatch(text,/undefined/);
});

test('old review-detail tabs are not described as newly opened tabs', () => {
 const text = browserToolPreview({ok:false,error:'입력창 연결 실패',observation:{failure:'no_editor_or_navigation_observed',before_tabs:[{title:'리뷰 상세'}],new_tabs:[],changed_tabs:[]}});
 assert.match(text,/기존 탭 1개 · 새 탭 0개/);
 assert.match(text,/관찰되지 않음/);
});
