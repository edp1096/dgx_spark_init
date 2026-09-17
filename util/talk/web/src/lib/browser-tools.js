function observationText(observation) {
 if (!observation) return '';
 const labels = {
  no_editor_or_navigation_observed: '새 입력창이나 탭 이동이 관찰되지 않음',
  navigation_observed_without_editor: '탭 이동은 관찰됐지만 입력창은 찾지 못함',
  editor_detected_but_unmatched: '입력칸은 발견했지만 해당 상품의 리뷰창으로 확인하지 못함',
  ambiguous_editors: '입력창 후보가 여러 개여서 중단',
 };
 return [labels[observation.failure], observation.before_windows ? `새 팝업 ${(observation.new_windows || []).filter(w => w.type === 'popup').length}개 · 재사용 팝업 ${(observation.reused_windows || []).filter(w => w.type === 'popup').length}개` : '', observation.open_result?.error, observation.open_result?.method === 'chrome_mouse_input' ? 'Chrome 마우스 입력 전달됨' : '', `기존 탭 ${observation.before_tabs?.length || 0}개 · 새 탭 ${observation.new_tabs?.length || 0}개 · 주소 변경 ${observation.changed_tabs?.length || 0}개`].filter(Boolean).join('\n');
}
export function browserToolPreview(result) {
  if (result.error) return [result.error, observationText(result.observation)].filter(Boolean).join('\n');
  if (Array.isArray(result.results)) {
    const lines = result.results.map(item => `${item.product || item.id || '구매후기'} · ${item.status === 'submitted' ? '등록 완료' : item.ok ? '완료' : '실패'}${item.error ? `\n${item.error}` : ''}${item.observation ? `\n${observationText(item.observation)}` : ''}`);
    if (result.not_attempted?.length) lines.push(`미실행 ${result.not_attempted.length}개`);
    return lines.join('\n\n');
  }
  if (Array.isArray(result.items)) return [`상품 ${result.items.length}개`, ...result.items.map(item => `${item.product}\n${item.summary || ''}`)].join('\n\n');
  if (Array.isArray(result.tabs)) return result.tabs.map(tab => `${tab.title}\n${tab.url}`).join('\n\n');
  if (result.text != null) return [result.product, `별점 ${result.rating}/5`, result.text, result.submitted === false ? '입력 완료 · 미등록' : ''].filter(Boolean).join('\n');
  return JSON.stringify(result, null, 2);
}
