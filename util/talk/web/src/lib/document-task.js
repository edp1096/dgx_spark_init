export function documentTaskFinished(run) {
  const task = run?.document;
	if (task?.pipeline) return task.terminal === true;
  return !!task && (run.status === 'completed' ||
    run.status === 'needs_review' && run.current >= 2 && task.phase === 'completed' && task.total > 0 && task.completed >= task.total);
}

export function documentTaskCompletionLabel(run) {
  if (run.document?.terms) return run.document.terms.unresolved || run.document.warnings?.length ? '완료 · 확인 필요' : '완료';
  return run.document?.warnings?.length || run.document?.pipeline && run.document?.quality_status !== 'reviewed' ? '완료 · 확인 필요' : '완료';
}

export function documentWarningLabel(warning) {
  if (warning.startsWith('glossary spelling requires review: ')) return warning.replace('glossary spelling requires review: ', '용어 표기 확인: ');
  if (warning.startsWith('translation language requires review: ')) return warning.replace('translation language requires review: ', '번역 어휘 확인 필요: ');
  if (warning.startsWith('term consistency unresolved: ')) return warning.replace('term consistency unresolved: ', '표기 교정 확인 필요: ');
  if (warning.startsWith('unexpectedly short translation: ')) return '원문에 비해 번역이 짧은 구간이 있습니다. 구간 결과를 확인해 주세요.';
  return warning;
}
