<script>
  import { onDestroy } from 'svelte';
  import { formatElapsed } from '../lib/performance.js';
  import InferenceMetrics from './InferenceMetrics.svelte';
  import { documentTaskFinished, documentTaskCompletionLabel, documentWarningLabel } from '../lib/document-task.js';
  export let run;
  export let sessionId;
  export let onChange = () => {};

  let busy = false, operation = '', error = '', units = [], offset = 0, showUnits = false;
  let editing = false, revised = '', reviseEffort = 'none', deleting = false, showFormats = false;
  let now = Date.now(), timer;
  const labels = { translate: '문서 번역', summarize: '문서 요약', compare: '자료 비교' };
  const states = { paused: '일시정지', running: '진행 중', completed: '완료', waiting_model: '모델 대기', needs_review: '검토 필요', failed: '실패', canceled: '취소', conversion_failed: '변환 실패' };
  const efforts = { none: '끔', off: '끔', low: '낮음', medium: '중간', high: '높음', xhigh: '높음' };
  $: task = run.document;
  $: conversions = run.conversions || [];
  $: converting = conversions.some(item => item.status === 'running');
  $: latestConversions = ['pdf', 'docx', 'hwp', 'hwpx'].map(format => conversions.find(item => item.format === format)).filter(Boolean);
  $: active = run.status === 'running';
  $: finished = documentTaskFinished(run);
  $: { clearInterval(timer); if (active || converting) { now = Date.now(); timer = setInterval(() => now = Date.now(), 1000); } }
  onDestroy(() => clearInterval(timer));
  $: elapsed = (task.elapsed_seconds || 0) + (active ? Math.max(0, (now - new Date(run.updated).getTime()) / 1000) || 0 : 0);
  $: sourceTitle = Object.values(task.source_titles || {}).join(' · ');
  $: activity = ({ waiting: '모델 응답 대기 중', retrying: '구간 재시도 준비 중', reasoning: '모델 추론 중', answer: '결과 생성 중' })[task.activity] || '처리 중';
  $: phase = ({ language_check: '번역 언어 확인 중', language_verify: '번역 어휘 교정 확인 중', terms_extract: '원문과 표기 대조 중', terms_decide: '표기 충돌 확인 중', terms_repair: '표기 교정 중', terms_verify: '교정 결과 확인 중', translate: '원문 번역 중', review: '원문 대조 검수 중', repair: '문제 문단 교정 중', synthesis: '결과 종합 중', export: '결과 파일 생성 중', convert: '문서 형식 변환 중' })[task.phase];
  $: base = `/api/document-tasks/${encodeURIComponent(run.id)}`;
  $: query = `session_id=${encodeURIComponent(sessionId)}`;
  $: draft = task.pipeline && !task.publication && !!task.draft_publication;
  $: exported = task.pipeline ? !!task.publication || draft : run.current >= 2;
  $: downloadBase = draft ? `${base}/draft` : base;
  $: finishedAt = task.finished_at || run.updated;
  $: progressDone = task.terms ? task.completed + task.terms.checked + (task.terms.language?.checked || 0) : task.pipeline ? task.completed + (task.reviewed || 0) : task.completed;
  $: progressTotal = task.terms ? task.total + (task.terms.cases?.length || 0) + (task.terms.language?.records?.length || 0) : task.pipeline ? task.total + (task.review_total || 0) : task.total;
  $: budgetReached = task.elapsed_seconds >= task.max_seconds || task.calls >= task.max_calls || task.output_tokens >= task.max_output_tokens;
  $: percent = progressTotal > 0 ? Math.min(100, Math.max(0, Math.floor(progressDone * 100 / progressTotal))) : 0;

  function currentConversion(format) { return conversions.find(item => item.format === format && item.current); }
  function hasFormat(format) { return task.formats?.includes(format) || !!currentConversion(format); }
  function fileURL(format) { return `${downloadBase}/output.${format}?${query}`; }
  function conversionElapsed(item) {
    return item.status === 'running' ? Math.max(0, (now - new Date(item.started_at).getTime()) / 1000) : item.seconds || Math.max(0, (new Date(item.finished_at) - new Date(item.started_at)) / 1000);
  }

  async function action(name, body = {}) {
    busy = true; operation = name; error = '';
    try {
      const r = await fetch(`${base}/${name}?${query}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      if (!r.ok) throw Error(await r.text());
      onChange(); return true;
    } catch (e) { error = e.message; return false; }
    finally { busy = false; operation = ''; }
  }
  async function removeTask() {
    busy = true; error = '';
    try {
      const r = await fetch(`${base}?${query}`, { method: 'DELETE' });
      if (!r.ok) throw Error(await r.text());
      onChange();
    } catch (e) { error = e.message; }
    finally { busy = false; }
  }
  async function loadUnits(next = 0) {
    error = '';
    try {
      const r = await fetch(`${base}?${query}&offset=${next}&limit=10`);
      if (!r.ok) throw Error(await r.text());
      const data = await r.json(); units = data.units || []; offset = next; showUnits = true;
    } catch (e) { error = e.message; }
  }
  async function submitRevision() {
    if (await action('revise', { instructions: revised, effort: reviseEffort })) editing = false;
  }
</script>

<section id={`document-task-${run.id}`} class="document-task" class:finished class:active aria-label={labels[task.kind] || task.kind}>
  <header class="task-heading">
    <div class="task-heading-text">
      <strong>{labels[task.kind] || task.kind}</strong>
      {#if sourceTitle}<p class="source-title" title={sourceTitle}>{sourceTitle}</p>{/if}
    </div>
    <span class="task-status" class:attention={!active && run.status !== 'canceled' && (!finished || (task.terms ? task.terms.unresolved > 0 : task.pipeline && task.quality_status !== 'reviewed'))}>{finished ? documentTaskCompletionLabel(run) : states[run.status] || run.status}</span>
  </header>

  <div class="task-progress-label">
    <span>{task.completed}/{task.total}구간{finished ? ' 완료' : ''}{#if task.image_total}<span class="image-count"> · 삽화 {task.images_ready}/{task.image_total}</span>{/if}</span>
    {#if !finished}<span>{percent}%</span>{/if}
  </div>
  {#if !finished}<progress value={progressDone} max={Math.max(1, progressTotal)} aria-label="문서 처리 진행"></progress>{/if}
  {#if task.pipeline && !task.terms}<p class="muted">검수 {task.reviewed || 0}/{task.review_total || 0}문단{#if task.unresolved} · 확인 필요 {task.unresolved}건{/if}</p>{/if}
  {#if task.terms}<p class="muted">표기 확인 {task.terms.checked}/{task.terms.cases?.length || 0}항목 · 교정 {task.terms.changed}문단{#if task.terms.unresolved} · 확인 필요 {task.terms.unresolved}곳{/if}{#if task.terms.layout_blocks} · 산문 줄바꿈 정리 {task.terms.layout_blocks}문단{/if}</p>{/if}
  {#if task.terms?.language?.records?.length}<p class="muted">번역 언어 확인 {task.terms.language.checked}/{task.terms.language.records.length}문단</p>{/if}
  {#if active}<p class="task-activity" role="status"><span class="activity-dot" aria-hidden="true"></span>{phase || activity}{#if task.active_units?.length}<span class="muted"> · {task.active_units.join(', ')}구간</span>{/if}</p>{/if}
  {#if active && task.current_chapter}<p class="chapter" title={task.current_chapter}>{task.current_chapter}</p>{/if}
  <div class="document-timing">
    <span>소요 <b>{formatElapsed(elapsed)}</b></span>
    {#if !active && finishedAt}<span>{finished ? '완료' : run.status === 'canceled' ? '취소' : run.status === 'paused' ? '일시정지' : '종료'} <time datetime={finishedAt} title={new Date(finishedAt).toLocaleString('ko-KR')}>{new Date(finishedAt).toLocaleTimeString('ko-KR', { hour12: false })}</time></span>{/if}
  </div>
  <div class="document-speeds">
    {#if active}<div class="document-performance-current"><span class="speed-label">현재 호출</span><InferenceMetrics performance={task.performance || { live: true }} /></div>{/if}
    {#if task.last_performance}<div class="document-performance-last"><span class="speed-label">{active ? '직전 호출' : '최근 호출'}</span><InferenceMetrics performance={task.last_performance} /></div>{/if}
  </div>

  {#if task.terms?.language?.records?.some(item => item.status === 'repaired' || item.status === 'uncertain')}
    <details class="quality-results"><summary>번역 어휘 확인 결과</summary>
      {#each task.terms.language.records.filter(item => item.status === 'repaired' || item.status === 'uncertain') as item}
        <p><b>{item.ordinal + 1}구간 · {item.status === 'repaired' ? '교정' : '확인 필요'}</b></p>
        {#each item.mentions as mention}
          {#if item.checks?.[mention.id]?.status === 'translate'}<p>{mention.text} → {item.checks[mention.id].after}</p>{/if}
        {/each}
        <p class="muted">{item.reason}</p>
      {/each}
    </details>
  {/if}
  {#if task.error}<p class="task-error" role="alert">{task.error}</p>{/if}
  {#if error}<p class="task-error" role="alert">{error}</p>{/if}
  {#if busy}<p class="muted" role="status">{operation === 'export' ? '문서 파일을 만드는 중입니다…' : '요청을 처리하는 중입니다…'}</p>{/if}
  <div class="actions primary-actions">
    {#if exported}
      <a class="primary" href={`${downloadBase}/output.zip?${query}`} download title="HTML 문서와 삽화를 ZIP으로 받기">{draft ? '본문 결과 받기' : '결과 받기'} <span class="file-type">ZIP</span></a>
      <button disabled={busy} aria-expanded={showFormats} aria-controls={`formats-${run.id}`} onclick={() => showFormats = !showFormats}>다른 형식 <span aria-hidden="true">{showFormats ? '⌃' : '⌄'}</span></button>
    {/if}
    {#if finished && task.kind === 'translate'}<button disabled={busy || converting} title="이름 표기, 번역 언어 혼입, 산문 줄바꿈을 확인합니다" onclick={() => action('repair_terms')}>표기·어휘 교정</button>{/if}
    {#if active}<button disabled={busy} onclick={() => action('pause')}>일시정지</button>
    {:else if !finished && run.status !== 'canceled'}<button class="primary" disabled={busy} onclick={() => action('resume', budgetReached ? { extend_budget: true } : {})}>{run.status === 'conversion_failed' ? '파일 변환 재시도' : budgetReached ? '한도 연장 후 재개' : '재개'}</button>{/if}
    {#if !finished && run.status !== 'canceled'}<button class="quiet" disabled={busy} onclick={() => action('cancel')}>취소</button>{/if}
  </div>
  {#if latestConversions.length}
    <div class="conversion-statuses" aria-label="파일 변환 상태">
      {#each latestConversions as item}
        <p class:task-error={item.status === 'failed'} role={item.status === 'failed' ? 'alert' : 'status'}>
          {item.format.toUpperCase()} {item.status === 'running' ? '변환 중' : item.status === 'failed' ? '변환 실패' : '변환 완료'} · 소요 {formatElapsed(conversionElapsed(item))}
          {#if item.finished_at} · 종료 <time datetime={item.finished_at}>{new Date(item.finished_at).toLocaleTimeString('ko-KR', { hour12: false })}</time>{/if}
          {#if item.error}<span class="conversion-error">{item.error}{#if hasFormat(item.format)} 기존 파일은 계속 받을 수 있습니다.{/if}</span>{/if}
        </p>
      {/each}
    </div>
  {/if}
  {#if draft}<p class="muted">본문과 기본 파일은 저장됐습니다. 요청한 문서 형식의 변환은 아직 끝나지 않았습니다.</p>{/if}
  {#if exported && showFormats}
    <div class="format-panel" id={`formats-${run.id}`}>
      <span class="panel-label">바로 받기</span>
      <div class="actions">{#each [['html', '단일 HTML'], ['txt', '텍스트'], ['md', 'Markdown']] as [ext, label]}<a href={`${downloadBase}/output.${ext}?${query}`} download>{label}</a>{/each}</div>
      <span class="panel-label">문서 파일 · 본문 변환</span>
      <p class="muted conversion-help">저장된 본문으로 파일만 만듭니다. 재변환에 실패해도 기존 파일은 유지됩니다.</p>
      <div class="conversion-formats">
        {#each ['docx', 'pdf', 'hwp', 'hwpx'] as format}
          <div class="conversion-format">
            <div class="actions">
              {#if hasFormat(format)}<a href={fileURL(format)} download>{format.toUpperCase()} 받기</a>{/if}
              {#if !hasFormat(format) || !draft}<button disabled={busy || active || converting} onclick={() => action(hasFormat(format) ? 'reexport' : 'export', { format })}>{format.toUpperCase()} {hasFormat(format) ? '다시 만들기' : '만들기'}</button>{/if}
            </div>
          </div>
        {/each}
      </div>
    </div>
  {/if}

  {#if task.warnings?.length || (exported && task.image_total > task.images_ready)}
    <details class="task-disclosure task-warnings">
      <summary>확인할 항목 {task.warnings?.length || 1}개</summary>
      <ul>{#each task.warnings || [] as warning}<li>{documentWarningLabel(warning)}</li>{/each}{#if !task.warnings?.length}<li>아직 받지 못한 삽화가 있습니다.</li>{/if}</ul>
      <div class="actions">
        {#if exported && !['running', 'canceled'].includes(run.status) && task.image_total > task.images_ready}<button disabled={busy} onclick={() => action('retry_assets')}>삽화 다시 받기</button>{/if}
        {#if exported && !['running', 'canceled'].includes(run.status) && task.warnings?.some(w => w.startsWith('unexpectedly short translation'))}<button disabled={busy} onclick={() => action('retry_quality')}>누락 의심 구간 재번역</button>{/if}
      </div>
    </details>
  {/if}

  {#if task.pipeline && !task.terms && finished && task.quality_status !== 'reviewed'}
    <div class="task-error"><p>검수에서 확인하지 못했거나 해결하지 못한 항목이 있습니다. 구간 결과에서 원문 근거와 지적 내용을 확인할 수 있습니다.</p><button disabled={busy} onclick={() => action('retry_quality')}>새 개정판으로 재검수</button></div>
  {/if}
  {#if task.terms?.cases?.length}
    <details class="task-disclosure term-results"><summary>표기 확인 결과</summary>
      <p class="muted">같은 대상을 가리키는 이름의 표기와 교정 결과입니다. 문서 전체의 의미 검수 결과와는 구분됩니다.</p>
      {#each task.terms.cases as item}<details><summary>{item.source} · {item.status === 'unresolved' ? '확인 필요' : item.status === 'complete' ? '확인 완료' : '진행 중'}</summary>
        {#each item.decisions || [] as decision}<p>선택한 표기: {decision.canonical || '판단 보류'}</p>{/each}
        {#each item.references.filter(ref => ref.status === 'repaired' || ref.status === 'uncertain') as ref}<div class="review-finding"><p>구간 {ref.ordinal + 1} · {ref.forms?.join(' / ') || '표기 확인 중'} · {ref.status === 'repaired' ? '교정됨' : '확인 필요'}</p><blockquote>{ref.evidence}</blockquote>{#if ref.status === 'uncertain'}<p>{ref.reason}</p>{/if}{#each ref.patches || [] as patch}{#if ref.status === 'repaired'}<p>{patch.before} → {patch.after}</p>{/if}{/each}</div>{/each}
      </details>{/each}
    </details>
  {/if}
  <details class="task-disclosure task-details">
    <summary>작업 상세</summary>
    <dl>
      <div><dt>모델</dt><dd>{task.model}</dd></div>
      <div><dt>모델 호출</dt><dd>{task.calls}회</dd></div>
      {#if task.parent}<div><dt>이전 개정판</dt><dd><a href={`#document-task-${task.parent}`}>기존 결과 보기</a></dd></div>{/if}
      {#each Object.entries(task.stage_metrics || {}) as [stage, metrics]}<div><dt>{{ translate: '번역', review: '검수', repair: '교정', terms_extract: '표기 수집', terms_decide: '표기 판정', terms_repair: '표기 교정', terms_verify: '교정 확인', language_check: '번역 언어 확인', language_verify: '번역 어휘 검증' }[stage] || stage}</dt><dd>{metrics.calls}회 · {formatElapsed(metrics.seconds)}{#if metrics.performance}<InferenceMetrics performance={metrics.performance} />{/if}</dd></div>{/each}
      {#if task.profiles}<div><dt>단계별 추론</dt><dd>번역 {efforts[task.profiles.translate.effort] || task.profiles.translate.effort} · 검수 {efforts[task.profiles.review.effort] || task.profiles.review.effort} · 교정 {efforts[task.profiles.repair.effort] || task.profiles.repair.effort}</dd></div>{:else}<div><dt>추론 설정</dt><dd>{efforts[task.effort] || task.effort}{#if task.reasoning_budget_tokens} · 예산 {task.reasoning_budget_tokens.toLocaleString('ko-KR')}토큰{/if}</dd></div>{/if}
      {#if active && task.reasoning_chars}<div><dt>현재 추론</dt><dd>{task.reasoning_chars.toLocaleString('ko-KR')}자</dd></div>{/if}
      {#if active && task.content_chars}<div><dt>현재 출력</dt><dd>{task.content_chars.toLocaleString('ko-KR')}자</dd></div>{/if}
    </dl>
    {#if Object.keys(task.glossary || {}).length}<details class="glossary"><summary>자동 생성 용어집 · {Object.keys(task.glossary).length}개</summary><dl>{#each Object.entries(task.glossary) as [source, target]}<div><dt>{source}</dt><dd>{target}</dd></div>{/each}</dl></details>{/if}
    <button aria-expanded={showUnits} disabled={busy} onclick={() => showUnits ? showUnits = false : loadUnits(0)}>구간 결과</button>
    {#if showUnits}
      <div class="unit-results">{#each units as unit}<details><summary>구간 {unit.ordinal + 1} · {states[unit.status] || unit.status}</summary>{#if unit.error}<p role="alert">{unit.error}</p>{/if}{#each unit.warnings || [] as warning}<p>{documentWarningLabel(warning)}</p>{/each}{#if unit.quality}<p class="muted">검수: {{ pending: '대기', reviewed: '대조 완료', needs_review: '확인 필요' }[unit.quality.status] || unit.quality.status} · 교정 {unit.quality.repairs || 0}회</p>{#each unit.quality.reviews || [] as review}{#each review.findings || [] as finding}<div class="review-finding"><p>{finding.uncertain ? '해석 확인 필요' : finding.severity === 'minor' ? '참고' : '의미 확인 필요'} · {finding.explanation}</p><dl><div><dt>원문 근거</dt><dd>{finding.source_quote}</dd></div><div><dt>번역 근거</dt><dd>{finding.target_quote || '대응 표현 없음'}</dd></div></dl></div>{/each}{/each}{/if}{#each Object.values(unit.output || {}) as text}<pre>{text}</pre>{/each}</details>{/each}</div>
      <div class="actions pagination"><button disabled={offset === 0} onclick={() => loadUnits(Math.max(0, offset - 10))}>이전</button><span>{offset + 1}–{Math.min(offset + units.length, task.total)} / {task.total}</span><button disabled={offset + 10 >= task.total} onclick={() => loadUnits(offset + 10)}>다음</button></div>
    {/if}
  </details>

  <details class="task-disclosure task-manage">
    <summary>작업 관리</summary>
    <div class="actions management-actions">
      <button disabled={busy} onclick={() => { editing = !editing; deleting = false; revised = task.instructions; reviseEffort = task.effort; }}>지침 변경</button>
      {#if !active}<button class="danger" disabled={busy} onclick={() => { deleting = !deleting; editing = false; }}>작업 삭제</button>{/if}
    </div>
    {#if deleting}<div class="confirmation"><p>이 작업의 기록과 결과 파일을 삭제합니다.</p><div class="actions"><button class="danger" disabled={busy} onclick={removeTask}>삭제</button><button disabled={busy} onclick={() => deleting = false}>돌아가기</button></div></div>{/if}
    {#if editing}
      <div class="revision-form">
        <label>새 작업 지침<textarea bind:value={revised} rows="4"></textarea></label>
        <label>문서 작업 추론<select bind:value={reviseEffort}><option value="none">끔</option><option value="low">낮음</option><option value="medium">중간</option><option value="xhigh">높음</option><option value="inherit">현재 채팅 설정</option></select></label>
        <p>{active ? '진행 중인 작업은 일시정지합니다. ' : ''}기존 결과를 보존하고 새 지침으로 별도 작업을 시작합니다.</p>
        <div class="actions"><button class="primary" disabled={busy || !revised.trim()} onclick={submitRevision}>새 지침으로 시작</button><button disabled={busy} onclick={() => editing = false}>돌아가기</button></div>
      </div>
    {/if}
  </details>
</section>

<style>
  .document-task { --task-bg:#151922; --task-line:#303744; --task-muted:#a0aaba; --task-hover:#242c3b; --task-accent:#9ab8ff; --task-primary:#476bd2; --task-success:#96d9ba; --task-success-bg:#18352b; --task-warning:#e8bf78; --task-warning-bg:#32291c; --muted:var(--task-muted); min-width:0; padding:16px; border:1px solid var(--task-line); border-radius:12px; background:var(--task-bg); font-size:13px; line-height:1.5; }
  :global(html[data-theme="light"]) .document-task { --task-bg:#fff; --task-line:#dce2eb; --task-muted:#657287; --task-hover:#edf1f8; --task-accent:#365fa8; --task-primary:#385ebc; --task-success:#24694e; --task-success-bg:#e8f4ed; --task-warning:#856020; --task-warning-bg:#fcf3e2; }
  .task-heading { display:flex; align-items:flex-start; justify-content:space-between; gap:12px; padding:0; border:0; background:transparent; }
  .task-heading-text { min-width:0; }
  strong { font-size:14px; font-weight:600; }
  .source-title, .chapter { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:var(--task-muted); margin:3px 0 0; font-size:12px; }
  .task-status { flex-shrink:0; padding:3px 9px; border-radius:6px; font-size:11px; color:var(--task-muted); background:var(--task-hover); }
  .finished .task-status { color:var(--task-success); background:var(--task-success-bg); }
  .active .task-status { color:var(--task-accent); }
  .task-status.attention { color:var(--task-warning); background:var(--task-warning-bg); }
  .task-progress-label { display:flex; justify-content:space-between; gap:8px; margin-top:12px; font-size:12px; font-variant-numeric:tabular-nums; }
  .image-count, .muted { color:var(--task-muted); }
  progress { display:block; appearance:none; width:100%; height:5px; margin:7px 0 0; border:0; border-radius:8px; overflow:hidden; background:var(--task-line); accent-color:var(--task-accent); }
  progress::-webkit-progress-bar { background:var(--task-line); }
  progress::-webkit-progress-value { background:var(--task-accent); border-radius:8px; }
  progress::-moz-progress-bar { background:var(--task-accent); border-radius:8px; }
  .task-activity { margin:10px 0 0; overflow-wrap:anywhere; }
  .activity-dot { display:inline-block; width:6px; height:6px; margin:0 7px 2px 0; border-radius:50%; background:var(--task-accent); }
  .document-timing { display:flex; flex-wrap:wrap; gap:3px 16px; margin-top:9px; color:var(--task-muted); font-size:12px; font-variant-numeric:tabular-nums; }
  .document-timing b, time { color:inherit; font-weight:500; }
  .document-speeds { display:grid; gap:3px; margin-top:5px; }
  .document-performance-current, .document-performance-last { display:flex; flex-wrap:wrap; align-items:baseline; gap:3px 12px; }
  .speed-label { color:var(--task-muted); font-size:11px; }
  .document-speeds :global(.inference-metrics) { margin:0; font-size:12px; }
  .actions { display:flex; flex-wrap:wrap; align-items:center; gap:8px; }
  .primary-actions { margin-top:14px; }
  button, a { display:inline-flex; align-items:center; justify-content:center; gap:7px; min-height:36px; max-width:100%; padding:7px 12px; border:1px solid var(--task-line); border-radius:7px; background:transparent; color:inherit; font:inherit; font-size:12px; text-decoration:none; text-align:center; }
  button:hover:not(:disabled), a:hover { background:var(--task-hover); }
  .primary { border-color:var(--task-primary); background:var(--task-primary); color:#fff; font-weight:500; }
  .primary:hover:not(:disabled) { background:var(--task-primary); filter:brightness(1.12); }
  .quiet { border-color:transparent; color:var(--task-muted); }
  .file-type { font-size:10px; opacity:.85; }
  .danger { color:#e8959c; }
  :global(html[data-theme="light"]) .danger { color:#b33c4a; }
  button:disabled { opacity:.45; cursor:default; }
  button:focus-visible, a:focus-visible, summary:focus-visible, textarea:focus-visible, select:focus-visible { outline:2px solid var(--task-accent); outline-offset:3px; }
  .conversion-statuses { margin-top:10px; color:var(--task-muted); font-size:12px; }
  .conversion-statuses p { margin:4px 0; }
  .conversion-error { display:block; overflow-wrap:anywhere; }
  .conversion-help { margin:0; font-size:12px; }
  .conversion-formats { display:grid; gap:10px; }
  .format-panel, .confirmation, .revision-form { display:grid; gap:8px; margin-top:10px; padding:12px; border:1px solid var(--task-line); border-radius:8px; }
  .panel-label { color:var(--task-muted); font-size:11px; }
  .panel-label:not(:first-child) { margin-top:4px; }
  .task-disclosure { border-top:1px solid var(--task-line); margin-top:12px; }
  .task-disclosure + .task-disclosure { margin-top:0; }
  summary { cursor:pointer; padding:10px 0; color:var(--task-muted); font-size:12px; }
  .task-warnings > summary { color:var(--task-warning); }
  .task-disclosure[open] { padding-bottom:12px; }
  .task-manage:not([open]) > summary { padding-bottom:0; }
  .task-error { color:var(--task-warning); background:var(--task-warning-bg); padding:10px; border-radius:7px; }
  p, li, dd, dt { overflow-wrap:anywhere; }
  ul { margin:0 0 10px; padding-left:18px; }
  li + li { margin-top:5px; }
  dl { display:grid; gap:6px; margin:0 0 12px; }
  dl > div { display:grid; grid-template-columns:minmax(70px,auto) minmax(0,1fr); gap:16px; }
  dt { color:var(--task-muted); } dd { margin:0; }
  .glossary { margin-bottom:10px; }
  .glossary dl { max-height:240px; overflow:auto; }
  .unit-results { margin-top:8px; }
  pre { white-space:pre-wrap; overflow-wrap:anywhere; max-height:320px; overflow:auto; font:inherit; }
  .pagination { margin-top:8px; color:var(--task-muted); }
  label { display:grid; gap:5px; }
  textarea, select { width:100%; min-width:0; padding:8px; border:1px solid var(--task-line); border-radius:6px; color:inherit; background:var(--task-bg); }
  textarea { resize:vertical; }
  .revision-form p, .confirmation p { margin:0; color:var(--task-muted); font-size:12px; }
  @media (max-width:520px) { .document-task { padding:12px; } button, a { min-height:40px; } .document-speeds :global(.inference-metrics) { gap:4px 10px; } .document-performance-current, .document-performance-last { gap:2px 8px; } .task-heading { gap:8px; } .image-count { display:inline-block; } }
</style>
