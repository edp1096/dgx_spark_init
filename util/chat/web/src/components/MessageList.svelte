<script>
  import DOMPurify from 'dompurify';
  import { marked } from 'marked';
  import Avatar from './Avatar.svelte';
  import MediaAttachments from './MediaAttachments.svelte';

  export let messages = [];
  export let running = false;
  export let retryingIndex = -1;
  export let reasoningOpen = {};
  export let editingMessageId = null;
  export let editInput = '';
  export let element;
  export let assistantAvatar = 'preset:spark';
  export let userAvatar = 'preset:person-blue';
  export let variantIndices = () => [];
  export let variantPosition = () => 0;
  export let onShowAdjacentVariant = () => {};
  export let onRetry = () => {};
  export let onEditKeydown = () => {};
  export let onCancelEdit = () => {};
  export let onSubmitEdit = () => {};
  export let onBeginEdit = () => {};
  export let onToolApproval = () => {};
  export let ttsEnabled = false;
  export let speechLoadingKey = '';
  export let speechPlayingKey = '';
  export let onSpeakReply = () => {};

  function replySpeechKey(message) {
    return `${message?.id || 'pending'}:${message?.variant_index ?? 0}`;
  }

  function setReasoningOpen(index, open) {
    reasoningOpen = { ...reasoningOpen, [index]: open };
  }

  function collapseDetails(event) {
    event.currentTarget.closest('details')?.removeAttribute('open');
  }

  function toolArgument(tool) {
    try {
      const args = JSON.parse(tool.arguments || '{}');
      if (tool.name === 'ssh_exec') return `${args.host || ''}${args.command ? ` · ${args.command}` : ''}`;
      return args.query || args.url || '';
    } catch { return tool.arguments || ''; }
  }

  function toolPreview(tool) {
    if (tool.output) return tool.output;
    if (!tool.result) return '';
    try {
      const parsed = JSON.parse(tool.result);
      if (parsed.results) return parsed.results.map((item) => `${item.title}\n${item.url}\n${item.snippet || ''}`).join('\n\n');
      if (parsed.content) return parsed.content;
      if (tool.name === 'ssh_exec') {
        const output = [parsed.stdout, parsed.stderr].filter(Boolean).join('');
        const meta = `\n\n종료 코드 ${parsed.exit_code} · ${parsed.duration_ms || 0}ms${parsed.truncated ? ' · 출력 잘림' : ''}`;
        return `${output}${meta}`.trim();
      }
    } catch { /* plain text result */ }
    return tool.result;
  }

  function sshResultMeta(tool) {
    if (tool.name !== 'ssh_exec' || !tool.result) return '';
    try {
      const parsed = JSON.parse(tool.result);
      if (parsed.exit_code === undefined) return '';
      return `종료 코드 ${parsed.exit_code} · ${parsed.duration_ms || 0}ms${parsed.truncated ? ' · 출력 잘림' : ''}`;
    } catch { return ''; }
  }

  function toolLabel(tool) {
    if (tool.name === 'web_search') return '웹 검색';
    if (tool.name === 'web_fetch') return '페이지 읽기';
    if (tool.name === 'ssh_exec') return 'SSH 실행';
    return tool.name || '도구';
  }

  function render(text) {
    return DOMPurify.sanitize(marked.parse(text || ''));
  }

  function visibleAssistantContent(text) {
    const source = text || '';
    const cleaned = source
      .replace(/<tool_call\b[^>]*>[\s\S]*?<\/tool_call>/gi, '')
      .replace(/<tool_call\b[^>]*>[\s\S]*$/gi, '')
      .trim();
    if (!cleaned && /<tool_call\b/i.test(source)) return '도구 호출 요청이 완료되지 않았습니다.';
    return cleaned;
  }
</script>

<section class="messages" bind:this={element}>
  {#if !messages.length}
    <div class="welcome"><div class="mark large"><Avatar value={assistantAvatar} alt="SparkTalk" /></div><h1>무엇을 도와드릴까요?</h1><p>연결된 모델에 메시지를 보내보세요.</p></div>
  {/if}
  {#each messages as message, index}
    <article class:mine={message.role === 'user'} class:message-failed={message.status === 'failed'} class:message-cancelled={message.status === 'cancelled'} data-message-id={message.id || ''}>
      <div class="avatar"><Avatar value={message.role === 'user' ? userAvatar : assistantAvatar} fallback={message.role === 'user' ? 'person-blue' : 'spark'} alt={message.role === 'user' ? '나' : 'AI'} /></div>
      <div class="message-body">
        {#if message.reasoning_content}
          <details class="reasoning" open={reasoningOpen[index] ?? false} ontoggle={(event) => setReasoningOpen(index, event.currentTarget.open)}>
            <summary class:activity-pulse={running && message.activity === 'reasoning'}>생각 과정</summary>
            <div class="reasoning-text">{@html render(message.reasoning_content)}</div>
            <div class="collapse-row"><button onclick={(event) => { setReasoningOpen(index, false); collapseDetails(event); }}>↑ 생각 과정 접기</button></div>
          </details>
        {/if}
        {#if message.tool_trace?.some((tool) => tool.approval_required)}
          <section class="tool-approval-panel" aria-label="SSH 명령 실행 승인">
            {#each message.tool_trace.filter((tool) => tool.approval_required) as tool}
              <div class="tool-approval" data-approval-id={tool.approval_id || ''}>
                <div class="tool-approval-title"><strong>SSH 명령 실행 승인</strong><span>{tool.host_name || tool.host}</span></div>
                {#if tool.host_key?.fingerprint}
                  <div class="tool-host-key-warning"><strong>처음 연결하는 서버</strong><span>호스트 키 지문을 확인하세요.</span><code>{tool.host_key.fingerprint}</code></div>
                {/if}
                <code>{tool.command}</code>
                {#if tool.reason}<small>{tool.reason}</small>{/if}
                {#if tool.approval_error}<p class="tool-error">{tool.approval_error}</p>{/if}
                <div class="tool-approval-actions">
                  <button onclick={() => onToolApproval(tool, 'reject')} disabled={tool.approving}>거부</button>
                  <button onclick={() => onToolApproval(tool, 'once')} disabled={tool.approving}>{tool.host_key?.fingerprint ? '키 신뢰 후 이번만' : '이번만 실행'}</button>
                  {#if tool.conversation_scope_available}<button class="approve" onclick={() => onToolApproval(tool, 'conversation')} disabled={tool.approving}>{tool.approving ? '처리 중…' : (tool.host_key?.fingerprint ? '키 신뢰·대화 허용' : '이 대화에서 허용')}</button>{/if}
                </div>
              </div>
            {/each}
          </section>
        {/if}
        {#if message.tool_trace?.length}
          <details class="tool-trace">
            <summary class:activity-pulse={running && message.activity === 'tool'}>{message.tool_trace.some((tool) => tool.running) ? '도구 실행 중…' : `도구 ${message.tool_trace.length}회`}</summary>
            <div class="tool-list">
              {#each message.tool_trace as tool}
                <div class="tool-item">
                  <div class="tool-heading"><strong>{toolLabel(tool)}</strong><span>{toolArgument(tool)}</span></div>
                  {#if tool.approval_required}<p class="tool-running">사용자 승인 대기 중…</p>
                  {:else if tool.approval_answered && !tool.approved}<p class="tool-error">사용자가 실행을 거부했습니다.</p>
                  {:else if tool.running && !tool.output}<p class="tool-running">{tool.execution_status === 'running' ? '명령 실행 중…' : '실행 준비 중…'}</p>{/if}
                  {#if toolPreview(tool)}<pre class:ssh-output={tool.name === 'ssh_exec'}>{toolPreview(tool)}</pre>{/if}
                  {#if tool.output && sshResultMeta(tool)}<small class="tool-exit-meta">{sshResultMeta(tool)}</small>{/if}
                  {#if !tool.running && tool.error}<p class="tool-error">{tool.error}</p>{/if}
                </div>
              {/each}
            </div>
            <div class="collapse-row"><button onclick={collapseDetails}>↑ 도구 접기</button></div>
          </details>
        {/if}
        {#if message.role === 'user' && editingMessageId === message.id}
          <div class="message-editor">
            <textarea bind:value={editInput} rows="3" onkeydown={(event) => onEditKeydown(event, message, index)}></textarea>
            <div><button onclick={onCancelEdit}>취소</button><button class="edit-submit" onclick={() => onSubmitEdit(message, index)} disabled={!editInput.trim() || (editInput.trim() === message.content && !['failed', 'cancelled'].includes(message.status))}>수정 후 전송</button></div>
          </div>
        {:else}
          {#if message.attachments?.length}
            <MediaAttachments attachments={message.attachments} />
          {/if}
          <div class="bubble prose">{@html render(message.role === 'assistant' ? visibleAssistantContent(message.content || (running && (index === messages.length - 1 || index === retryingIndex) ? '▍' : '')) : message.content)}</div>
        {/if}
        {#if message.status === 'failed' || message.status === 'cancelled'}
          <div class="message-status" class:cancelled={message.status === 'cancelled'}>
            <strong>{message.status === 'cancelled' ? (message.role === 'assistant' ? '불완전한 답변' : '중지됨') : (message.role === 'assistant' ? '불완전한 답변' : '실패')}</strong>
            {#if message.error}<span>{message.error}</span>{/if}
          </div>
        {/if}
        {#if message.role === 'assistant'}
          <div class="message-actions">
            {#if variantIndices(message, index).length > 1}
              <div class="variant-pager" aria-label="답변 버전 선택">
                <button onclick={() => onShowAdjacentVariant(message, index, -1)} disabled={running || variantPosition(message, index) <= 0} aria-label="이전 답변">‹</button>
                <span>{variantPosition(message, index) + 1}/{variantIndices(message, index).length}</span>
                <button onclick={() => onShowAdjacentVariant(message, index, 1)} disabled={running || variantPosition(message, index) >= variantIndices(message, index).length - 1} aria-label="다음 답변">›</button>
              </div>
            {/if}
            {#if message.content && !['failed', 'cancelled'].includes(message.status)}
              {@const speechKey = replySpeechKey(message)}
              <button class:active-speech={speechPlayingKey === speechKey} onclick={() => onSpeakReply(message)} disabled={running || !ttsEnabled || (speechLoadingKey && speechLoadingKey !== speechKey)} title={!ttsEnabled ? '설정에서 답변 음성을 활성화하세요' : 'Qwen3-TTS로 답변 읽기'}>
                {speechLoadingKey === speechKey ? '◌ 음성 생성 중' : speechPlayingKey === speechKey ? '■ 정지' : '🔊 읽기'}
              </button>
            {/if}
            <button onclick={() => onRetry(message, index)} disabled={running || !message.id}>↻ 재시도</button>
          </div>
        {:else if message.id && editingMessageId !== message.id}
          <div class="message-actions user-actions">
            {#if message.variants?.length > 1}
              <div class="variant-pager" aria-label="질문 버전 선택">
                <button onclick={() => onShowAdjacentVariant(message, index, -1)} disabled={running || message.variant_index <= 0} aria-label="이전 질문">‹</button>
                <span>{message.variant_index + 1}/{message.variants.length}</span>
                <button onclick={() => onShowAdjacentVariant(message, index, 1)} disabled={running || message.variant_index >= message.variants.length - 1} aria-label="다음 질문">›</button>
              </div>
            {/if}
            <button onclick={() => onBeginEdit(message)} disabled={running}>✎ {message.status === 'failed' || message.status === 'cancelled' ? '수정·재시도' : '수정'}</button>
          </div>
        {/if}
      </div>
    </article>
  {/each}
</section>
