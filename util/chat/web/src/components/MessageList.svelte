<script>
  import DOMPurify from 'dompurify';
  import { marked } from 'marked';
  import Avatar from './Avatar.svelte';

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

  function setReasoningOpen(index, open) {
    reasoningOpen = { ...reasoningOpen, [index]: open };
  }

  function collapseDetails(event) {
    event.currentTarget.closest('details')?.removeAttribute('open');
  }

  function toolArgument(tool) {
    try {
      const args = JSON.parse(tool.arguments || '{}');
      return args.query || args.url || '';
    } catch { return tool.arguments || ''; }
  }

  function toolPreview(tool) {
    if (!tool.result) return '';
    try {
      const parsed = JSON.parse(tool.result);
      if (parsed.results) return parsed.results.map((item) => `${item.title}\n${item.url}\n${item.snippet || ''}`).join('\n\n');
      if (parsed.content) return parsed.content;
    } catch { /* plain text result */ }
    return tool.result;
  }

  function render(text) {
    return DOMPurify.sanitize(marked.parse(text || ''));
  }
</script>

<section class="messages" bind:this={element}>
  {#if !messages.length}
    <div class="welcome"><div class="mark large"><Avatar value={assistantAvatar} alt="SparkTalk" /></div><h1>무엇을 도와드릴까요?</h1><p>연결된 모델에 메시지를 보내보세요.</p></div>
  {/if}
  {#each messages as message, index}
    <article class:mine={message.role === 'user'}>
      <div class="avatar"><Avatar value={message.role === 'user' ? userAvatar : assistantAvatar} fallback={message.role === 'user' ? 'person-blue' : 'spark'} alt={message.role === 'user' ? '나' : 'AI'} /></div>
      <div class="message-body">
        {#if message.reasoning_content}
          <details class="reasoning" open={reasoningOpen[index] ?? false} ontoggle={(event) => setReasoningOpen(index, event.currentTarget.open)}>
            <summary class:activity-pulse={running && message.activity === 'reasoning'}>생각 과정</summary>
            <div class="reasoning-text">{@html render(message.reasoning_content)}</div>
            <div class="collapse-row"><button onclick={(event) => { setReasoningOpen(index, false); collapseDetails(event); }}>↑ 생각 과정 접기</button></div>
          </details>
        {/if}
        {#if message.tool_trace?.length}
          <details class="tool-trace">
            <summary class:activity-pulse={running && message.activity === 'tool'}>{message.tool_trace.some((tool) => tool.running) ? '웹 도구 실행 중…' : `웹 도구 ${message.tool_trace.length}회`}</summary>
            <div class="tool-list">
              {#each message.tool_trace as tool}
                <div class="tool-item">
                  <div class="tool-heading"><strong>{tool.name === 'web_search' ? '웹 검색' : '페이지 읽기'}</strong><span>{toolArgument(tool)}</span></div>
                  {#if tool.running}<p class="tool-running">실행 중…</p>{:else if tool.error}<p class="tool-error">{tool.error}</p>{:else if tool.result}<pre>{toolPreview(tool)}</pre>{/if}
                </div>
              {/each}
            </div>
            <div class="collapse-row"><button onclick={collapseDetails}>↑ 웹 도구 접기</button></div>
          </details>
        {/if}
        {#if message.role === 'user' && editingMessageId === message.id}
          <div class="message-editor">
            <textarea bind:value={editInput} rows="3" onkeydown={(event) => onEditKeydown(event, message, index)}></textarea>
            <div><button onclick={onCancelEdit}>취소</button><button class="edit-submit" onclick={() => onSubmitEdit(message, index)} disabled={!editInput.trim() || editInput.trim() === message.content}>수정 후 전송</button></div>
          </div>
        {:else}
          {#if message.attachments?.length}
            <div class="image-gallery">
              {#each message.attachments as attachment}
                <a href={attachment.url} target="_blank" rel="noreferrer" title={attachment.name}><img src={attachment.url} alt={attachment.name} loading="lazy" /></a>
              {/each}
            </div>
          {/if}
          <div class="bubble prose">{@html render(message.content || (running && (index === messages.length - 1 || index === retryingIndex) ? '▍' : ''))}</div>
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
            <button onclick={() => onBeginEdit(message)} disabled={running}>✎ 수정</button>
          </div>
        {/if}
      </div>
    </article>
  {/each}
</section>
