<script>
  import { isGenerationTool, generationProgressLabel } from '../lib/generation-progress.js';
  import { browserToolPreview } from '../lib/browser-tools.js';
  import { onDestroy, tick } from 'svelte';
  import DOMPurify from 'dompurify';
  import 'katex/dist/katex.min.css';
  import { stripInternalEvidence } from '../lib/internal-evidence.js';
  import { parseMarkdown } from '../lib/markdown.js';
  import { markdownView } from '../lib/markdown-view.js';
  import { copyText, replyPlainText } from '../lib/clipboard.js';
  import { artifactsFromMessage } from '../lib/artifacts.js';
  import Avatar from './Avatar.svelte';
  import UserInput from './UserInput.svelte';
  import SpeechPreview from './SpeechPreview.svelte';
  import MediaAttachments from './MediaAttachments.svelte';
  import InferenceMetrics from './InferenceMetrics.svelte';
  import { initialMessageStart, messageWindowAround, shiftedMessageWindow } from '../lib/message-window.js';

  export let messages = [];
  export let messageSessionId = '';
  export let running = false;
  export let retryingIndex = -1;
  export let reasoningOpen = {};
  export let editingMessageId = null;
  export let editInput = '';
  export let element;
  export let assistantName = 'SparkTalk';
  export let userName = '나';
  export let assistantAvatar = 'preset:spark';
  export let userAvatar = 'preset:person-blue';
  export let onViewAvatar = () => {};
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
  export let speechOmitParentheticals = true;
  export let speechLoadingKey = '';
  export let speechPlayingKey = '';
  export let onSpeakReply = () => {};
  export let onOpenArtifact = () => {};
  export let onRemember = () => {};
  export let rememberingId = 0;
  export let rememberedIds = [];

  let windowSessionId = null;
  let windowMessageCount = 0;
  let visibleStart = 0;
  let visibleEnd = 0;
  let visibleMessages = [];
  let topSpacerHeight = 0;
  let bottomSpacerHeight = 0;
  let heightCache = {};
  let averageMessageHeight = 220;
  let shiftingWindow = false;
  let scrollFrame = 0;
  let armFrame = 0;
  let windowReady = false;
  let copyFeedback = null;
  let copyFeedbackTimer;
  let copyRequest = 0;

  function armWindowScrolling() {
    windowReady = false;
    cancelAnimationFrame(armFrame);
    armFrame = requestAnimationFrame(() => {
      armFrame = requestAnimationFrame(() => { windowReady = true; });
    });
  }

  $: if (messageSessionId !== windowSessionId) {
    windowSessionId = messageSessionId;
    windowMessageCount = messages.length;
    heightCache = {};
    averageMessageHeight = 220;
    visibleStart = initialMessageStart(messages.length);
    visibleEnd = messages.length;
    armWindowScrolling();
  } else if (messages.length !== windowMessageCount) {
    const previousCount = windowMessageCount;
    windowMessageCount = messages.length;
    if (previousCount === 0 && messages.length > 0) {
      visibleStart = initialMessageStart(messages.length);
      visibleEnd = messages.length;
      armWindowScrolling();
    } else {
      if (visibleEnd >= previousCount) visibleEnd = messages.length;
      else visibleEnd = Math.min(visibleEnd, messages.length);
      visibleStart = Math.min(visibleStart, Math.max(0, visibleEnd - 1));
    }
  }
  $: visibleMessages = messages.slice(visibleStart, visibleEnd);
  $: topSpacerHeight = estimatedRangeHeight(0, visibleStart);
  $: bottomSpacerHeight = estimatedRangeHeight(visibleEnd, messages.length);

  function messageKey(message, index) {
    return message?.render_key || (message?.id ? `id:${message.id}` : `pending:${messageSessionId}:${index}:${message?.role || ''}`);
  }

  function estimatedRangeHeight(start, end) {
    let result = 0;
    for (let index = start; index < end; index += 1) {
      result += heightCache[messageKey(messages[index], index)] || averageMessageHeight;
    }
    return result;
  }

  function measureRenderedMessages() {
    if (!element) return;
    const measured = { ...heightCache };
    const samples = [];
    for (const article of element.querySelectorAll('article[data-message-index]')) {
      const index = Number(article.dataset.messageIndex);
      const style = getComputedStyle(article);
      const height = article.getBoundingClientRect().height
        + Number.parseFloat(style.marginTop || 0) + Number.parseFloat(style.marginBottom || 0);
      if (height > 0) {
        measured[messageKey(messages[index], index)] = height;
        samples.push(Math.min(height, 800));
      }
    }
    if (samples.length) averageMessageHeight = Math.max(100, samples.reduce((sum, value) => sum + value, 0) / samples.length);
    heightCache = measured;
  }

  function viewportAnchor() {
    if (!element) return null;
    const paneTop = element.getBoundingClientRect().top;
    const articles = [...element.querySelectorAll('article[data-message-index]')];
    const article = articles.find((item) => item.getBoundingClientRect().bottom > paneTop) || articles[0];
    if (!article) return null;
    return { index: Number(article.dataset.messageIndex), offset: article.getBoundingClientRect().top - paneTop };
  }

  async function setMessageWindow(next, preserveAnchor = true) {
    if (!next || shiftingWindow || (next.start === visibleStart && next.end === visibleEnd)) return;
    shiftingWindow = true;
    measureRenderedMessages();
    const anchor = preserveAnchor ? viewportAnchor() : null;
    visibleStart = next.start;
    visibleEnd = next.end;
    await tick();
    if (anchor && anchor.index >= visibleStart && anchor.index < visibleEnd) {
      const article = element?.querySelector(`article[data-message-index="${anchor.index}"]`);
      if (article) {
        const paneTop = element.getBoundingClientRect().top;
        const behavior = element.style.scrollBehavior;
        element.style.scrollBehavior = 'auto';
        element.scrollTop += article.getBoundingClientRect().top - paneTop - anchor.offset;
        element.style.scrollBehavior = behavior;
      }
    }
    shiftingWindow = false;
  }

  function updateWindowForScroll() {
    if (!element || shiftingWindow || !windowReady) return;
    const preload = Math.max(500, element.clientHeight * 0.75);
    if (visibleStart > 0 && element.scrollTop < topSpacerHeight + preload) {
      setMessageWindow(shiftedMessageWindow(messages.length, visibleStart, visibleEnd, 'previous'));
      return;
    }
    const renderedBottom = element.scrollHeight - bottomSpacerHeight;
    if (visibleEnd < messages.length && element.scrollTop + element.clientHeight > renderedBottom - preload) {
      setMessageWindow(shiftedMessageWindow(messages.length, visibleStart, visibleEnd, 'next'));
    }
  }

  function handleScroll() {
    if (scrollFrame) return;
    scrollFrame = requestAnimationFrame(() => {
      scrollFrame = 0;
      updateWindowForScroll();
    });
  }

  export async function revealMessage(messageId) {
    const targetIndex = messages.findIndex((message) => String(message.id) === String(messageId));
    if (targetIndex < 0) return false;
    if (targetIndex < visibleStart || targetIndex >= visibleEnd) {
      await setMessageWindow(messageWindowAround(messages.length, targetIndex), false);
    }
    element?.querySelector(`[data-message-id="${CSS.escape(String(messageId))}"]`)?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    return true;
  }

  onDestroy(() => {
    if (scrollFrame) cancelAnimationFrame(scrollFrame);
    if (armFrame) cancelAnimationFrame(armFrame);
    clearTimeout(copyFeedbackTimer);
    copyRequest += 1;
  });

  function replySpeechKey(message) {
    return `${message?.id || 'pending'}:${message?.variant_index ?? 0}`;
  }

  function setReasoningOpen(index, open) {
    reasoningOpen = { ...reasoningOpen, [index]: open };
  }

  function collapseDetails(event) {
    event.currentTarget.closest('details')?.removeAttribute('open');
  }

  // Keep old persisted tool events readable after the generic image-tool rename.
  function isImageGenerateTool(name) {
    return name === 'image_generate' || name === 'krea_image';
  }

  function isImageCapabilitiesTool(name) {
    return name === 'image_capabilities' || name === 'krea_capabilities';
  }

  function imageReferenceGroups(tool) {
    try {const groups=JSON.parse(tool.result || '{}').reference_groups;return Array.isArray(groups)?groups:[];} catch{return [];}
  }
  function imageToolInputs(tool) {
    if (!isImageGenerateTool(tool.name)) return [];
    try { const inputs = JSON.parse(tool.result || '{}').input_images; return Array.isArray(inputs) ? inputs : []; }
    catch { return []; }
  }

  function toolArgument(tool) {
    try {
      const args = JSON.parse(tool.arguments || '{}');
      if (tool.name === 'ssh_download') return `${args.host || ''} · ${args.path || ''}`;
      if (tool.name === 'ssh_exec') return `${args.host || ''}${args.command ? ` · ${args.command}` : ''}`;
      if (tool.name === 'memory_propose') return args.title || '';
      if (tool.name === 'memory_manage') return `${memoryActionLabel(args.action)}${args.title ? ` · ${args.title}` : args.query ? ` · ${args.query}` : args.memory_id ? ` · #${args.memory_id}` : ''}`;
		if (tool.name === 'knowledge_import') return args.action === 'list_collections' ? '보관함 목록' : `${args.collection || '보관함'} · ${args.urls?.length || 0}개`;
      if (isImageGenerateTool(tool.name)) return `${args.operation || 'generate'}${args.prompt ? ` · ${args.prompt}` : ''}`;
      return args.query || args.url || '';
    } catch { return tool.arguments || ''; }
  }

  function toolPreview(tool) {
    if (tool.output) return tool.output;
    if (!tool.result) return '';
    try {
      const parsed = JSON.parse(tool.result);
      if (tool.name === 'browser_reviews' || tool.name === 'browser') return browserToolPreview(parsed);
      if (tool.name === 'memory_manage') {
        if (parsed.memories) return parsed.memories.length
          ? parsed.memories.map((item) => `#${item.id} · ${memoryKindLabel(item.kind)} · ${memoryPriorityLabel(item.priority)} · ${item.enabled ? '사용' : '중지'}\n${item.title || '제목 없음'}\n${item.content}`).join('\n\n')
          : '조건에 맞는 기억 없음';
        const item = parsed.memory || parsed.deleted;
        if (item) return `#${item.id} · ${item.title || '제목 없음'}\n${item.content}`;
      }
		if (tool.name === 'knowledge_import') {
			if (parsed.collections) return parsed.collections.map((item) => `${item.name} · 자료 ${item.documents}개`).join('\n');
			if (parsed.results) return parsed.results.map((item) => `${item.status === 'failed' ? '실패' : item.duplicate ? '이미 있음' : '추가'} · ${item.document?.title || item.url}${item.error ? `\n${item.error}` : ''}`).join('\n\n');
		}
      if (parsed.results) return parsed.results.map((item) => `${item.title}\n${item.url}\n${item.snippet || ''}`).join('\n\n');
		if (tool.name === 'web_collect') {
			const publication = parsed.publication ? `전자책 ${parsed.publication.page_count}쪽 · ${parsed.publication.adapter} 어댑터` : '';
			return [parsed.content, publication].filter(Boolean).join('\n\n') || `${parsed.title || '페이지'} · ${parsed.method || 'collector'}`;
		}
      if (parsed.content) return parsed.content;
      if ((tool.name === 'media_import' || tool.name === 'ssh_download') && parsed.attachment) return `${parsed.attachment.name} · ${(parsed.attachment.size / 1024 / 1024).toFixed(1)} MB`;
      if (isImageCapabilitiesTool(tool.name)) return `작업 ${parsed.operations?.length || 0}개 · 사용자 LoRA ${parsed.user_loras?.length || 0}개`;
      if ((isImageGenerateTool(tool.name) || tool.name === 'document_generate') && parsed.attachments) return [...parsed.attachments.map((item) => item.name), ...(parsed.warning ? ['PDF를 생성하지 못했습니다. 원본 파일을 내려받을 수 있습니다.'] : [])].join('\n');
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
		if (tool.name === 'web_collect') return '브라우저 수집';
    if (tool.name === 'browser_reviews') return '구매후기';
    if (tool.name === 'browser') return '브라우저';
    if (tool.name === 'ssh_exec') return 'SSH 실행';
    if (tool.name === 'ssh_download') return '파일 가져오기';
    if (tool.name === 'document_generate') return '문서 생성';
    if (tool.name === 'media_import') return '미디어 가져오기';
    if (tool.name === 'video_generate') return '영상 생성';
    if (tool.name === 'memory_propose') return '기억 제안';
    if (tool.name === 'memory_manage') return '기억 관리';
		if (tool.name === 'knowledge_import') return '지식 가져오기';
    if (isImageCapabilitiesTool(tool.name)) return '이미지 기능 확인';
    if (isImageGenerateTool(tool.name)) return '이미지 생성';
    return tool.name || '도구';
  }

  function toolRunningLabel(tool) {
    if (isGenerationTool(tool)) return generationProgressLabel(tool);
    if (tool.name === 'document_generate') return '문서와 PDF 생성 중…';
    if (tool.name === 'media_import') return '미디어 다운로드·분석 준비 중…';
    if (tool.name === 'memory_manage') return '기억 검색·변경 준비 중…';
		if (tool.name === 'knowledge_import') return '지식 자료 확인·색인 중…';
    if (isImageCapabilitiesTool(tool.name)) return '이미지 모듈·LoRA 확인 중…';
    if (isImageGenerateTool(tool.name)) return '이미지 생성·편집 중…';
    return tool.execution_status === 'running' ? '명령 실행 중…' : '실행 준비 중…';
  }

  function memoryKindLabel(kind) {
    return kind === 'user' ? '항상 참고' : '관련 있을 때 참고';
  }

  function memoryPriorityLabel(priority) {
    return priority === 'reference' ? '참고' : '우선 적용';
  }

  function memoryActionLabel(action) {
    return ({ search: '검색', create: '추가', update: '수정', delete: '삭제' })[action] || '관리';
  }

  function memoryApprovalTitle(action) {
    return `기억 ${memoryActionLabel(action)} 승인`;
  }

  function memoryApprovalButton(action) {
    if (action === 'delete') return '삭제 승인';
    if (action === 'update') return '변경 승인';
    return '기억에 저장';
  }

  function render(text) {
    return DOMPurify.sanitize(parseMarkdown(text));
  }

  function visibleAssistantContent(text) {
    const source = stripInternalEvidence(text);
    const cleaned = source
      .replace(/<tool_call\b[^>]*>[\s\S]*?<\/tool_call>/gi, '')
      .replace(/<tool_call\b[^>]*>[\s\S]*$/gi, '')
      .trim();
    if (!cleaned && /<tool_call\b/i.test(source)) return '도구 호출 요청이 완료되지 않았습니다.';
    return cleaned;
  }

  async function copyReply(message, index, format) {
    const request = ++copyRequest;
    const key = `${messageKey(message, index)}:${message.variant_index ?? 0}`;
    const source = visibleAssistantContent(message.content);
    clearTimeout(copyFeedbackTimer);
    copyFeedback = null;
    let text;
    try {
      await copyText(format === 'markdown' ? source : replyPlainText(render(source)));
      text = format === 'markdown' ? 'Markdown 복사됨' : '일반 텍스트 복사됨';
    } catch {
      text = '복사하지 못했습니다. 답변을 선택해 직접 복사해 주세요.';
    }
    if (request !== copyRequest) return;
    copyFeedback = { key, text };
    copyFeedbackTimer = setTimeout(() => { copyFeedback = null; }, 3000);
  }

  async function copyCode(button, source) {
    const previous = button.textContent;
    try {
      await copyText(source);
      button.textContent = '복사됨';
    } catch {
      button.textContent = '복사 실패';
    }
    setTimeout(() => { if (button.isConnected) button.textContent = previous; }, 1200);
  }

  function handleMessageClick(event) {
    const copyButton = event.target.closest?.('[data-code-copy]');
    if (copyButton) {
      const source = copyButton.closest('[data-code-card]')?.querySelector('code')?.textContent || '';
      copyCode(copyButton, source);
      return;
    }
    const toggleButton = event.target.closest?.('[data-code-toggle]');
    const collapseButton = event.target.closest?.('[data-code-collapse]');
    if (!toggleButton && !collapseButton) return;
    const card = (toggleButton || collapseButton).closest('[data-code-card]');
    if (!card) return;
    const expanded = !collapseButton && !card.classList.contains('expanded');
    card.classList.toggle('expanded', expanded);
    const headerToggle = card.querySelector('[data-code-toggle]');
    if (headerToggle) {
      headerToggle.textContent = expanded ? '접기' : '전체 보기';
      headerToggle.setAttribute('aria-expanded', String(expanded));
    }
    if (!expanded) {
      card.scrollIntoView({ block: collapseButton ? 'start' : 'nearest', behavior: 'instant' });
      if (collapseButton) headerToggle?.focus({ preventScroll: true });
    }
  }

  function codeCardActions(node) {
    node.addEventListener('click', handleMessageClick);
    return { destroy: () => node.removeEventListener('click', handleMessageClick) };
  }
</script>

<section class="messages" bind:this={element} use:codeCardActions onscroll={handleScroll}>
  {#if !messages.length}
    <div class="welcome"><button type="button" class="mark large profile-avatar" aria-label="AI 아바타 크게 보기" title="AI 아바타 크게 보기" onclick={() => onViewAvatar('assistant')}><Avatar value={assistantAvatar} alt={assistantName} /></button><h1>무엇을 도와드릴까요?</h1><p>{assistantName}에게 메시지를 보내세요.</p></div>
  {/if}
  {#if topSpacerHeight > 0}<div class="message-virtual-spacer" style:height={`${topSpacerHeight}px`} aria-hidden="true"></div>{/if}
  {#each visibleMessages as message, offset (messageKey(message, visibleStart + offset))}
    {@const index = visibleStart + offset}
    {@const messageArtifacts = artifactsFromMessage(message, index)}
    <article class:mine={message.role === 'user'} class:message-failed={message.status === 'failed'} class:message-cancelled={message.status === 'cancelled'} data-message-id={message.id || ''} data-message-index={index}>
      <button type="button" class="avatar profile-avatar" aria-label={message.role === 'user' ? '사용자 아바타 크게 보기' : 'AI 아바타 크게 보기'} title={message.role === 'user' ? '사용자 아바타 크게 보기' : 'AI 아바타 크게 보기'} onclick={() => onViewAvatar(message.role)}><Avatar value={message.role === 'user' ? userAvatar : assistantAvatar} fallback={message.role === 'user' ? 'person-blue' : 'spark'} alt={message.role === 'user' ? userName : assistantName} /></button>
      <div class="message-body">
        <div class="speaker-name">{message.role === 'user' ? userName : assistantName}</div>
        {#if message.reasoning_content}
          <section class="reasoning">
            <button type="button" class="reasoning-toggle" aria-expanded={reasoningOpen[index] ?? false} aria-controls={`reasoning-${index}`} onclick={() => setReasoningOpen(index, !(reasoningOpen[index] ?? false))}><span aria-hidden="true">{reasoningOpen[index] ? '▾' : '▸'}</span> <span class="activity-label" class:activity-scanner={running && message.activity === 'reasoning'}>생각 과정</span></button>
            <div id={`reasoning-${index}`} hidden={!(reasoningOpen[index] ?? false)}>
              <div class="reasoning-text prose" use:markdownView={render(message.reasoning_content)}></div>
              <div class="collapse-row"><button onclick={() => setReasoningOpen(index, false)}>↑ 생각 과정 접기</button></div>
            </div>
          </section>
        {/if}
        {#if message.tool_trace?.some((tool) => tool.approval_required)}
          <section class="tool-approval-panel" aria-label="도구 실행 승인">
            {#each message.tool_trace.filter((tool) => tool.approval_required) as tool}
              <div class="tool-approval" data-approval-id={tool.approval_id || ''}>
                {#if tool.approval_kind === 'memory'}
                  <div class="tool-approval-title"><strong>기억 저장 승인</strong><span>{tool.kind === 'user' ? '항상 참고' : '관련 있을 때 참고'} · {memoryPriorityLabel(tool.priority)}</span></div>
                  <strong>{tool.title}</strong><p class="memory-proposal-content">{tool.content}</p>
                  {#if tool.approval_error}<p class="tool-error">{tool.approval_error}</p>{/if}
                  <div class="tool-approval-actions"><button onclick={() => onToolApproval(tool, 'reject')} disabled={tool.approving}>거부</button><button class="approve" onclick={() => onToolApproval(tool, 'once')} disabled={tool.approving}>{tool.approving ? '저장 중…' : '기억에 저장'}</button></div>
                {:else if tool.approval_kind === 'memory_manage'}
                  <div class="tool-approval-title"><strong>{memoryApprovalTitle(tool.action)}</strong><span>{tool.memory_id ? `#${tool.memory_id}` : '새 항목'} · {memoryKindLabel(tool.kind)} · {memoryPriorityLabel(tool.priority)}</span></div>
                  {#if tool.action === 'update'}
                    <div class="memory-change-grid">
                      <section><small>현재 · {memoryKindLabel(tool.before_kind)} · {memoryPriorityLabel(tool.before_priority)} · {tool.before_enabled ? '사용' : '중지'}</small><strong>{tool.before_title || '제목 없는 기억'}</strong><p class="memory-proposal-content">{tool.before_content}</p></section>
                      <section><small>변경 후 · {memoryKindLabel(tool.kind)} · {memoryPriorityLabel(tool.priority)} · {tool.enabled ? '사용' : '중지'}</small><strong>{tool.title || '제목 없는 기억'}</strong><p class="memory-proposal-content">{tool.content}</p></section>
                    </div>
                  {:else}
                    <strong>{tool.title || '제목 없는 기억'}</strong><p class="memory-proposal-content">{tool.content}</p>
                  {/if}
                  {#if tool.approval_error}<p class="tool-error">{tool.approval_error}</p>{/if}
                  <div class="tool-approval-actions"><button onclick={() => onToolApproval(tool, 'reject')} disabled={tool.approving}>거부</button><button class="approve" class:danger={tool.action === 'delete'} onclick={() => onToolApproval(tool, 'once')} disabled={tool.approving}>{tool.approving ? '처리 중…' : memoryApprovalButton(tool.action)}</button></div>
				{:else if tool.approval_kind === 'browser_reviews'}
                  <div class="tool-approval-title"><strong>구매후기 등록</strong><span>{tool.reviews?.length || 0}개</span></div>
                  {#each tool.reviews || [] as review}
                    <section><strong>{review.product}</strong><small> · 별점 {review.rating}/5</small><p class="memory-proposal-content">{review.text}</p></section>
                  {/each}
                  <small>승인한 내용 그대로 Chrome에서 등록합니다. 내용을 바꾸려면 취소 후 수정해 주세요.</small>
                  {#if tool.approval_error}<p class="tool-error">{tool.approval_error}</p>{/if}
                  <div class="tool-approval-actions"><button onclick={() => onToolApproval(tool, 'reject')} disabled={tool.approving}>취소</button><button class="approve" onclick={() => onToolApproval(tool, 'once')} disabled={tool.approving}>{tool.approving ? '처리 중…' : '이대로 등록'}</button></div>
                {:else if tool.approval_kind === 'knowledge_import'}
					<div class="tool-approval-title"><strong>지식 가져오기 승인</strong><span>{tool.collection_name} · {tool.urls?.length || 0}개</span></div>
					<div class="knowledge-import-urls">{#each tool.urls || [] as url}<code>{url}</code>{/each}</div>
					<small>승인하면 원문을 내려받아 보관하고 검색 가능한 형태로 색인합니다.</small>
					{#if tool.approval_error}<p class="tool-error">{tool.approval_error}</p>{/if}
					<div class="tool-approval-actions"><button onclick={() => onToolApproval(tool, 'reject')} disabled={tool.approving}>거부</button><button class="approve" onclick={() => onToolApproval(tool, 'once')} disabled={tool.approving}>{tool.approving ? '가져오는 중…' : '가져오기 승인'}</button></div>
                {:else}
                  <div class="tool-approval-title"><strong>{tool.name === 'ssh_download' ? 'SSH 파일 가져오기 승인' : 'SSH 명령 실행 승인'}</strong><span>{tool.host_name || tool.host}</span></div>
                  {#if tool.host_key?.fingerprint}<div class="tool-host-key-warning"><strong>처음 연결하는 서버</strong><span>호스트 키 지문을 확인하세요.</span><code>{tool.host_key.fingerprint}</code></div>{/if}
                  <code>{tool.command}</code>
                  {#if tool.reason}<small>{tool.reason}</small>{/if}
                  {#if tool.approval_error}<p class="tool-error">{tool.approval_error}</p>{/if}
                  <div class="tool-approval-actions">
                    <button onclick={() => onToolApproval(tool, 'reject')} disabled={tool.approving}>거부</button>
                    <button onclick={() => onToolApproval(tool, 'once')} disabled={tool.approving}>{tool.host_key?.fingerprint ? '키 신뢰 후 이번만' : (tool.name === 'ssh_download' ? '이번만 가져오기' : '이번만 실행')}</button>
                    {#if tool.conversation_scope_available}<button class="approve" onclick={() => onToolApproval(tool, 'conversation')} disabled={tool.approving}>{tool.approving ? '처리 중…' : (tool.host_key?.fingerprint ? '키 신뢰·대화 허용' : '이 대화에서 허용')}</button>{/if}
                  </div>
                {/if}
              </div>
            {/each}
          </section>
        {/if}
        {#if message.tool_trace?.some(tool => tool.name === 'skill_view' && !tool.error && tool.result)}
          <div class="message-skills" title="모델이 읽은 작업 지침입니다. 실제 도구 실행 여부는 아래 기록에서 확인하세요."><small>불러온 지침: {Array.from(new Set(message.tool_trace.filter(tool => tool.name === 'skill_view' && !tool.error && tool.result).map(tool => { try { return JSON.parse(tool.arguments).name; } catch { return ''; } }).filter(Boolean))).join(' · ')}</small></div>
        {/if}
        {#if message.tool_trace?.length}
          <details class="tool-trace">
            <summary><span class="activity-label" class:activity-scanner={running && (message.activity === 'tool' || message.tool_trace.some((tool) => tool.running))}>{message.tool_trace.some((tool) => tool.running) ? (message.tool_trace.find(tool => tool.running && isGenerationTool(tool)) ? generationProgressLabel(message.tool_trace.find(tool => tool.running && isGenerationTool(tool))) : '도구 실행 중…') : `도구 ${message.tool_trace.length}회`}</span></summary>
            <div class="tool-list">
              {#each message.tool_trace as tool}
                <div class="tool-item">
                  <div class="tool-heading"><strong>{toolLabel(tool)}</strong><span>{toolArgument(tool)}</span></div>
                  {#if tool.running && tool.progress}
                    <div class="generation-progress" role="status">
                      <small>{generationProgressLabel(tool)}</small>
                      {#if tool.progress.total > 0 && tool.progress.step !== null}<progress max={tool.progress.total} value={tool.progress.step} aria-label="현재 생성 단계 진행률"></progress>{/if}
                    </div>
                  {/if}
                  {#if tool.approval_required}<p class="tool-running">사용자 승인 대기 중…</p>
                  {:else if tool.approval_answered && !tool.approved}<p class="tool-error">사용자가 실행을 거부했습니다.</p>
                  {:else if tool.running && !tool.output}<p class="tool-running">{toolRunningLabel(tool)}</p>{/if}
                  {#if toolPreview(tool)}<pre class:ssh-output={tool.name === 'ssh_exec'}>{toolPreview(tool)}</pre>{/if}
                  {#each imageReferenceGroups(tool) as group}
                    <details class="image-input-evidence"><summary>{group.subject} · 후보 {group.candidates?.length || 0}장</summary>
                      <small>후보 전체는 대화 모델이 검토하며, 아래 표시된 선택 사진만 해당 생성·보정 단계에 전달됩니다.</small>
                      <div class="image-input-grid">
                        {#each group.candidates || [] as candidate, index}
                          <div class="image-input-card">
                            <small>후보 {index + 1}{candidate.id === group.scene_image_id ? ' · 장면 생성 선택' : ''}{candidate.id === group.head_image_id ? ' · 얼굴 보정 선택' : ''}</small>
                            <MediaAttachments attachments={[{id:candidate.id,name:candidate.name,mime:'image/png',url:candidate.url}]} />
                          </div>
                        {/each}
                      </div>
                      <small>{group.selection_reason}</small>
                    </details>
                  {/each}
                  {#if imageToolInputs(tool).length}
                    <div class="image-input-evidence" aria-label="실제 사용한 입력 이미지">
                      <small>실제 사용한 입력 이미지</small>
                      <div class="image-input-grid">
                        {#each imageToolInputs(tool) as input}
                          <div class="image-input-card">
                            <small>{input.index} · {input.role === 'mask' ? '마스크' : input.role === 'source' ? '편집 대상' : '참조'} · {input.origin === 'user' ? '원본 첨부' : '생성 결과'}</small>
                            <MediaAttachments attachments={[{id:input.id,name:input.name,mime:'image/png',url:input.url}]} />
                            {#if input.description}<small>{input.description}</small>{/if}
                            {#if input.crop_box?.length === 4}<small>사용 영역: {input.crop_box.join(', ')}</small>{/if}
                          </div>
                        {/each}
                      </div>
                    </div>
                  {/if}
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
          {#if message.role === 'user'}
            <UserInput content={message.content} {render} />
          {:else}
            <div class="bubble prose" use:markdownView={render(visibleAssistantContent(message.content || (running && (index === messages.length - 1 || index === retryingIndex) ? '▍' : '')))}></div>
          {/if}
          {#if message.role === 'user'}
            {#each message.turn_inputs || [] as extra (extra.id)}
              <UserInput content={extra.content} {render} label="추가 지시" />
            {/each}
          {/if}
        {/if}
        {#if message.status === 'failed' || message.status === 'cancelled'}
          <div class="message-status" class:cancelled={message.status === 'cancelled'}>
            <strong>{message.status === 'cancelled' ? (message.role === 'assistant' ? '불완전한 답변' : '중지됨') : (message.role === 'assistant' ? '불완전한 답변' : '실패')}</strong>
            {#if message.error}<span>{message.error}</span>{/if}
          </div>
        {/if}
        {#if message.role === 'assistant'}
          <InferenceMetrics performance={message.performance} />
          {#if message.content && !running}<SpeechPreview original={message.content} omitParentheticals={speechOmitParentheticals} />{/if}
          <div class="message-actions">
            {#if visibleAssistantContent(message.content)}
              <button type="button" onclick={() => copyReply(message, index, 'text')} title="서식 기호 없이 답변 본문 복사">일반 복사</button>
              <button type="button" onclick={() => copyReply(message, index, 'markdown')} title="Markdown 원문으로 답변 본문 복사">Markdown 복사</button>
            {/if}
            {#if variantIndices(message, index).length > 1}
              <div class="variant-pager" aria-label="답변 버전 선택">
                <button onclick={() => onShowAdjacentVariant(message, index, -1)} disabled={running || variantPosition(message, index) <= 0} aria-label="이전 답변">‹</button>
                <span>{variantPosition(message, index) + 1}/{variantIndices(message, index).length}</span>
                <button onclick={() => onShowAdjacentVariant(message, index, 1)} disabled={running || variantPosition(message, index) >= variantIndices(message, index).length - 1} aria-label="다음 답변">›</button>
              </div>
            {/if}
            {#if message.content && !['failed', 'cancelled'].includes(message.status)}
              {@const speechKey = replySpeechKey(message)}
              <button class:active-speech={speechPlayingKey === speechKey} onclick={() => onSpeakReply(message)} disabled={running || !ttsEnabled || (speechLoadingKey && speechLoadingKey !== speechKey)} title={!ttsEnabled ? '설정에서 답변 음성을 활성화하세요' : 'TTS로 답변 읽기'}>
                {speechLoadingKey === speechKey ? '◌ 음성 생성 중' : speechPlayingKey === speechKey ? '■ 정지' : '🔊 읽기'}
              </button>
            {/if}
            {#if messageArtifacts.length}
              <button class="artifact-open-button" onclick={() => onOpenArtifact(messageArtifacts[0])} disabled={running}>◫ 미리보기</button>
            {/if}
            <button onclick={() => onRemember(message)} disabled={running || rememberingId === message.id || rememberedIds.includes(message.id)}>{rememberedIds.includes(message.id) ? '✓ 기억됨' : rememberingId === message.id ? '기억 중…' : '＋ 기억'}</button>
            <button onclick={() => onRetry(message, index)} disabled={running || !message.id}>↻ 재시도</button>
            {#if copyFeedback?.key === `${messageKey(message, index)}:${message.variant_index ?? 0}`}
              <span class="copy-feedback" role="status">{copyFeedback.text}</span>
            {/if}
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
            <button onclick={() => onRemember(message)} disabled={running || rememberingId === message.id || rememberedIds.includes(message.id)}>{rememberedIds.includes(message.id) ? '✓ 기억됨' : rememberingId === message.id ? '기억 중…' : '＋ 기억'}</button>
            <button onclick={() => onBeginEdit(message)} disabled={running}>✎ {message.status === 'failed' || message.status === 'cancelled' ? '수정·재시도' : '수정'}</button>
          </div>
        {/if}
      </div>
    </article>
  {/each}
  {#if bottomSpacerHeight > 0}<div class="message-virtual-spacer" style:height={`${bottomSpacerHeight}px`} aria-hidden="true"></div>{/if}
</section>

<style>
.generation-progress {display:grid;gap:6px;margin:8px 0}.generation-progress progress {width:100%;height:7px;accent-color:#648de5}
  .message-actions { flex-wrap: wrap; align-items: center; }
  .copy-feedback { font-size: 12px; color: inherit; }
  .image-input-evidence { margin: 10px 0; }
  .image-input-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin-top: 6px; }
  .image-input-card { min-width: 0; padding: 8px; border: 1px solid #80808040; border-radius: 8px; }
  .image-input-card > small { display: block; overflow-wrap: anywhere; }
</style>
