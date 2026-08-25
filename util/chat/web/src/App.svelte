<script>
  import { onMount, tick } from 'svelte';
  import SettingsModal from './components/SettingsModal.svelte';
  import Sidebar from './components/Sidebar.svelte';
  import ChatHeader from './components/ChatHeader.svelte';
  import Composer from './components/Composer.svelte';
  import MessageList from './components/MessageList.svelte';
  import ContextRail from './components/ContextRail.svelte';
  import { hydrateMessages, variantIndices as getVariantIndices, applyVariant as applyMessageVariant } from './lib/message-variants.js';
  import { createStreamHandlers } from './lib/chat-stream.js';
  import { createChatSessionController } from './lib/chat-session-controller.js';
  import {
    listSessions, createSession, deleteSession, renameSession, listMessages, streamChat,
    getHealth, getModels, getConfig, retryMessage, editMessage as editChatMessage, uploadAttachment, uploadMediaURL,
    setSessionGroup, listGroups, createGroup, renameGroup, moveGroup, deleteGroup,
    getContextState, compactContext, clearContext, answerToolApproval, transcribeVoice,
    listSSHConversationGrants, revokeSSHConversationGrant, clearSSHConversationGrants, streamSpeech,
  } from './api.js';
  import { hasFileDrag, isSupportedAttachmentFile } from './lib/attachments.js';
  import { createAttachmentController } from './lib/attachment-controller.js';
  import { createVoiceController } from './lib/voice-controller.js';
  import { createSpeechBatcher, createSpeechChunker, speechTextFromMarkdown } from './lib/speech-text.js';
  import { createSpeechController } from './lib/speech-controller.js';
  import { applyTheme } from './lib/theme.js';
  import { normalizePublicSettings } from './lib/settings.js';

  let sessions = [];
  let groups = [];
  let ungroupedSessions = [];
  let sessionsByGroup = {};
  let collapsedGroups = {};
  let activeId = '';
  let messages = [];
  let reasoningOpen = {};
  let models = [];
  let selectedModel = '';
  let reasoningEffort = '';
  let webToolsEnabled = false;
  let appearance = { assistant_avatar: 'preset:spark', user_avatar: 'preset:person-blue', theme: 'system' };
  let input = '';
  let running = false;
  let retryingIndex = -1;
  let error = '';
  let health = { status: 'checking', model: '' };
  let sessionRuns = {};
  let messagePane;
  let sidebarOpen = true;
  let sidebarWidth = 260;
  let settingsOpen = false;
  let settings = null;
  let editingMessageId = null;
  let editInput = '';
  let pendingAttachments = [];
  let uploadingAttachments = false;
  let attachmentDrafts = {};
  let attachmentInput;
  let dragActive = false;
  let dragResetTimer;
  let sourceDownloading = false;
  let editingTitle = false;
  let titleInput = '';
  let titleEditor;
  let titleSaving = false;
  let controlsOpen = false;
  let contextState = null;
  let contextOpen = false;
  let contextLoading = false;
  let sshConversationGrants = [];
  let composerInput;
  let microphoneAvailable = false;
  let voiceState = 'idle';
  let voiceSeconds = 0;
  let continuousVoiceEnabled = false;
  let continuousVoiceState = 'off';
  let continuousQueueCount = 0;
  let speechLoadingKey = '';
  let speechPlayingKey = '';
  let systemThemeQuery;

  const voiceController = createVoiceController({
    environment: () => window,
    transcribe: transcribeVoice,
    getActiveSessionId: () => activeId,
    isSessionRunning: (sessionId) => Boolean(sessionRuns[sessionId]),
    isUploading: () => uploadingAttachments,
    filterFillers: () => settings?.asr?.filter_fillers !== false,
    hasInput: (sessionId) => activeId === sessionId && Boolean(input.trim()),
    onTranscript: (_sessionId, transcript) => { input = input.trim() ? `${input.trimEnd()} ${transcript}` : transcript; },
    onAutoSend: () => send(),
    onFocus: (sessionId) => focusComposer(sessionId),
    onBeforeContinuous: () => composerInput?.blur(),
    onUtteranceStart: () => { if (speechLoadingKey) stopReplySpeech(); },
    onError: (sessionId, message) => setSessionError(sessionId, message),
    onState: (state) => {
      voiceState = state.manualState;
      voiceSeconds = state.seconds;
      continuousVoiceEnabled = state.continuousEnabled;
      continuousVoiceState = state.continuousState;
      continuousQueueCount = state.queueCount;
    },
  });

  const chatController = createChatSessionController({
    loadMessages: listMessages,
    hydrate: hydrateMessages,
    onActive: (id) => { activeId = id; },
    onMessages: (nextMessages) => { messages = nextMessages; scrollBottom(); },
    onRuns: (runs) => { sessionRuns = runs; },
    onError: (message) => { error = message; },
  });

  const speechController = createSpeechController({
    stream: streamSpeech,
    cryptoSource: () => window.crypto,
    onStatus: ({ loadingKey, playingKey }) => {
      speechLoadingKey = loadingKey;
      speechPlayingKey = playingKey;
    },
    onPlaybackChange: (playing) => {
      voiceController.setPlaybackPaused(playing);
    },
    onError: (sessionId, message) => setSessionError(sessionId, message),
  });

  const attachmentController = createAttachmentController({
    uploadFile: uploadAttachment,
    uploadURL: uploadMediaURL,
    onState: (state) => {
      attachmentDrafts = state.drafts;
      pendingAttachments = state.pending;
      uploadingAttachments = state.uploading;
      sourceDownloading = state.sourceDownloading;
    },
    onError: (sessionId, message) => {
      if (sessionId) setSessionError(sessionId, message);
      else error = message;
    },
    onQueueIdle: () => { if (attachmentInput) attachmentInput.value = ''; },
  });

  $: applyTheme(appearance?.theme || 'system');

  $: activeSession = sessions.find((item) => item.id === activeId);
  $: ungroupedSessions = sessions.filter((session) => !session.group_id);
  $: sessionsByGroup = groups.reduce((result, group) => {
    result[group.id] = sessions.filter((session) => session.group_id === group.id);
    return result;
  }, {});
  $: activeRun = sessionRuns[activeId] || null;
  $: running = Boolean(activeRun);
  $: retryingIndex = activeRun?.retryingIndex ?? -1;

  onMount(() => {
    const visualViewport = window.visualViewport;
    const syncViewportHeight = () => {
      const height = visualViewport?.height || window.innerHeight;
      document.documentElement.style.setProperty('--sparktalk-app-height', `${Math.round(height)}px`);
    };
    syncViewportHeight();
    window.addEventListener('resize', syncViewportHeight);
    visualViewport?.addEventListener('resize', syncViewportHeight);
    microphoneAvailable = voiceController.supported();
    const mobile = window.matchMedia('(max-width: 600px)').matches;
    sidebarOpen = mobile ? false : localStorage.getItem('sparktalk.sidebar-open') !== 'false';
    sidebarWidth = Number(localStorage.getItem('sparktalk.sidebar-width')) || 260;
    try { collapsedGroups = JSON.parse(localStorage.getItem('sparktalk.collapsed-groups') || '{}'); } catch { collapsedGroups = {}; }
    systemThemeQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const syncSystemTheme = () => {
      if ((appearance?.theme || 'system') === 'system') applyTheme('system', false);
    };
    systemThemeQuery.addEventListener?.('change', syncSystemTheme);
    load();
    const timer = setInterval(refreshHealth, 15000);
    window.addEventListener('dragenter', onWindowDragEnter, true);
    window.addEventListener('dragover', onWindowDragOver, true);
    window.addEventListener('dragleave', onWindowDragLeave, true);
    window.addEventListener('drop', onWindowDrop, true);
    return () => {
      clearInterval(timer);
      clearTimeout(dragResetTimer);
      voiceController.dispose().catch(() => {});
      stopReplySpeech();
      window.removeEventListener('dragenter', onWindowDragEnter, true);
      window.removeEventListener('dragover', onWindowDragOver, true);
      window.removeEventListener('dragleave', onWindowDragLeave, true);
      window.removeEventListener('drop', onWindowDrop, true);
      window.removeEventListener('resize', syncViewportHeight);
      visualViewport?.removeEventListener('resize', syncViewportHeight);
      document.documentElement.style.removeProperty('--sparktalk-app-height');
      systemThemeQuery?.removeEventListener?.('change', syncSystemTheme);
    };
  });

  async function load() {
    try {
      const cfg = await getConfig();
      normalizePublicSettings(cfg);
      settings = cfg;
      reasoningEffort = cfg.model.reasoning_effort || '';
      webToolsEnabled = cfg.tools?.enabled ?? false;
      appearance = cfg.appearance || appearance;
      await Promise.all([refreshModels(), refreshHealth()]);
      selectedModel = cfg.model.default_model || models[0] || '';
      [groups, sessions] = await Promise.all([listGroups(), listSessions()]);
      if (sessions.length) await select(sessions[0].id);
      else {
        await chatController.activate('');
        sshConversationGrants = [];
      }
    } catch (e) { error = e.message; }
  }

  async function refreshModels() {
    try { models = await getModels(); }
    catch (e) { models = []; }
  }

  async function refreshHealth() {
    try { health = await getHealth(); }
    catch (e) { health = { status: 'degraded', model: '', error: e.message }; }
  }

  async function refreshSessions() {
    sessions = await listSessions();
  }

  function toggleGroup(id) {
    collapsedGroups = { ...collapsedGroups, [id]: !collapsedGroups[id] };
    localStorage.setItem('sparktalk.collapsed-groups', JSON.stringify(collapsedGroups));
  }

  async function addGroup() {
    const name = prompt('새 그룹 이름을 입력하세요.');
    if (name === null || !name.trim()) return;
    try {
      const group = await createGroup(name.trim());
      groups = [...groups, group];
      collapsedGroups = { ...collapsedGroups, [group.id]: false };
      error = '';
    } catch (e) { error = e.message; }
  }

  async function editGroup(group) {
    const name = prompt('그룹 이름을 수정하세요.', group.name);
    if (name === null || !name.trim() || name.trim() === group.name) return;
    try {
      const result = await renameGroup(group.id, name.trim());
      groups = groups.map((item) => item.id === group.id ? { ...item, name: result.name } : item);
      error = '';
    } catch (e) { error = e.message; }
  }

  async function reorderGroup(group, direction) {
    try {
      await moveGroup(group.id, direction);
      groups = await listGroups();
      error = '';
    } catch (e) { error = e.message; }
  }

  async function removeGroup(group) {
    if (!confirm(`'${group.name}' 그룹을 삭제할까요? 안의 대화는 그룹 없음으로 이동합니다.`)) return;
    try {
      await deleteGroup(group.id);
      groups = groups.filter((item) => item.id !== group.id);
      sessions = sessions.map((session) => session.group_id === group.id ? { ...session, group_id: '' } : session);
      error = '';
    } catch (e) { error = e.message; }
  }

  async function changeSessionGroup(session, groupId) {
    const previousGroupId = session.group_id || '';
    if (previousGroupId === groupId) return;
    sessions = sessions.map((item) => item.id === session.id ? { ...item, group_id: groupId } : item);
    try {
      await setSessionGroup(session.id, groupId);
      error = '';
    } catch (e) {
      sessions = sessions.map((item) => item.id === session.id ? { ...item, group_id: previousGroupId } : item);
      error = e.message;
    }
  }

  async function addSession() {
    // 빈 값으로 요청하면 서버가 YAML/설정 화면의 전역 기본값을 적용한다.
    const item = await createSession('새 대화', '', '');
    sessions = [item, ...sessions];
    await select(item.id);
  }

  async function select(id) {
    if (activeId && activeId !== id) stopReplySpeech();
    await chatController.activate(id);
    if (activeId !== id) return;
    attachmentController.select(id);
    reasoningOpen = {};
    editingMessageId = null;
    editInput = '';
    editingTitle = false;
    const session = sessions.find((item) => item.id === id);
    if (session?.model) selectedModel = session.model;
    if (session?.reasoning_effort) reasoningEffort = session.reasoning_effort;
    await Promise.all([refreshContext(id), refreshSSHGrants(id)]);
    await scrollBottom(true);
    closeSidebarOnMobile();
    const pendingTranscript = voiceController.consumePendingTranscript(id);
    if (pendingTranscript) input = input.trim() ? `${input.trimEnd()} ${pendingTranscript}` : pendingTranscript;
    voiceController.flushAutoSend(id);
    await focusComposer(id);
  }

  async function refreshContext(sessionId = activeId) {
    if (!sessionId) { contextState = null; return; }
    try {
      const next = await getContextState(sessionId);
      if (activeId === sessionId) contextState = next;
    } catch (e) {
      if (activeId === sessionId) contextState = { notice: e.message, segments: [] };
    }
  }

  async function refreshSSHGrants(sessionId = activeId) {
    if (!sessionId) { sshConversationGrants = []; return; }
    try {
      const grants = await listSSHConversationGrants(sessionId);
      if (activeId === sessionId) sshConversationGrants = grants;
    } catch (e) {
      if (activeId === sessionId) sshConversationGrants = [];
    }
  }

  async function revokeSSHGrant(hostId) {
    if (!activeId) return;
    const sessionId = activeId;
    try {
      await revokeSSHConversationGrant(sessionId, hostId);
      if (activeId === sessionId) sshConversationGrants = sshConversationGrants.filter((grant) => grant.host_id !== hostId);
      error = '';
    } catch (e) { error = e.message; }
  }

  async function revokeAllSSHGrants() {
    if (!activeId || !sshConversationGrants.length || !confirm('이 대화의 SSH 자동 허용을 모두 해제할까요?')) return;
    const sessionId = activeId;
    try {
      await clearSSHConversationGrants(sessionId);
      if (activeId === sessionId) sshConversationGrants = [];
      error = '';
    } catch (e) { error = e.message; }
  }

  async function compactActiveContext() {
    if (!activeId || running || contextLoading) return;
    contextLoading = true;
    try {
      contextState = await compactContext(activeId);
      error = contextState.notice || '';
    } catch (e) { error = e.message; }
    finally { contextLoading = false; }
  }

  async function resetActiveContext() {
    if (!activeId || running || contextLoading || !confirm('저장된 컨텍스트 요약을 초기화할까요? 대화 원본은 삭제되지 않습니다.')) return;
    contextLoading = true;
    try {
      await clearContext(activeId);
      await refreshContext(activeId);
      error = '';
    } catch (e) { error = e.message; }
    finally { contextLoading = false; }
  }

  function jumpToMessage(messageId) {
    contextOpen = false;
    requestAnimationFrame(() => document.querySelector(`[data-message-id="${messageId}"]`)?.scrollIntoView({ behavior: 'smooth', block: 'center' }));
  }

  async function remove(id) {
    if (sessionRuns[id]) return;
    if (!confirm('이 대화를 삭제할까요?')) return;
    const previousSessions = sessions;
    sessions = sessions.filter((item) => item.id !== id);
    try {
      await deleteSession(id);
      attachmentController.discard(id);
      const wasActive = activeId === id;
      chatController.remove(id);
      if (wasActive) {
        if (sessions.length) await select(sessions[0].id);
        else {
          attachmentController.select('');
          sshConversationGrants = [];
          error = '';
        }
      }
    } catch (e) {
      sessions = previousSessions;
      error = e.message;
    }
  }

  async function beginTitleEdit() {
    if (!activeSession || running) return;
    titleInput = activeSession.title;
    editingTitle = true;
    await tick();
    titleEditor?.focus();
    titleEditor?.select();
  }

  async function saveTitle() {
    const title = titleInput.trim();
    if (titleSaving || !activeSession || !title || title.length > 120) return;
    if (title === activeSession.title) {
      editingTitle = false;
      return;
    }
    titleSaving = true;
    try {
      const result = await renameSession(activeSession.id, title);
      sessions = sessions.map((item) => item.id === activeSession.id ? { ...item, title: result.title } : item);
      editingTitle = false;
      error = '';
    } catch (e) { error = e.message; }
    finally { titleSaving = false; }
  }

  function titleKeydown(event) {
    if (event.key === 'Enter') { event.preventDefault(); saveTitle(); }
    if (event.key === 'Escape') editingTitle = false;
  }

  async function send() {
    const content = input.trim();
    if (!content || running || uploadingAttachments || voiceState !== 'idle' || !activeId) return;
    const sessionId = activeId;
    stopReplySpeech();
    voiceController.clearAutoSend(sessionId);
    const attachments = pendingAttachments;
    input = '';
    setPendingAttachments(sessionId, []);
    setSessionError(sessionId, '');
    const chatMessages = [...messages,
      { role: 'user', status: 'pending', content, reasoning_content: '', attachments },
      { role: 'assistant', status: 'pending', content: '', reasoning_content: '', tool_trace: [], activity: '' },
    ];
    const replyIndex = chatMessages.length - 1;
    const run = startSessionRun(sessionId, chatMessages, replyIndex);
    await scrollBottom(true);
    try {
      await streamChat(sessionId, content, attachments, selectedModel, reasoningEffort, webToolsEnabled, run.controller.signal,
        streamHandlersFor(run.messages[replyIndex], sessionId, run.messages));
      publishMessages(sessionId, hydrateMessages(await listMessages(sessionId)));
      await refreshContext(sessionId);
      await refreshSessions();
      setTimeout(refreshSessions, 1800);
      run.completed = true;
    } catch (e) {
      if (e.name !== 'AbortError') setSessionError(sessionId, e.message);
      if (e.name === 'AbortError') await new Promise((resolve) => setTimeout(resolve, 80));
      try {
        publishMessages(sessionId, hydrateMessages(await listMessages(sessionId)));
        await refreshContext(sessionId);
      } catch { /* keep the optimistic transcript if the refresh also fails */ }
    } finally {
      finishSessionRun(sessionId, run);
    }
  }

  function onKeydown(event) {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); send(); }
  }

  async function startVoiceInput() {
    await voiceController.startManual();
  }

  async function stopVoiceInput() {
    await voiceController.stopManual();
  }

  async function toggleContinuousVoice() {
    if (speechLoadingKey) stopReplySpeech();
    await voiceController.toggleContinuous();
  }

  async function stopContinuousVoice() {
    await voiceController.stopContinuous();
  }

  function stop() { chatController.abort(activeId); }
  async function retry(message, index) {
    if (running || !message.id) return;
    if (index < messages.length - 1 && !confirm('이 답변을 재시도하면 이후 대화가 제거됩니다. 계속할까요?')) return;
    const sessionId = activeId;
    setSessionError(sessionId, '');
    const original = { content: message.content, reasoning_content: message.reasoning_content, tool_trace: message.tool_trace };
    message.content = '';
    message.reasoning_content = '';
    message.tool_trace = [];
    message.activity = '';
    const run = startSessionRun(sessionId, messages, index);
    reasoningOpen = { ...reasoningOpen, [index]: false };
    try {
      const userVariant = run.messages[index - 1]?.role === 'user' ? (run.messages[index - 1].variant_index ?? 0) : 0;
      await retryMessage(message.id, selectedModel, reasoningEffort, webToolsEnabled, userVariant, run.controller.signal,
        streamHandlersFor(message, sessionId, run.messages));
      const updated = hydrateMessages(await listMessages(sessionId));
      const parent = updated[index - 1];
      const answer = updated[index];
      if (parent?.role === 'user' && answer?.role === 'assistant' && parent.variants?.[userVariant]) {
        applyVariant(parent, userVariant, index - 1, sessionId);
        const matching = variantIndices(answer, index, updated);
        if (matching.length) applyVariant(answer, matching[matching.length - 1], index, sessionId);
      }
      publishMessages(sessionId, updated);
      await refreshContext(sessionId);
      await refreshSessions();
      run.completed = true;
    } catch (e) {
      message.content = original.content;
      message.reasoning_content = original.reasoning_content;
      message.tool_trace = original.tool_trace;
      message.activity = '';
      publishMessages(sessionId, run.messages);
      if (e.name !== 'AbortError') setSessionError(sessionId, e.message);
    } finally {
      finishSessionRun(sessionId, run);
    }
  }
  function beginEdit(message) {
    if (running || !message.id) return;
    editingMessageId = message.id;
    editInput = message.content;
  }
  function cancelEdit() {
    editingMessageId = null;
    editInput = '';
  }
  function onEditKeydown(event, message, index) {
    if (event.key === 'Escape') {
      cancelEdit();
      return;
    }
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      submitEdit(message, index);
    }
  }
  async function submitEdit(message, index) {
    const content = editInput.trim();
    if (!content || running || !message.id || (content === message.content && !['failed', 'cancelled'].includes(message.status))) return;
    const hasPairedAnswer = messages[index + 1]?.role === 'assistant';
    const firstDiscardedIndex = index + (hasPairedAnswer ? 2 : 1);
    if (firstDiscardedIndex < messages.length && !confirm('이 질문을 수정하면 이후 대화가 새 분기로 바뀝니다. 계속할까요?')) return;
    const sessionId = activeId;
    const originalMessages = structuredClone(messages);
    setSessionError(sessionId, '');
    message.content = content;
    let replyIndex = index + 1;
    if (messages[replyIndex]?.role !== 'assistant') {
      messages = [...messages.slice(0, replyIndex), { role: 'assistant', content: '', reasoning_content: '', tool_trace: [], activity: '' }];
    } else {
      messages = messages.slice(0, replyIndex + 1);
      messages[replyIndex].content = '';
      messages[replyIndex].reasoning_content = '';
      messages[replyIndex].tool_trace = [];
      messages[replyIndex].activity = '';
    }
    editingMessageId = null;
    reasoningOpen = { ...reasoningOpen, [replyIndex]: false };
    const run = startSessionRun(sessionId, messages, replyIndex);
    try {
      await editChatMessage(message.id, content, message.attachments || [], selectedModel, reasoningEffort, webToolsEnabled, run.controller.signal,
        streamHandlersFor(run.messages[replyIndex], sessionId, run.messages));
      publishMessages(sessionId, hydrateMessages(await listMessages(sessionId)));
      await refreshContext(sessionId);
      await refreshSessions();
      setTimeout(refreshSessions, 1800);
      editInput = '';
      run.completed = true;
    } catch (e) {
      publishMessages(sessionId, originalMessages);
      if (e.name !== 'AbortError') setSessionError(sessionId, e.message);
    } finally {
      finishSessionRun(sessionId, run);
    }
  }

  function startSessionRun(sessionId, runMessages, runRetryingIndex) {
    stopReplySpeech();
    const extra = {};
    if (continuousVoiceEnabled && settings?.tts?.enabled && settings.tts.auto_play) {
      extra.speechChunker = createSpeechChunker();
      extra.speechBatcher = createSpeechBatcher();
      extra.speechSession = speechController.create(`live:${sessionId}:${Date.now()}`, sessionId, settings?.tts?.seed);
    }
    return chatController.start(sessionId, runMessages, runRetryingIndex, extra);
  }
  async function finishSessionRun(sessionId, run) {
    if (sessionRuns[sessionId] !== run) return;
    const canAutoPlay = run.completed && activeId === sessionId && settings?.tts?.enabled && settings.tts.auto_play
      && !voiceController.hasAutoSendPending(sessionId);
    if (run.speechSession) {
      if (canAutoPlay) {
        const finalChunks = run.speechChunker?.finish() || [];
        for (const batch of run.speechBatcher?.push(finalChunks) || []) speechController.enqueue(run.speechSession, batch);
        for (const batch of run.speechBatcher?.finish() || []) speechController.enqueue(run.speechSession, batch);
        speechController.close(run.speechSession);
      } else if (speechController.isCurrent(run.speechSession)) {
        stopReplySpeech();
      }
    }
    chatController.finish(sessionId, run);
    const reply = chatController.getMessages(sessionId)?.[run.retryingIndex];
    if (canAutoPlay && !run.speechSession && reply?.role === 'assistant' && reply.content) {
      speakReply(reply);
    }
    await voiceController.flushAutoSend(sessionId);
    await focusComposer(sessionId);
  }

  async function focusComposer(sessionId = activeId) {
    await tick();
    if (activeId === sessionId && !sessionRuns[sessionId] && !settingsOpen && !controlsOpen && !continuousVoiceEnabled) {
      composerInput?.focus({ preventScroll: true });
    }
  }
  function publishMessages(sessionId, nextMessages) {
    chatController.publish(sessionId, nextMessages);
  }
  function setSessionError(sessionId, message) {
    chatController.setError(sessionId, message);
  }
  function variantIndices(message, messageIndex, messageList = messages) {
    return getVariantIndices(message, messageIndex, messageList);
  }
  function variantPosition(message, messageIndex) {
    return variantIndices(message, messageIndex).indexOf(message.variant_index);
  }
  function showAdjacentVariant(message, messageIndex, direction) {
    const indices = variantIndices(message, messageIndex);
    const position = indices.indexOf(message.variant_index);
    const next = indices[position + direction];
    if (next !== undefined) showVariant(message, next, messageIndex);
  }
  function applyVariant(message, variantIndex, messageIndex, sessionId = activeId) {
    if (!applyMessageVariant(message, variantIndex)) return;
    if (sessionId === activeId) reasoningOpen = { ...reasoningOpen, [messageIndex]: false };
  }
  function showVariant(message, variantIndex, messageIndex) {
    if (running || !message.variants?.[variantIndex]) return;
    stopReplySpeech();
    applyVariant(message, variantIndex, messageIndex);
    if (message.role === 'user') {
      const answer = messages[messageIndex + 1];
      if (answer?.role === 'assistant') {
        const matching = variantIndices(answer, messageIndex + 1);
        if (matching.length) applyVariant(answer, matching[matching.length - 1], messageIndex + 1);
      }
    }
    messages = messages;
  }
  function streamHandlersFor(message, sessionId = activeId, messageList = messages) {
    const handlers = createStreamHandlers(message, () => publishMessages(sessionId, messageList));
    const handleDelta = handlers.delta;
    handlers.delta = (delta) => {
      handleDelta(delta);
      const run = sessionRuns[sessionId];
      if (!run?.speechChunker || !run.speechBatcher || !run.speechSession) return;
      const chunks = run.speechChunker.push(delta);
      for (const batch of run.speechBatcher.push(chunks)) speechController.enqueue(run.speechSession, batch);
    };
    const handleToolApproval = handlers.toolApproval;
    handlers.toolApproval = (data) => {
      handleToolApproval(data);
      if (activeId === sessionId) requestAnimationFrame(() => document.querySelector(`[data-approval-id="${data.approval_id}"]`)?.scrollIntoView({ behavior: 'smooth', block: 'center' }));
    };
    handlers.context = (next) => { if (activeId === sessionId) contextState = next; };
    handlers.sshGrantChanged = () => { refreshSSHGrants(sessionId); };
    handlers.mediaAttached = (attachment) => {
      const assistantIndex = messageList.indexOf(message);
      if (attachment.target_role === 'assistant') {
        if (!(message.attachments || []).some((item) => item.id === attachment.id)) {
          message.attachments = [...(message.attachments || []), attachment];
          publishMessages(sessionId, messageList);
        }
        return;
      }
      const userMessage = assistantIndex > 0 ? messageList[assistantIndex - 1] : null;
      if (!userMessage || userMessage.role !== 'user') return;
      if (!(userMessage.attachments || []).some((item) => item.id === attachment.id)) {
        userMessage.attachments = [...(userMessage.attachments || []), attachment];
        publishMessages(sessionId, messageList);
      }
    };
    return handlers;
  }

  async function respondToToolApproval(tool, decision) {
    if (!tool.approval_id || tool.approving) return;
    tool.approving = true;
    tool.approval_error = '';
    messages = messages;
    try {
      await answerToolApproval(tool.approval_id, decision);
      tool.approval_required = false;
      tool.approval_answered = true;
      tool.approved = decision !== 'reject';
      tool.approval_decision = decision;
    } catch (approvalError) {
      tool.approval_error = approvalError.message;
    } finally {
      tool.approving = false;
      messages = messages;
    }
  }
  function addAttachmentFiles(files) {
    attachmentController.addFiles(files, { sessionId: activeId, blocked: running });
  }
  async function addMediaURL(rawURL) {
    return attachmentController.addURL(rawURL, { sessionId: activeId, blocked: running || uploadingAttachments });
  }
  function setPendingAttachments(sessionId, items) {
    attachmentController.setPending(sessionId, items);
  }
  function removePendingAttachment(id) {
    attachmentController.remove(id, { sessionId: activeId, blocked: running });
  }
  function onPaste(event) {
    const files = Array.from(event.clipboardData?.files || []);
    if (files.some(isSupportedAttachmentFile)) addAttachmentFiles(files);
  }
  function showDropOverlay() {
    clearTimeout(dragResetTimer);
    dragActive = true;
    // Some browsers emit dragover only a few times per second while the
    // pointer is stationary. Keep the overlay alive unless the drag really
    // leaves the viewport; this timer is only a stale-state fallback.
    dragResetTimer = setTimeout(() => { dragActive = false; }, 30000);
  }
  function onWindowDragEnter(event) {
    if (!hasFileDrag(event.dataTransfer)) return;
    event.preventDefault();
    if (settingsOpen || running || !activeId) return;
    showDropOverlay();
  }
  function onWindowDragOver(event) {
    if (!hasFileDrag(event.dataTransfer)) return;
    event.preventDefault();
    if (settingsOpen || running || !activeId) return;
    showDropOverlay();
    if (event.dataTransfer) event.dataTransfer.dropEffect = 'copy';
  }
  function onWindowDragLeave(event) {
    const outside = event.clientX <= 0 || event.clientY <= 0
      || event.clientX >= window.innerWidth || event.clientY >= window.innerHeight;
    if (!event.relatedTarget && outside) {
      clearTimeout(dragResetTimer);
      dragActive = false;
    }
  }
  function onWindowDrop(event) {
    if (!hasFileDrag(event.dataTransfer)) return;
    // Process DataTransfer.files directly. Native drops onto a transparent
    // file input do not reliably emit change in Chromium/Linux.
    event.preventDefault();
    event.stopPropagation();
    clearTimeout(dragResetTimer);
    dragActive = false;
    const files = Array.from(event.dataTransfer?.files || []);
    if (!settingsOpen && !running && activeId && files.length) addAttachmentFiles(files);
  }
  function onAttachmentInputChange(event) {
    const files = Array.from(event.currentTarget.files || []);
    clearTimeout(dragResetTimer);
    dragActive = false;
    addAttachmentFiles(files);
  }
  async function scrollBottom(force = false) {
    if (!messagePane) return;
    const nearBottom = messagePane.scrollHeight - messagePane.scrollTop - messagePane.clientHeight < 90;
    await tick();
    if (force || nearBottom) messagePane.scrollTop = messagePane.scrollHeight;
  }

  function toggleSidebar() {
    sidebarOpen = !sidebarOpen;
    if (sidebarOpen) controlsOpen = false;
    localStorage.setItem('sparktalk.sidebar-open', String(sidebarOpen));
  }

  function closeSidebar() {
    sidebarOpen = false;
    localStorage.setItem('sparktalk.sidebar-open', 'false');
  }

  function closeSidebarOnMobile() {
    if (window.matchMedia('(max-width: 600px)').matches) closeSidebar();
  }

  function toggleControls() {
    controlsOpen = !controlsOpen;
    if (controlsOpen) closeSidebar();
  }

  function closeControls() {
    controlsOpen = false;
  }

  function startResize(event) {
    event.preventDefault();
    const move = (e) => {
      sidebarWidth = Math.max(190, Math.min(480, e.clientX));
      localStorage.setItem('sparktalk.sidebar-width', String(sidebarWidth));
    };
    const stopResize = () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', stopResize);
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', stopResize);
  }

  async function openSettings() {
    try {
      settings = await getConfig();
      normalizePublicSettings(settings);
      settingsOpen = true;
      closeSidebarOnMobile();
      closeControls();
    } catch (e) { error = e.message; }
  }

  async function applySavedSettings(next) {
    stopReplySpeech();
    settings = next;
    reasoningEffort = settings.model.reasoning_effort || reasoningEffort;
    webToolsEnabled = settings.tools?.enabled ?? false;
    appearance = settings.appearance || appearance;
    await Promise.all([refreshModels(), refreshHealth()]);
    if (settings.model.default_model) selectedModel = settings.model.default_model;
  }

  function replySpeechKey(message) {
    return `${message?.id || 'pending'}:${message?.variant_index ?? 0}`;
  }

  function replySpeechText(message) {
    return speechTextFromMarkdown(message?.content || '');
  }

  function stopReplySpeech() {
    speechController.stop();
  }

  async function speakReply(message) {
    const key = replySpeechKey(message);
    if (speechLoadingKey === key || speechPlayingKey === key) {
      stopReplySpeech();
      return;
    }
    if (!settings?.tts?.enabled) {
      setSessionError(activeId, '설정에서 답변 음성을 활성화해야 합니다.');
      return;
    }
    const text = replySpeechText(message);
    if (!text) return;
    stopReplySpeech();
    const session = speechController.create(key, activeId, settings?.tts?.seed);
    speechController.enqueue(session, text);
    speechController.close(session);
  }
</script>

<div class="shell" style:grid-template-columns={sidebarOpen ? `${sidebarWidth}px 1fr` : '1fr'}>
  {#if sidebarOpen}
    <Sidebar
      {groups}
      {sessionsByGroup}
      {ungroupedSessions}
      {collapsedGroups}
      {activeId}
      {sessionRuns}
      assistantAvatar={appearance.assistant_avatar}
      onclose={closeSidebar}
      onAddSession={addSession}
      onAddGroup={addGroup}
      onToggleGroup={toggleGroup}
      onEditGroup={editGroup}
      onReorderGroup={reorderGroup}
      onRemoveGroup={removeGroup}
      onSelect={select}
      onChangeSessionGroup={changeSessionGroup}
      onRemoveSession={remove}
      onOpenSettings={openSettings}
      onStartResize={startResize}
    />
  {/if}

  <main>
    <ChatHeader
      {activeSession}
      {running}
      {editingTitle}
      bind:titleInput
      bind:titleEditor
      {models}
      bind:selectedModel
      bind:reasoningEffort
      bind:webToolsEnabled
      {health}
      {microphoneAvailable}
      {continuousVoiceEnabled}
      {continuousVoiceState}
      {continuousQueueCount}
      sshGrants={sshConversationGrants}
      {controlsOpen}
      onToggleSidebar={toggleSidebar}
      onBeginTitleEdit={beginTitleEdit}
      onTitleKeydown={titleKeydown}
      onSaveTitle={saveTitle}
      onToggleControls={toggleControls}
      onCloseControls={closeControls}
      onRevokeSSHGrant={revokeSSHGrant}
      onClearSSHGrants={revokeAllSSHGrants}
      onToggleContinuousVoice={toggleContinuousVoice}
    />
    <MessageList
      {messages}
      {running}
      {retryingIndex}
      bind:reasoningOpen
      {editingMessageId}
      bind:editInput
      bind:element={messagePane}
      assistantAvatar={appearance.assistant_avatar}
      userAvatar={appearance.user_avatar}
      {variantIndices}
      {variantPosition}
      onShowAdjacentVariant={showAdjacentVariant}
      onRetry={retry}
      {onEditKeydown}
      onCancelEdit={cancelEdit}
      onSubmitEdit={submitEdit}
      onBeginEdit={beginEdit}
      onToolApproval={respondToToolApproval}
      ttsEnabled={settings?.tts?.enabled ?? false}
      {speechLoadingKey}
      {speechPlayingKey}
      onSpeakReply={speakReply}
    />
    <ContextRail
      state={contextState}
      open={contextOpen}
      loading={contextLoading}
      disabled={running || !activeId}
      onToggle={() => contextOpen = !contextOpen}
      onCompact={compactActiveContext}
      onReset={resetActiveContext}
      onJump={jumpToMessage}
    />
    {#if error}<div class="error">{error}</div>{/if}
    <Composer
      {pendingAttachments}
      {uploadingAttachments}
      {sourceDownloading}
      {running}
      {activeId}
      bind:input
      bind:element={composerInput}
      bind:attachmentInput
      {reasoningEffort}
      {webToolsEnabled}
      {microphoneAvailable}
      {voiceState}
      {voiceSeconds}
      {continuousVoiceEnabled}
      onRemoveAttachment={removePendingAttachment}
      {onAttachmentInputChange}
      onAttachURL={addMediaURL}
      {onKeydown}
      {onPaste}
      onStop={stop}
      onSend={send}
      onStartVoice={startVoiceInput}
      onStopVoice={stopVoiceInput}
    />
  </main>
</div>

{#if dragActive}
  <div class="drop-overlay" role="region" aria-label="미디어 드롭 영역"><div><span>＋</span><strong>미디어를 여기에 놓으세요</strong><small>이미지 · MP3/WAV/OGG · AVI/MOV/MP4/OGG/WMV/WebM · 최대 6개</small></div></div>
{/if}

{#if settingsOpen && settings}
  <SettingsModal
    {settings}
    {models}
    keepMediaIds={Object.values(attachmentDrafts).flat().map((item) => item.id)}
    onclose={() => settingsOpen = false}
    onsaved={applySavedSettings}
  />
{/if}
