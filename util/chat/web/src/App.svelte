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
  import {
    listSessions, createSession, deleteSession, renameSession, listMessages, streamChat,
    getHealth, getModels, getConfig, retryMessage, editMessage as editChatMessage, uploadAttachment, uploadMediaURL,
    setSessionGroup, listGroups, createGroup, renameGroup, moveGroup, deleteGroup,
    getContextState, compactContext, clearContext, answerToolApproval, transcribeVoice,
    listSSHConversationGrants, revokeSSHConversationGrant, clearSSHConversationGrants, streamSpeech,
  } from './api.js';
  import {
    attachmentKind, hasFileDrag, isSupportedAttachmentFile, maxAttachmentBytes, maxImageBytes, maxMessageBytes,
  } from './lib/attachments.js';
  import { beginVoiceRecording, voiceFilename, voiceRecordingSupported } from './lib/voice-recorder.js';
  import { beginContinuousVoice, isIgnorableVoiceTranscript } from './lib/continuous-voice.js';
  import { createSpeechChunker, speechTextFromMarkdown } from './lib/speech-text.js';
  import { PCMStreamPlayer } from './lib/pcm-player.js';

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
  let appearance = { assistant_avatar: 'preset:spark', user_avatar: 'preset:person-blue' };
  let input = '';
  let running = false;
  let retryingIndex = -1;
  let error = '';
  let health = { status: 'checking', model: '' };
  let sessionRuns = {};
  let messageCache = {};
  let sessionErrors = {};
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
  let attachmentQueue = [];
  let queueProcessing = false;
  let uploadingSessionId = '';
  let sourceDownloadingSessionId = '';
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
  let voiceRecording = null;
  let voiceSessionId = '';
  let voiceTimer;
  let pendingVoiceTranscripts = {};
  let continuousVoiceEnabled = false;
  let continuousVoiceState = 'off';
  let continuousVoiceListener = null;
  let continuousUtterances = [];
  let continuousQueueCount = 0;
  let continuousQueueProcessing = false;
  let continuousAutoSendPending = {};
  let speechSession = null;
  let speechLoadingKey = '';
  let speechPlayingKey = '';
  let speechPausedContinuousVoice = false;

  $: activeSession = sessions.find((item) => item.id === activeId);
  $: ungroupedSessions = sessions.filter((session) => !session.group_id);
  $: sessionsByGroup = groups.reduce((result, group) => {
    result[group.id] = sessions.filter((session) => session.group_id === group.id);
    return result;
  }, {});
  $: activeRun = sessionRuns[activeId] || null;
  $: running = Boolean(activeRun);
  $: retryingIndex = activeRun?.retryingIndex ?? -1;
  $: sourceDownloading = Boolean(activeId) && sourceDownloadingSessionId === activeId;

  onMount(() => {
    microphoneAvailable = voiceRecordingSupported(window);
    const mobile = window.matchMedia('(max-width: 600px)').matches;
    sidebarOpen = mobile ? false : localStorage.getItem('sparktalk.sidebar-open') !== 'false';
    sidebarWidth = Number(localStorage.getItem('sparktalk.sidebar-width')) || 260;
    try { collapsedGroups = JSON.parse(localStorage.getItem('sparktalk.collapsed-groups') || '{}'); } catch { collapsedGroups = {}; }
    load();
    const timer = setInterval(refreshHealth, 15000);
    window.addEventListener('dragenter', onWindowDragEnter, true);
    window.addEventListener('dragover', onWindowDragOver, true);
    window.addEventListener('dragleave', onWindowDragLeave, true);
    window.addEventListener('drop', onWindowDrop, true);
    return () => {
      clearInterval(timer);
      clearTimeout(dragResetTimer);
      clearInterval(voiceTimer);
      voiceRecording?.stop().catch(() => {});
      continuousVoiceListener?.stop().catch(() => {});
      stopReplySpeech();
      window.removeEventListener('dragenter', onWindowDragEnter, true);
      window.removeEventListener('dragover', onWindowDragOver, true);
      window.removeEventListener('dragleave', onWindowDragLeave, true);
      window.removeEventListener('drop', onWindowDrop, true);
    };
  });

  async function load() {
    try {
      const cfg = await getConfig();
      normalizePromptPresetSettings(cfg);
      settings = cfg;
      reasoningEffort = cfg.model.reasoning_effort || '';
      webToolsEnabled = cfg.tools?.enabled ?? false;
      appearance = cfg.appearance || appearance;
      await Promise.all([refreshModels(), refreshHealth()]);
      selectedModel = cfg.model.default_model || models[0] || '';
      [groups, sessions] = await Promise.all([listGroups(), listSessions()]);
      if (sessions.length) await select(sessions[0].id);
      else {
        activeId = '';
        messages = [];
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
    if (activeId && messages) messageCache = { ...messageCache, [activeId]: messages };
    activeId = id;
    syncActiveAttachmentState();
    reasoningOpen = {};
    editingMessageId = null;
    editInput = '';
    editingTitle = false;
    const session = sessions.find((item) => item.id === id);
    if (session?.model) selectedModel = session.model;
    if (session?.reasoning_effort) reasoningEffort = session.reasoning_effort;
    if (sessionRuns[id]?.messages) {
      messages = sessionRuns[id].messages;
    } else if (messageCache[id]) {
      messages = messageCache[id];
    } else {
      const loaded = hydrateMessages(await listMessages(id));
      messageCache = { ...messageCache, [id]: loaded };
      if (activeId === id) messages = loaded;
    }
    if (activeId !== id) return;
    await Promise.all([refreshContext(id), refreshSSHGrants(id)]);
    error = sessionErrors[id] || '';
    await scrollBottom(true);
    closeSidebarOnMobile();
    applyPendingVoiceTranscript(id);
    flushContinuousAutoSend(id);
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
    if (!activeId || running || contextLoading || !confirm('저장된 문맥 요약을 초기화할까요? 대화 원본은 삭제되지 않습니다.')) return;
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
      const nextDrafts = { ...attachmentDrafts };
      delete nextDrafts[id];
      attachmentDrafts = nextDrafts;
      attachmentQueue = attachmentQueue.filter((item) => item.sessionId !== id);
      const nextCache = { ...messageCache };
      delete nextCache[id];
      messageCache = nextCache;
      const nextErrors = { ...sessionErrors };
      delete nextErrors[id];
      sessionErrors = nextErrors;
      if (activeId === id) {
        if (sessions.length) await select(sessions[0].id);
        else {
          activeId = '';
          messages = [];
          pendingAttachments = [];
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
    continuousAutoSendPending = { ...continuousAutoSendPending, [sessionId]: false };
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

  function formatVoiceError(error) {
    if (error?.name === 'NotAllowedError' || error?.name === 'SecurityError') {
      return '마이크 권한이 거부되었습니다. 주소창의 사이트 권한에서 마이크를 허용하세요.';
    }
    if (error?.name === 'NotFoundError') return '사용 가능한 마이크를 찾지 못했습니다.';
    return error?.message || '마이크 녹음에 실패했습니다.';
  }

  async function startVoiceInput() {
    if (!activeId || running || uploadingAttachments || voiceState !== 'idle' || continuousVoiceEnabled || ['requesting', 'stopping'].includes(continuousVoiceState)) return;
    if (!microphoneAvailable) {
      error = '마이크는 HTTPS, localhost 또는 안전한 출처로 허용한 주소에서 사용할 수 있습니다.';
      return;
    }
    voiceState = 'requesting';
    voiceSessionId = activeId;
    setSessionError(voiceSessionId, '');
    try {
      voiceRecording = await beginVoiceRecording(window);
      voiceSeconds = 0;
      voiceState = 'recording';
      clearInterval(voiceTimer);
      voiceTimer = setInterval(() => {
        voiceSeconds += 1;
        if (voiceSeconds >= 300) stopVoiceInput();
      }, 1000);
    } catch (voiceError) {
      setSessionError(voiceSessionId, formatVoiceError(voiceError));
      voiceRecording = null;
      voiceState = 'idle';
      voiceSessionId = '';
    }
  }

  async function stopVoiceInput() {
    if (voiceState !== 'recording' || !voiceRecording) return;
    const recording = voiceRecording;
    const sessionId = voiceSessionId;
    voiceRecording = null;
    voiceState = 'transcribing';
    clearInterval(voiceTimer);
    try {
      const blob = await recording.stop();
      if (!blob.size) throw new Error('녹음된 음성이 없습니다.');
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 5 * 60 * 1000);
      try {
        const result = await transcribeVoice(blob, voiceFilename(blob.type), controller.signal);
        appendVoiceTranscript(sessionId, result.text || '');
        setSessionError(sessionId, '');
      } finally {
        clearTimeout(timeout);
      }
    } catch (voiceError) {
      setSessionError(sessionId, voiceError?.name === 'AbortError' ? '음성 인식 시간이 초과되었습니다.' : formatVoiceError(voiceError));
    } finally {
      voiceState = 'idle';
      voiceSeconds = 0;
      voiceSessionId = '';
      await flushContinuousAutoSend(sessionId);
      await focusComposer(sessionId);
    }
  }

  function appendVoiceTranscript(sessionId, text) {
    const transcript = text.trim();
    if (!transcript) return;
    if (activeId === sessionId) {
      input = input.trim() ? `${input.trimEnd()} ${transcript}` : transcript;
      return;
    }
    pendingVoiceTranscripts = {
      ...pendingVoiceTranscripts,
      [sessionId]: [pendingVoiceTranscripts[sessionId], transcript].filter(Boolean).join(' '),
    };
  }

  function applyPendingVoiceTranscript(sessionId) {
    const transcript = pendingVoiceTranscripts[sessionId];
    if (!transcript) return;
    input = input.trim() ? `${input.trimEnd()} ${transcript}` : transcript;
    const next = { ...pendingVoiceTranscripts };
    delete next[sessionId];
    pendingVoiceTranscripts = next;
  }

  async function toggleContinuousVoice() {
    if (continuousVoiceEnabled || continuousVoiceState === 'error') {
      await stopContinuousVoice();
      return;
    }
    if (!activeId || voiceState !== 'idle' || ['requesting', 'stopping'].includes(continuousVoiceState)) return;
    if (!microphoneAvailable) {
      error = '마이크는 HTTPS, localhost 또는 안전한 출처로 허용한 주소에서 사용할 수 있습니다.';
      return;
    }
    // Continuous dictation is hands-free. In particular, Android browsers
    // must not reopen the software keyboard while listening or after a reply.
    composerInput?.blur();
    continuousVoiceState = 'requesting';
    try {
      const listener = await beginContinuousVoice(window, {
        getContext: () => ({ sessionId: activeId }),
        onState: (state) => { continuousVoiceState = state; },
        onUtterance: (blob, context) => {
          if (speechLoadingKey) stopReplySpeech();
          enqueueContinuousUtterance(blob, context?.sessionId || activeId);
        },
        onError: (voiceError) => {
          setSessionError(activeId, formatVoiceError(voiceError));
          continuousVoiceEnabled = false;
          continuousVoiceState = 'error';
          continuousVoiceListener = null;
        },
      });
      continuousVoiceListener = listener;
      continuousVoiceEnabled = true;
      error = '';
    } catch (voiceError) {
      setSessionError(activeId, formatVoiceError(voiceError));
      continuousVoiceEnabled = false;
      continuousVoiceListener = null;
      continuousVoiceState = 'off';
    }
  }

  async function stopContinuousVoice() {
    const listener = continuousVoiceListener;
    continuousVoiceEnabled = false;
    continuousVoiceListener = null;
    continuousVoiceState = 'stopping';
    try { await listener?.stop(); }
    catch (voiceError) { setSessionError(activeId, formatVoiceError(voiceError)); }
    finally { continuousVoiceState = 'off'; }
  }

  function enqueueContinuousUtterance(blob, sessionId) {
    if (!blob?.size || !sessionId) return;
    continuousUtterances = [...continuousUtterances, { blob, sessionId }];
    continuousQueueCount = continuousUtterances.length + (continuousQueueProcessing ? 1 : 0);
    processContinuousUtterances();
  }

  async function processContinuousUtterances() {
    if (continuousQueueProcessing) return;
    continuousQueueProcessing = true;
    try {
      while (continuousUtterances.length) {
        const item = continuousUtterances[0];
        continuousUtterances = continuousUtterances.slice(1);
        continuousQueueCount = continuousUtterances.length + 1;
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 5 * 60 * 1000);
        try {
          const result = await transcribeVoice(item.blob, voiceFilename(item.blob.type), controller.signal);
          const transcript = result.text || '';
          if (settings?.asr?.filter_fillers !== false && isIgnorableVoiceTranscript(transcript)) {
            setSessionError(item.sessionId, '');
            continue;
          }
          appendVoiceTranscript(item.sessionId, transcript);
          continuousAutoSendPending = { ...continuousAutoSendPending, [item.sessionId]: true };
          setSessionError(item.sessionId, '');
          flushContinuousAutoSend(item.sessionId);
        } catch (voiceError) {
          const message = voiceError?.name === 'AbortError' ? '음성 인식 시간이 초과되었습니다.' : formatVoiceError(voiceError);
          if (!/empty text|no audio|녹음된 음성이 없습니다/i.test(message)) setSessionError(item.sessionId, message);
        } finally {
          clearTimeout(timeout);
        }
      }
    } finally {
      continuousQueueProcessing = false;
      continuousQueueCount = 0;
    }
  }

  async function flushContinuousAutoSend(sessionId) {
    if (!continuousAutoSendPending[sessionId] || activeId !== sessionId || sessionRuns[sessionId] || voiceState !== 'idle') return;
    await tick();
    if (!continuousAutoSendPending[sessionId] || activeId !== sessionId || sessionRuns[sessionId] || !input.trim()) return;
    continuousAutoSendPending = { ...continuousAutoSendPending, [sessionId]: false };
    send();
  }

  function stop() { sessionRuns[activeId]?.controller.abort(); }
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
    const run = { controller: new AbortController(), messages: runMessages, retryingIndex: runRetryingIndex };
    if (continuousVoiceEnabled && settings?.tts?.enabled && settings.tts.auto_play) {
      run.speechChunker = createSpeechChunker();
      run.speechSession = createSpeechSession(`live:${sessionId}:${Date.now()}`, sessionId);
    }
    sessionRuns = { ...sessionRuns, [sessionId]: run };
    publishMessages(sessionId, runMessages);
    return run;
  }
  async function finishSessionRun(sessionId, run) {
    if (sessionRuns[sessionId] !== run) return;
    const canAutoPlay = run.completed && activeId === sessionId && settings?.tts?.enabled && settings.tts.auto_play
      && !continuousAutoSendPending[sessionId];
    if (run.speechSession) {
      if (canAutoPlay) {
        for (const chunk of run.speechChunker?.finish() || []) enqueueSpeech(run.speechSession, chunk);
        closeSpeechSession(run.speechSession);
      } else if (speechSession === run.speechSession) {
        stopReplySpeech();
      }
    }
    const next = { ...sessionRuns };
    delete next[sessionId];
    sessionRuns = next;
    const reply = messageCache[sessionId]?.[run.retryingIndex];
    if (canAutoPlay && !run.speechSession && reply?.role === 'assistant' && reply.content) {
      speakReply(reply);
    }
    await flushContinuousAutoSend(sessionId);
    await focusComposer(sessionId);
  }

  async function focusComposer(sessionId = activeId) {
    await tick();
    if (activeId === sessionId && !sessionRuns[sessionId] && !settingsOpen && !controlsOpen && !continuousVoiceEnabled) {
      composerInput?.focus({ preventScroll: true });
    }
  }
  function publishMessages(sessionId, nextMessages) {
    messageCache = { ...messageCache, [sessionId]: nextMessages };
    if (activeId === sessionId) {
      messages = nextMessages;
      scrollBottom();
    }
  }
  function setSessionError(sessionId, message) {
    sessionErrors = { ...sessionErrors, [sessionId]: message };
    if (activeId === sessionId) error = message;
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
      if (!run?.speechChunker || !run.speechSession) return;
      for (const chunk of run.speechChunker.push(delta)) enqueueSpeech(run.speechSession, chunk);
    };
    const handleToolApproval = handlers.toolApproval;
    handlers.toolApproval = (data) => {
      handleToolApproval(data);
      if (activeId === sessionId) requestAnimationFrame(() => document.querySelector(`[data-approval-id="${data.approval_id}"]`)?.scrollIntoView({ behavior: 'smooth', block: 'center' }));
    };
    handlers.context = (next) => { if (activeId === sessionId) contextState = next; };
    handlers.sshGrantChanged = () => { refreshSSHGrants(sessionId); };
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
    const dropped = Array.from(files || []);
    const mediaFiles = dropped.filter(isSupportedAttachmentFile);
    if (!mediaFiles.length || running || !activeId) {
      if (dropped.length && !running) error = '지원되는 이미지·음성·비디오 파일만 첨부할 수 있습니다.';
      return;
    }
    const oversized = mediaFiles.find((file) => file.size > (attachmentKind(file) === 'image' ? maxImageBytes : maxAttachmentBytes));
    if (oversized) {
      error = `${oversized.name}: ${attachmentKind(oversized) === 'image' ? '이미지는 15MB' : '음성·비디오는 64MB'} 이하여야 합니다.`;
      return;
    }
    const sessionId = activeId;
    const draft = attachmentDrafts[sessionId] || [];
    const queued = attachmentQueue.filter((item) => item.sessionId === sessionId).length;
    if (draft.length + queued + mediaFiles.length > 6) {
      error = '미디어는 한 메시지에 최대 6개까지 첨부할 수 있습니다.';
      return;
    }
    const totalBytes = draft.reduce((sum, item) => sum + (item.size || 0), 0)
      + attachmentQueue.filter((item) => item.sessionId === sessionId).reduce((sum, item) => sum + (item.file.size || 0), 0)
      + mediaFiles.reduce((sum, file) => sum + (file.size || 0), 0);
    if (totalBytes > maxMessageBytes) {
      error = '한 메시지의 첨부 파일 합계는 96MB 이하여야 합니다.';
      return;
    }
    attachmentDrafts = { ...attachmentDrafts, [sessionId]: draft };
    attachmentQueue = [...attachmentQueue, ...mediaFiles.map((file) => ({ file, sessionId }))];
    syncActiveAttachmentState();
    processAttachmentQueue();
  }
  async function processAttachmentQueue() {
    if (queueProcessing) return;
    queueProcessing = true;
    try {
      while (attachmentQueue.length) {
        const item = attachmentQueue[0];
        uploadingSessionId = item.sessionId;
        syncActiveAttachmentState();
        if (item.sessionId === activeId) error = '';
        const uploadController = new AbortController();
        const timeout = setTimeout(() => uploadController.abort(), 120000);
        try {
          const attachment = await uploadAttachment(item.file, uploadController.signal);
          if (Object.prototype.hasOwnProperty.call(attachmentDrafts, item.sessionId)) {
            setPendingAttachments(item.sessionId, [...(attachmentDrafts[item.sessionId] || []), attachment]);
          }
        } catch (e) {
          if (item.sessionId === activeId) error = e.name === 'AbortError' ? '미디어 업로드 시간이 초과되었습니다.' : e.message;
        } finally {
          clearTimeout(timeout);
          attachmentQueue = attachmentQueue.filter((queued) => queued !== item);
          uploadingSessionId = '';
          syncActiveAttachmentState();
        }
      }
    } finally {
      queueProcessing = false;
      uploadingSessionId = '';
      syncActiveAttachmentState();
      if (attachmentInput) attachmentInput.value = '';
    }
  }
  async function addMediaURL(rawURL) {
    if (!activeId || running || uploadingAttachments) return false;
    try {
      const parsed = new URL(rawURL);
      if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') throw new Error('HTTP 또는 HTTPS 주소를 입력하세요.');
    } catch (e) {
      error = e.message === 'HTTP 또는 HTTPS 주소를 입력하세요.' ? e.message : '올바른 영상 주소를 입력하세요.';
      return false;
    }
    const sessionId = activeId;
    const draft = attachmentDrafts[sessionId] || [];
    if (draft.length >= 6) {
      error = '미디어는 한 메시지에 최대 6개까지 첨부할 수 있습니다.';
      return false;
    }
    uploadingSessionId = sessionId;
    sourceDownloadingSessionId = sessionId;
    syncActiveAttachmentState();
    error = '';
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 30 * 60 * 1000);
    try {
      const attachment = await uploadMediaURL(rawURL, controller.signal);
      const current = attachmentDrafts[sessionId] || [];
      const totalBytes = current.reduce((sum, item) => sum + (item.size || 0), 0) + (attachment.size || 0);
      if (current.length >= 6 || totalBytes > maxMessageBytes) {
        error = '한 메시지의 첨부 파일 합계는 96MB 이하여야 합니다.';
        return false;
      }
      setPendingAttachments(sessionId, [...current, attachment]);
      return true;
    } catch (e) {
      if (sessionId === activeId) error = e.name === 'AbortError' ? 'URL 미디어 취득 시간이 초과되었습니다.' : e.message;
      return false;
    } finally {
      clearTimeout(timeout);
      uploadingSessionId = '';
      sourceDownloadingSessionId = '';
      syncActiveAttachmentState();
    }
  }
  function setPendingAttachments(sessionId, items) {
    attachmentDrafts = { ...attachmentDrafts, [sessionId]: items };
    if (sessionId === activeId) pendingAttachments = items;
  }
  function syncActiveAttachmentState() {
    pendingAttachments = attachmentDrafts[activeId] || [];
    uploadingAttachments = Boolean(activeId) && (uploadingSessionId === activeId || attachmentQueue.some((item) => item.sessionId === activeId));
  }
  function removePendingAttachment(id) {
    if (running) return;
    setPendingAttachments(activeId, pendingAttachments.filter((item) => item.id !== id));
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
      normalizePromptPresetSettings(settings);
      settingsOpen = true;
      closeSidebarOnMobile();
      closeControls();
    } catch (e) { error = e.message; }
  }

  function normalizePromptPresetSettings(config) {
    if (!config?.model) return;
    if (!Array.isArray(config.model.system_prompt_presets)) config.model.system_prompt_presets = [];
    if (!config.model.system_prompt_preset) config.model.system_prompt_preset = '';
    if (!config.context) config.context = { enabled: true, window_tokens: 0, compact_at_percent: 80, output_reserve: 8192, safety_margin: 4096, recent_tokens: 32768, image_tokens: 2048 };
    if (!config.asr) config.asr = { enabled: true, ffmpeg_endpoint: 'http://127.0.0.1:8698', endpoint: 'http://127.0.0.1:8694', model: 'qwen3-asr', language: 'auto', prompt: '', filter_fillers: true, timeout: '30m' };
    if (config.asr.filter_fillers === undefined) config.asr.filter_fillers = true;
    if (!config.tts) config.tts = { enabled: true, endpoint: 'http://127.0.0.1:8692', model: 'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice', language: 'Korean', voice: 'Sohee', instructions: '', seed: -1, auto_play: false, timeout: '10m' };
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

  function releaseReplySpeech(session, resumeVoice = true) {
    if (speechSession !== session) return;
    speechSession = null;
    speechLoadingKey = '';
    speechPlayingKey = '';
    if (speechPausedContinuousVoice) {
      speechPausedContinuousVoice = false;
      if (resumeVoice) continuousVoiceListener?.setPaused(false);
    }
  }

  function stopReplySpeech() {
    const session = speechSession;
    if (!session) return;
    session.stopped = true;
    session.controller.abort();
    session.player?.stop();
    releaseReplySpeech(session, true);
  }

  function createSpeechSession(key, sessionId = activeId) {
    const session = {
      key, sessionId, controller: new AbortController(), player: null,
      queue: [], processing: false, closed: false, stopped: false,
    };
    speechSession = session;
    return session;
  }

  function enqueueSpeech(session, text) {
    const value = String(text || '').trim();
    if (!value || session.stopped || session.closed || speechSession !== session) return;
    session.queue.push(value);
    if (!session.player) speechLoadingKey = session.key;
    processSpeechQueue(session);
  }

  function closeSpeechSession(session) {
    if (session.stopped || speechSession !== session) return;
    session.closed = true;
    processSpeechQueue(session);
  }

  async function processSpeechQueue(session) {
    if (session.processing || session.stopped || speechSession !== session) return;
    session.processing = true;
    try {
      while (session.queue.length && !session.stopped) {
        const text = session.queue.shift();
        await streamSpeech(text, session.controller.signal, async (bytes, sampleRate) => {
          if (session.stopped || speechSession !== session) return;
          if (!session.player) {
            session.player = new PCMStreamPlayer({
              sampleRate,
              onStart: () => {
                if (speechSession !== session) return;
                speechLoadingKey = '';
                speechPlayingKey = session.key;
                if (continuousVoiceEnabled && continuousVoiceListener) {
                  continuousVoiceListener.setPaused(true);
                  speechPausedContinuousVoice = true;
                }
              },
            });
          }
          await session.player.append(bytes);
        });
      }
      if (session.closed && !session.stopped) {
        if (session.player) await session.player.finish();
        releaseReplySpeech(session, true);
      }
    } catch (speechError) {
      if (!session.stopped && speechError?.name !== 'AbortError') {
        setSessionError(session.sessionId, speechError?.message || '답변 음성 생성에 실패했습니다.');
      }
      session.player?.stop();
      releaseReplySpeech(session, true);
    } finally {
      session.processing = false;
    }
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
    const session = createSpeechSession(key);
    for (const chunk of text.split('\n').map((value) => value.trim()).filter(Boolean)) {
      enqueueSpeech(session, chunk);
    }
    closeSpeechSession(session);
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
