<script>
  import { onMount, tick } from 'svelte';
  import DOMPurify from 'dompurify';
  import { marked } from 'marked';
  import {
    listSessions, createSession, deleteSession, renameSession, listMessages, streamChat,
    getHealth, getModels, getConfig, saveConfig, retryMessage, editMessage as editChatMessage, uploadImage,
    getMediaUsage, cleanupMedia, setSessionGroup, listGroups, createGroup, renameGroup, moveGroup, deleteGroup,
  } from './api.js';

  let sessions = [];
  let groups = [];
  let ungroupedSessions = [];
  let sessionsByGroup = {};
  let collapsedGroups = {};
  let sessionMenuId = '';
  let activeId = '';
  let messages = [];
  let reasoningOpen = {};
  let models = [];
  let selectedModel = '';
  let reasoningEffort = '';
  let webToolsEnabled = false;
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
  let settingsAPIKey = '';
  let clearAPIKey = false;
  let settingsNotice = '';
  let editingMessageId = null;
  let editInput = '';
  let pendingImages = [];
  let uploadingImages = false;
  let imageDrafts = {};
  let imageInput;
  let dragActive = false;
  let dragResetTimer;
  let imageQueue = [];
  let queueProcessing = false;
  let uploadingSessionId = '';
  let mediaUsage = null;
  let cleaningMedia = false;
  let editingTitle = false;
  let titleInput = '';
  let titleEditor;
  let titleSaving = false;
  let controlsOpen = false;

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
    const mobile = window.matchMedia('(max-width: 600px)').matches;
    sidebarOpen = mobile ? false : localStorage.getItem('sparktalk.sidebar-open') !== 'false';
    sidebarWidth = Number(localStorage.getItem('sparktalk.sidebar-width')) || 260;
    try { collapsedGroups = JSON.parse(localStorage.getItem('sparktalk.collapsed-groups') || '{}'); } catch { collapsedGroups = {}; }
    load();
    const timer = setInterval(refreshHealth, 15000);
    window.addEventListener('dragenter', onWindowDragEnter, true);
    window.addEventListener('dragover', onWindowDragOver, true);
    window.addEventListener('dragleave', onWindowDragLeave, true);
    return () => {
      clearInterval(timer);
      clearTimeout(dragResetTimer);
      window.removeEventListener('dragenter', onWindowDragEnter, true);
      window.removeEventListener('dragover', onWindowDragOver, true);
      window.removeEventListener('dragleave', onWindowDragLeave, true);
    };
  });

  async function load() {
    try {
      const cfg = await getConfig();
      normalizePromptPresetSettings(cfg);
      settings = cfg;
      reasoningEffort = cfg.model.reasoning_effort || '';
      webToolsEnabled = cfg.tools?.enabled ?? false;
      await Promise.all([refreshModels(), refreshHealth()]);
      selectedModel = cfg.model.default_model || models[0] || '';
      [groups, sessions] = await Promise.all([listGroups(), listSessions()]);
      if (sessions.length) await select(sessions[0].id);
      else {
        activeId = '';
        messages = [];
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
    sessionMenuId = '';
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
    if (activeId && messages) messageCache = { ...messageCache, [activeId]: messages };
    activeId = id;
    syncActiveImageState();
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
    error = sessionErrors[id] || '';
    await scrollBottom(true);
    closeSidebarOnMobile();
  }

  async function remove(id) {
    if (sessionRuns[id]) return;
    if (!confirm('이 대화를 삭제할까요?')) return;
    sessionMenuId = '';
    const previousSessions = sessions;
    sessions = sessions.filter((item) => item.id !== id);
    try {
      await deleteSession(id);
      const nextDrafts = { ...imageDrafts };
      delete nextDrafts[id];
      imageDrafts = nextDrafts;
      imageQueue = imageQueue.filter((item) => item.sessionId !== id);
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
          pendingImages = [];
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
    if (!content || running || uploadingImages || !activeId) return;
    const sessionId = activeId;
    const attachments = pendingImages;
    input = '';
    setPendingImages(sessionId, []);
    setSessionError(sessionId, '');
    const chatMessages = [...messages,
      { role: 'user', content, reasoning_content: '', attachments },
      { role: 'assistant', content: '', reasoning_content: '', tool_trace: [], activity: '' },
    ];
    const replyIndex = chatMessages.length - 1;
    const run = startSessionRun(sessionId, chatMessages, replyIndex);
    await scrollBottom(true);
    try {
      await streamChat(sessionId, content, attachments, selectedModel, reasoningEffort, webToolsEnabled, run.controller.signal, {
        reasoning(delta) {
          const reply = run.messages[replyIndex];
          reply.activity = 'reasoning';
          reply.reasoning_content += delta;
          publishMessages(sessionId, run.messages);
        },
        delta(delta) {
          const reply = run.messages[replyIndex];
          reply.activity = 'answer';
          reply.content += delta;
          publishMessages(sessionId, run.messages);
        },
        toolStart(data) { addToolStart(run.messages[replyIndex], data, sessionId, run.messages); },
        toolResult(data) { finishTool(run.messages[replyIndex], data, sessionId, run.messages); },
      });
      publishMessages(sessionId, hydrateMessages(await listMessages(sessionId)));
      await refreshSessions();
      setTimeout(refreshSessions, 1800);
    } catch (e) {
      if (e.name !== 'AbortError') setSessionError(sessionId, e.message);
    } finally {
      finishSessionRun(sessionId, run);
    }
  }

  function onKeydown(event) {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); send(); }
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
      await retryMessage(message.id, selectedModel, reasoningEffort, webToolsEnabled, userVariant, run.controller.signal, {
        reasoning(delta) {
          message.activity = 'reasoning';
          message.reasoning_content += delta;
          publishMessages(sessionId, run.messages);
        },
        delta(delta) {
          message.activity = 'answer';
          message.content += delta;
          publishMessages(sessionId, run.messages);
        },
        toolStart(data) { addToolStart(message, data, sessionId, run.messages); },
        toolResult(data) { finishTool(message, data, sessionId, run.messages); },
      });
      const updated = hydrateMessages(await listMessages(sessionId));
      const parent = updated[index - 1];
      const answer = updated[index];
      if (parent?.role === 'user' && answer?.role === 'assistant' && parent.variants?.[userVariant]) {
        applyVariant(parent, userVariant, index - 1, sessionId);
        const matching = variantIndices(answer, index, updated);
        if (matching.length) applyVariant(answer, matching[matching.length - 1], index, sessionId);
      }
      publishMessages(sessionId, updated);
      await refreshSessions();
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
    if (!content || running || !message.id || content === message.content) return;
    if (index < messages.length - 2 && !confirm('이 질문을 수정하면 이후 대화가 새 분기로 바뀝니다. 계속할까요?')) return;
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
      await editChatMessage(message.id, content, message.attachments || [], selectedModel, reasoningEffort, webToolsEnabled, run.controller.signal, {
        reasoning(delta) {
          run.messages[replyIndex].activity = 'reasoning';
          run.messages[replyIndex].reasoning_content += delta;
          publishMessages(sessionId, run.messages);
        },
        delta(delta) {
          run.messages[replyIndex].activity = 'answer';
          run.messages[replyIndex].content += delta;
          publishMessages(sessionId, run.messages);
        },
        toolStart(data) { addToolStart(run.messages[replyIndex], data, sessionId, run.messages); },
        toolResult(data) { finishTool(run.messages[replyIndex], data, sessionId, run.messages); },
      });
      publishMessages(sessionId, hydrateMessages(await listMessages(sessionId)));
      await refreshSessions();
      setTimeout(refreshSessions, 1800);
      editInput = '';
    } catch (e) {
      publishMessages(sessionId, originalMessages);
      if (e.name !== 'AbortError') setSessionError(sessionId, e.message);
    } finally {
      finishSessionRun(sessionId, run);
    }
  }

  function startSessionRun(sessionId, runMessages, runRetryingIndex) {
    const run = { controller: new AbortController(), messages: runMessages, retryingIndex: runRetryingIndex };
    sessionRuns = { ...sessionRuns, [sessionId]: run };
    publishMessages(sessionId, runMessages);
    return run;
  }
  function finishSessionRun(sessionId, run) {
    if (sessionRuns[sessionId] !== run) return;
    const next = { ...sessionRuns };
    delete next[sessionId];
    sessionRuns = next;
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
  function setReasoningOpen(index, open) {
    reasoningOpen = { ...reasoningOpen, [index]: open };
  }
  function hydrateMessages(items) {
    const hydrated = items.map((item) => ({
      ...item,
      variant_index: Math.max(0, (item.variants?.length || 1) - 1),
      activity: '',
    }));
    for (let index = 1; index < hydrated.length; index += 1) {
      const message = hydrated[index];
      const parent = hydrated[index - 1];
      if (message.role !== 'assistant' || parent?.role !== 'user' || !message.variants?.length) continue;
      const matching = message.variants
        .map((variant, variantIndex) => ({ variant, variantIndex }))
        .filter(({ variant }) => (variant.parent_variant ?? 0) === (parent.variant_index ?? 0));
      if (!matching.length) continue;
      const selected = matching[matching.length - 1];
      message.variant_index = selected.variantIndex;
      message.content = selected.variant.content || '';
      message.reasoning_content = selected.variant.reasoning_content || '';
      message.tool_trace = selected.variant.tool_trace || [];
    }
    return hydrated;
  }
  function variantIndices(message, messageIndex, messageList = messages) {
    const indices = (message.variants || []).map((_, index) => index);
    if (message.role !== 'assistant') return indices;
    const parent = messageList[messageIndex - 1];
    if (parent?.role !== 'user') return indices;
    return indices.filter((index) => (message.variants[index].parent_variant ?? 0) === (parent.variant_index ?? 0));
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
    const variant = message.variants?.[variantIndex];
    if (!variant) return;
    message.content = variant.content || '';
    message.reasoning_content = variant.reasoning_content || '';
    message.tool_trace = variant.tool_trace || [];
    message.attachments = variant.attachments || [];
    message.variant_index = variantIndex;
    if (sessionId === activeId) reasoningOpen = { ...reasoningOpen, [messageIndex]: false };
  }
  function showVariant(message, variantIndex, messageIndex) {
    if (running || !message.variants?.[variantIndex]) return;
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
  function addToolStart(message, data, sessionId = activeId, messageList = messages) {
    message.activity = 'tool';
    message.tool_trace = [...(message.tool_trace || []), { ...data, result: '', error: '', running: true }];
    publishMessages(sessionId, messageList);
  }
  function finishTool(message, data, sessionId = activeId, messageList = messages) {
    const trace = [...(message.tool_trace || [])];
    const index = trace.findIndex((item) => item.id === data.id && item.running);
    if (index >= 0) trace[index] = { ...trace[index], ...data, running: false };
    else trace.push({ ...data, running: false });
    message.tool_trace = trace;
    message.activity = 'reasoning';
    publishMessages(sessionId, messageList);
  }
  function addImageFiles(files) {
    const dropped = Array.from(files || []);
    const images = dropped.filter(isSupportedImageFile);
    if (!images.length || running || !activeId) {
      if (dropped.length && !running) error = 'PNG, JPEG, WebP 이미지 파일만 첨부할 수 있습니다.';
      return;
    }
    const sessionId = activeId;
    const draft = imageDrafts[sessionId] || [];
    const queued = imageQueue.filter((item) => item.sessionId === sessionId).length;
    if (draft.length + queued + images.length > 6) {
      error = '이미지는 한 메시지에 최대 6개까지 첨부할 수 있습니다.';
      return;
    }
    imageDrafts = { ...imageDrafts, [sessionId]: draft };
    imageQueue = [...imageQueue, ...images.map((file) => ({ file, sessionId }))];
    syncActiveImageState();
    processImageQueue();
  }
  function isSupportedImageFile(file) {
    const mime = (file.type || '').toLowerCase();
    const supportedMime = ['image/png', 'image/jpeg', 'image/webp'].includes(mime);
    const supportedExtension = /\.(png|jpe?g|webp)$/i.test(file.name || '');
    // Linux file managers and Chromium variants sometimes report a generic or
    // incorrect MIME type. The server validates the actual file bytes.
    return supportedMime || supportedExtension;
  }
  async function processImageQueue() {
    if (queueProcessing) return;
    queueProcessing = true;
    try {
      while (imageQueue.length) {
        const item = imageQueue[0];
        uploadingSessionId = item.sessionId;
        syncActiveImageState();
        if (item.sessionId === activeId) error = '';
        const uploadController = new AbortController();
        const timeout = setTimeout(() => uploadController.abort(), 30000);
        try {
          const attachment = await uploadImage(item.file, uploadController.signal);
          if (Object.prototype.hasOwnProperty.call(imageDrafts, item.sessionId)) {
            setPendingImages(item.sessionId, [...(imageDrafts[item.sessionId] || []), attachment]);
          }
        } catch (e) {
          if (item.sessionId === activeId) error = e.name === 'AbortError' ? '이미지 업로드 시간이 초과되었습니다.' : e.message;
        } finally {
          clearTimeout(timeout);
          imageQueue = imageQueue.filter((queued) => queued !== item);
          uploadingSessionId = '';
          syncActiveImageState();
        }
      }
    } finally {
      queueProcessing = false;
      uploadingSessionId = '';
      syncActiveImageState();
      if (imageInput) imageInput.value = '';
    }
  }
  function setPendingImages(sessionId, items) {
    imageDrafts = { ...imageDrafts, [sessionId]: items };
    if (sessionId === activeId) pendingImages = items;
  }
  function syncActiveImageState() {
    pendingImages = imageDrafts[activeId] || [];
    uploadingImages = Boolean(activeId) && (uploadingSessionId === activeId || imageQueue.some((item) => item.sessionId === activeId));
  }
  function removePendingImage(id) {
    if (running) return;
    setPendingImages(activeId, pendingImages.filter((item) => item.id !== id));
  }
  function onPaste(event) {
    const files = Array.from(event.clipboardData?.files || []);
    if (files.some((file) => file.type.startsWith('image/'))) addImageFiles(files);
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
    if (settingsOpen || running) return;
    if (event.target !== imageInput) event.preventDefault();
    showDropOverlay();
  }
  function onWindowDragOver(event) {
    if (settingsOpen || running) return;
    // Once the native file input covers the viewport, let Chromium perform
    // its built-in file drop rather than cancelling that default action.
    if (event.target !== imageInput) event.preventDefault();
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
  function onImageInputChange(event) {
    const files = Array.from(event.currentTarget.files || []);
    clearTimeout(dragResetTimer);
    dragActive = false;
    addImageFiles(files);
  }
  function collapseDetails(event) {
    const details = event.currentTarget.closest('details');
    if (details) details.open = false;
  }
  function toolArgument(tool) {
    try {
      const args = JSON.parse(tool.arguments || '{}');
      return args.query || args.url || tool.arguments;
    } catch (e) { return tool.arguments || ''; }
  }
  function toolPreview(tool) {
    if (!tool.result) return '';
    try {
      const result = JSON.parse(tool.result);
      if (Array.isArray(result.results)) {
        return result.results.map((item) => `${item.title}\n${item.url}`).join('\n\n').slice(0, 1800);
      }
      if (result.content) return result.content.slice(0, 1800);
    } catch (e) {}
    return tool.result.slice(0, 1800);
  }
  function render(text) { return DOMPurify.sanitize(marked.parse(text || '')); }
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
    settingsNotice = '';
    settingsAPIKey = '';
    clearAPIKey = false;
    settings = await getConfig();
    normalizePromptPresetSettings(settings);
    settingsOpen = true;
    closeSidebarOnMobile();
    closeControls();
    try { mediaUsage = await getMediaUsage(); }
    catch (e) { settingsNotice = e.message; }
  }

  function normalizePromptPresetSettings(config) {
    if (!config?.model) return;
    if (!Array.isArray(config.model.system_prompt_presets)) config.model.system_prompt_presets = [];
    if (!config.model.system_prompt_preset) config.model.system_prompt_preset = '';
  }

  function selectPromptPreset(event) {
    const name = event.currentTarget.value;
    settings.model.system_prompt_preset = name;
    if (!name) return;
    const preset = settings.model.system_prompt_presets.find((item) => item.name === name);
    if (preset) settings.model.system_prompt = preset.prompt;
  }

  function addPromptPreset() {
    const name = window.prompt('새 시스템 프롬프트 프리셋 이름을 입력하세요.');
    if (name === null || !name.trim()) return;
    const trimmed = name.trim();
    if (settings.model.system_prompt_presets.some((item) => item.name === trimmed)) {
      settingsNotice = '같은 이름의 시스템 프롬프트 프리셋이 있습니다.';
      return;
    }
    settings.model.system_prompt_presets = [
      ...settings.model.system_prompt_presets,
      { name: trimmed, prompt: settings.model.system_prompt || '' },
    ];
    settings.model.system_prompt_preset = trimmed;
    settingsNotice = `'${trimmed}' 프리셋을 추가했습니다. 설정 저장을 누르면 반영됩니다.`;
  }

  function savePromptPreset() {
    const name = settings.model.system_prompt_preset;
    if (!name) {
      addPromptPreset();
      return;
    }
    settings.model.system_prompt_presets = settings.model.system_prompt_presets.map((item) =>
      item.name === name ? { ...item, prompt: settings.model.system_prompt || '' } : item);
    settingsNotice = `'${name}' 프리셋의 내용을 갱신했습니다. 설정 저장을 누르면 반영됩니다.`;
  }

  function renamePromptPreset() {
    const current = settings.model.system_prompt_preset;
    if (!current) return;
    const name = window.prompt('프리셋 이름을 수정하세요.', current);
    if (name === null || !name.trim() || name.trim() === current) return;
    const trimmed = name.trim();
    if (settings.model.system_prompt_presets.some((item) => item.name === trimmed)) {
      settingsNotice = '같은 이름의 시스템 프롬프트 프리셋이 있습니다.';
      return;
    }
    settings.model.system_prompt_presets = settings.model.system_prompt_presets.map((item) =>
      item.name === current ? { ...item, name: trimmed } : item);
    settings.model.system_prompt_preset = trimmed;
    settingsNotice = `'${trimmed}'으로 이름을 변경했습니다. 설정 저장을 누르면 반영됩니다.`;
  }

  function removePromptPreset() {
    const name = settings.model.system_prompt_preset;
    if (!name || !confirm(`'${name}' 시스템 프롬프트 프리셋을 삭제할까요? 현재 프롬프트 내용은 유지됩니다.`)) return;
    settings.model.system_prompt_presets = settings.model.system_prompt_presets.filter((item) => item.name !== name);
    settings.model.system_prompt_preset = '';
    settingsNotice = `'${name}' 프리셋을 삭제했습니다. 설정 저장을 누르면 반영됩니다.`;
  }

  function promptPresetDirty() {
    const name = settings?.model?.system_prompt_preset;
    if (!name) return false;
    const preset = settings.model.system_prompt_presets.find((item) => item.name === name);
    return Boolean(preset) && preset.prompt !== (settings.model.system_prompt || '');
  }

  function formatBytes(value) {
    if (!value) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB'];
    const index = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1);
    return `${(value / (1024 ** index)).toFixed(index ? 1 : 0)} ${units[index]}`;
  }

  async function removeUnusedMedia() {
    if (cleaningMedia || !mediaUsage?.unused_files) return;
    if (!confirm(`대화에서 사용하지 않는 이미지 ${mediaUsage.unused_files}개를 삭제할까요?`)) return;
    cleaningMedia = true;
    settingsNotice = '';
    try {
      const keepIds = Object.values(imageDrafts).flat().map((item) => item.id);
      const result = await cleanupMedia(keepIds);
      mediaUsage = result.usage;
      settingsNotice = `미사용 이미지 ${result.removed.files}개(${formatBytes(result.removed.bytes)})를 정리했습니다.`;
    } catch (e) { settingsNotice = e.message; }
    finally { cleaningMedia = false; }
  }

  async function persistSettings() {
    try {
      const result = await saveConfig({
        server: settings.server,
        model: settings.model,
        tools: settings.tools,
        api_key: settingsAPIKey,
        clear_api_key: clearAPIKey,
      });
      settings = result.config;
      reasoningEffort = settings.model.reasoning_effort || reasoningEffort;
      webToolsEnabled = settings.tools?.enabled ?? false;
      await Promise.all([refreshModels(), refreshHealth()]);
      if (settings.model.default_model) selectedModel = settings.model.default_model;
      settingsNotice = result.restart_required
        ? '저장했습니다. 주소 또는 DB 변경은 앱을 재시작하면 반영됩니다.'
        : '저장했으며 즉시 반영했습니다.';
      settingsAPIKey = '';
      clearAPIKey = false;
    } catch (e) { settingsNotice = e.message; }
  }
</script>

<div class="shell" style:grid-template-columns={sidebarOpen ? `${sidebarWidth}px 1fr` : '1fr'}>
  {#if sidebarOpen}
    <aside>
      <div class="brand"><span class="mark">S</span><strong>SparkTalk</strong><button class="sidebar-close" onclick={closeSidebar} aria-label="사이드바 닫기">×</button></div>
      <div class="sidebar-actions">
        <button class="new-chat" onclick={addSession}>＋ 새 대화</button>
        <button class="new-group" onclick={addGroup} title="그룹 만들기" aria-label="그룹 만들기">＋ 폴더</button>
      </div>
      <nav>
        {#each groups as group, groupIndex}
          <section class="chat-group">
            <div class="group-heading">
              <button class="group-toggle" onclick={() => toggleGroup(group.id)} aria-expanded={!collapsedGroups[group.id]}>
                <span>{collapsedGroups[group.id] ? '▸' : '▾'} 📁 {group.name}</span><small>{(sessionsByGroup[group.id] || []).length}</small>
              </button>
              <div class="group-actions">
                <button onclick={() => reorderGroup(group, 'up')} disabled={groupIndex === 0} title="위로 이동">↑</button>
                <button onclick={() => reorderGroup(group, 'down')} disabled={groupIndex === groups.length - 1} title="아래로 이동">↓</button>
                <button onclick={() => editGroup(group)} title="이름 변경">✎</button>
                <button class="danger" onclick={() => removeGroup(group)} title="그룹 삭제">×</button>
              </div>
            </div>
            {#if !collapsedGroups[group.id]}
              {#each sessionsByGroup[group.id] || [] as session}
                <div class="session-row" class:active={session.id === activeId} class:generating={Boolean(sessionRuns[session.id])}>
                  <button class="session-select" onclick={() => select(session.id)}>{session.title}</button>
                  {#if sessionRuns[session.id]}<span class="session-running" title="답변 생성 중" aria-label="답변 생성 중">●</span>{/if}
                  <button class="session-more" onclick={() => sessionMenuId = sessionMenuId === session.id ? '' : session.id} aria-label={`${session.title} 메뉴`}>⋯</button>
                  {#if sessionMenuId === session.id}
                    <div class="session-menu">
                      <strong>그룹 이동</strong>
                      <button onclick={() => changeSessionGroup(session, '')}>그룹 없음</button>
                      {#each groups as target}<button class:current={target.id === session.group_id} onclick={() => changeSessionGroup(session, target.id)}>▸ {target.name}</button>{/each}
                      <hr /><button class="danger" onclick={() => remove(session.id)} disabled={Boolean(sessionRuns[session.id])}>대화 삭제</button>
                    </div>
                  {/if}
                </div>
              {/each}
            {/if}
          </section>
        {/each}
        <section class="chat-group ungrouped">
          <button class="group-toggle" onclick={() => toggleGroup('__ungrouped__')} aria-expanded={!collapsedGroups.__ungrouped__}>
            <span>{collapsedGroups.__ungrouped__ ? '▸' : '▾'} 대화</span><small>{ungroupedSessions.length}</small>
          </button>
          {#if !collapsedGroups.__ungrouped__}
            {#each ungroupedSessions as session}
              <div class="session-row" class:active={session.id === activeId} class:generating={Boolean(sessionRuns[session.id])}>
                <button class="session-select" onclick={() => select(session.id)}>{session.title}</button>
                {#if sessionRuns[session.id]}<span class="session-running" title="답변 생성 중" aria-label="답변 생성 중">●</span>{/if}
                <button class="session-more" onclick={() => sessionMenuId = sessionMenuId === session.id ? '' : session.id} aria-label={`${session.title} 메뉴`}>⋯</button>
                {#if sessionMenuId === session.id}
                  <div class="session-menu">
                    <strong>그룹 이동</strong>
                    <button class:current={!session.group_id} onclick={() => changeSessionGroup(session, '')}>그룹 없음</button>
                    {#each groups as group}<button onclick={() => changeSessionGroup(session, group.id)}>▸ {group.name}</button>{/each}
                    <hr /><button class="danger" onclick={() => remove(session.id)} disabled={Boolean(sessionRuns[session.id])}>대화 삭제</button>
                  </div>
                {/if}
              </div>
            {/each}
          {/if}
        </section>
      </nav>
      <button class="settings-button" onclick={openSettings}>⚙ 설정</button>
      <button class="resize-handle" onpointerdown={startResize} aria-label="사이드바 폭 조절"></button>
    </aside>
    <button class="sidebar-backdrop" onclick={closeSidebar} aria-label="사이드바 닫기"></button>
  {/if}

  <main>
    <header>
      <button class="sidebar-toggle" onclick={toggleSidebar} aria-label="사이드바 열기 또는 닫기">☰</button>
      <div class="chat-heading">
        {#if editingTitle}
          <input class="title-editor" bind:this={titleEditor} bind:value={titleInput} maxlength="120" onkeydown={titleKeydown} onblur={saveTitle} aria-label="대화 제목" />
        {:else}
          <button class="chat-title" onclick={beginTitleEdit} disabled={!activeSession || running} title="대화 제목 수정"><span>{activeSession?.title || '새 대화'}</span><i>✎</i></button>
        {/if}
      </div>
      <div class="model-controls">
        <select bind:value={selectedModel} aria-label="모델 선택">
          {#if !models.length}<option value={selectedModel}>{selectedModel || '모델 없음'}</option>{/if}
          {#each models as model}<option value={model}>{model}</option>{/each}
        </select>
        <input bind:value={reasoningEffort} list="reasoning-levels" placeholder="reasoning effort" aria-label="Reasoning effort" />
        <button class:active={webToolsEnabled} class="web-toggle" onclick={() => webToolsEnabled = !webToolsEnabled} title="모델이 필요할 때 웹검색 사용">{webToolsEnabled ? '웹검색 자동' : '웹검색 꺼짐'}</button>
        <datalist id="reasoning-levels">
          <option value="none"></option><option value="minimal"></option><option value="low"></option>
          <option value="medium"></option><option value="high"></option><option value="xhigh"></option><option value="max"></option>
        </datalist>
        <span class:offline={health.status !== 'ok'} class="status">● {health.status === 'ok' ? '연결됨' : '연결 오류'}</span>
      </div>
      <button class="mobile-controls-toggle" class:active={controlsOpen} onclick={toggleControls} aria-label="모델 및 대화 설정" aria-expanded={controlsOpen}>☷</button>
    </header>
    {#if controlsOpen}
      <button class="controls-backdrop" onclick={closeControls} aria-label="모델 설정 패널 닫기"></button>
      <div class="controls-drawer" role="dialog" aria-modal="true" aria-label="모델 및 대화 설정">
        <div class="controls-title"><strong>대화 설정</strong><button onclick={closeControls} aria-label="닫기">×</button></div>
        <label>모델
          <select bind:value={selectedModel} aria-label="모델 선택">
            {#if !models.length}<option value={selectedModel}>{selectedModel || '모델 없음'}</option>{/if}
            {#each models as model}<option value={model}>{model}</option>{/each}
          </select>
        </label>
        <label>Reasoning effort<input bind:value={reasoningEffort} list="reasoning-levels" placeholder="reasoning effort" /></label>
        <button class:active={webToolsEnabled} class="drawer-web-toggle" onclick={() => webToolsEnabled = !webToolsEnabled}>{webToolsEnabled ? '웹검색 자동' : '웹검색 꺼짐'}</button>
        <div class="drawer-status"><span class:offline={health.status !== 'ok'}>● {health.status === 'ok' ? '연결됨' : '연결 오류'}</span><small>{selectedModel || health.model || '모델 확인 중'}</small></div>
      </div>
    {/if}
    <section class="messages" bind:this={messagePane}>
      {#if !messages.length}
        <div class="welcome"><div class="mark large">S</div><h1>무엇을 도와드릴까요?</h1><p>연결된 모델에 메시지를 보내보세요.</p></div>
      {/if}
      {#each messages as message, index}
        <article class:mine={message.role === 'user'}>
          <div class="avatar">{message.role === 'user' ? '나' : 'S'}</div>
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
                <div><button onclick={cancelEdit}>취소</button><button class="edit-submit" onclick={() => submitEdit(message, index)} disabled={!editInput.trim() || editInput.trim() === message.content}>수정 후 전송</button></div>
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
                    <button onclick={() => showAdjacentVariant(message, index, -1)} disabled={running || variantPosition(message, index) <= 0} aria-label="이전 답변">‹</button>
                    <span>{variantPosition(message, index) + 1}/{variantIndices(message, index).length}</span>
                    <button onclick={() => showAdjacentVariant(message, index, 1)} disabled={running || variantPosition(message, index) >= variantIndices(message, index).length - 1} aria-label="다음 답변">›</button>
                  </div>
                {/if}
                <button onclick={() => retry(message, index)} disabled={running || !message.id}>↻ 재시도</button>
              </div>
            {:else if message.id && editingMessageId !== message.id}
              <div class="message-actions user-actions">
                {#if message.variants?.length > 1}
                  <div class="variant-pager" aria-label="질문 버전 선택">
                    <button onclick={() => showAdjacentVariant(message, index, -1)} disabled={running || message.variant_index <= 0} aria-label="이전 질문">‹</button>
                    <span>{message.variant_index + 1}/{message.variants.length}</span>
                    <button onclick={() => showAdjacentVariant(message, index, 1)} disabled={running || message.variant_index >= message.variants.length - 1} aria-label="다음 질문">›</button>
                  </div>
                {/if}
                <button onclick={() => beginEdit(message)} disabled={running}>✎ 수정</button>
              </div>
            {/if}
          </div>
        </article>
      {/each}
    </section>
    {#if error}<div class="error">{error}</div>{/if}
    <footer>
      {#if pendingImages.length || uploadingImages}
        <div class="pending-images">
          {#each pendingImages as attachment}
            <div><img src={attachment.url} alt={attachment.name} /><button onclick={() => removePendingImage(attachment.id)} disabled={running} aria-label={`${attachment.name} 첨부 제거`}>×</button></div>
          {/each}
          {#if uploadingImages}<span class="uploading">이미지 업로드 중…</span>{/if}
        </div>
      {/if}
      <div class="composer" role="group" aria-label="메시지와 이미지 입력">
        <input class="image-input" class:drop-active={dragActive} bind:this={imageInput} type="file" accept="image/png,image/jpeg,image/webp" multiple onchange={onImageInputChange} />
        <button class="attach" onclick={() => imageInput?.click()} disabled={!activeId || running || uploadingImages || pendingImages.length >= 6} aria-label="이미지 첨부" title="이미지 첨부">＋</button>
        <textarea bind:value={input} onkeydown={onKeydown} onpaste={onPaste} placeholder={activeId ? '메시지를 입력하세요' : '새 대화를 만든 뒤 메시지를 입력하세요'} rows="1" disabled={!activeId || running}></textarea>
        {#if running}<button class="send stop" onclick={stop}>■</button>{:else}<button class="send" onclick={send} disabled={!activeId || !input.trim() || uploadingImages}>↑</button>{/if}
      </div>
      <small>이미지 붙여넣기·드래그 가능 · Enter 전송 · Shift+Enter 줄바꿈 · reasoning: {reasoningEffort || '서버 기본값'} · 웹: {webToolsEnabled ? '자동' : '꺼짐'}</small>
    </footer>
  </main>
</div>

{#if dragActive}
  <div class="drop-overlay" role="region" aria-label="이미지 드롭 영역"><div><span>＋</span><strong>이미지를 여기에 놓으세요</strong><small>PNG, JPEG, WebP · 최대 6개</small></div></div>
{/if}

{#if settingsOpen && settings}
  <div class="modal-backdrop" role="presentation" onclick={(e) => e.target === e.currentTarget && (settingsOpen = false)}>
    <div class="settings-modal" role="dialog" aria-modal="true" aria-labelledby="settings-title">
      <div class="modal-title"><h2 id="settings-title">설정</h2><button onclick={() => settingsOpen = false} aria-label="닫기">×</button></div>
      <label>API endpoint<input bind:value={settings.model.endpoint} placeholder="http://192.168.100.61:8000" /></label>
      <label>기본 모델<input bind:value={settings.model.default_model} list="model-list" placeholder="비우면 첫 모델 자동 선택" /></label>
      <datalist id="model-list">{#each models as model}<option value={model}></option>{/each}</datalist>
      <label>기본 reasoning effort<input bind:value={settings.model.reasoning_effort} list="reasoning-levels" placeholder="medium 또는 0.0~0.99" /></label>
      <fieldset class="prompt-presets">
        <legend>전역 시스템 프롬프트</legend>
        <label>프리셋
          <select value={settings.model.system_prompt_preset || ''} onchange={selectPromptPreset}>
            <option value="">직접 입력</option>
            {#each settings.model.system_prompt_presets as preset}<option value={preset.name}>{preset.name}</option>{/each}
          </select>
        </label>
        <textarea class="system-prompt" bind:value={settings.model.system_prompt} rows="6" placeholder="예: 모든 답변은 한국어 존댓말로 작성한다."></textarea>
        {#if promptPresetDirty()}<small class="preset-dirty">선택한 프리셋에서 내용이 변경되었습니다.</small>{/if}
        <div class="preset-actions">
          <button onclick={addPromptPreset}>＋ 새 프리셋</button>
          <button onclick={savePromptPreset}>{settings.model.system_prompt_preset ? '현재 내용 저장' : '현재 내용으로 만들기'}</button>
          <button onclick={renamePromptPreset} disabled={!settings.model.system_prompt_preset}>이름 변경</button>
          <button class="danger" onclick={removePromptPreset} disabled={!settings.model.system_prompt_preset}>삭제</button>
        </div>
      </fieldset>
      <fieldset>
        <legend>웹 도구</legend>
        <label class="check"><input type="checkbox" bind:checked={settings.tools.enabled} /> web_search / web_fetch 활성화</label>
        <label>최대 호출 라운드<input type="number" min="1" max="8" bind:value={settings.tools.max_rounds} /></label>
        <label>검색 결과 수<input type="number" min="1" max="10" bind:value={settings.tools.search_results} /></label>
        <label>도구 타임아웃<input bind:value={settings.tools.timeout} placeholder="15s" /></label>
      </fieldset>
      <label>Listen address<input bind:value={settings.server.listen_addr} placeholder="0.0.0.0:8585" /></label>
      <label>SQLite 파일<input bind:value={settings.server.database} placeholder="sparktalk.db" /></label>
      <fieldset>
        <legend>이미지 보관</legend>
        {#if mediaUsage}
          <div class="media-usage"><span>전체 {mediaUsage.files}개 · {formatBytes(mediaUsage.bytes)}</span><span>미사용 {mediaUsage.unused_files}개 · {formatBytes(mediaUsage.unused_bytes)}</span></div>
          <button class="media-cleanup" onclick={removeUnusedMedia} disabled={cleaningMedia || !mediaUsage.unused_files}>{cleaningMedia ? '정리 중…' : '미사용 이미지 정리'}</button>
        {:else}<span class="media-loading">보관 현황을 불러오는 중…</span>{/if}
        <small>현재 대화에 첨부됐거나 전송 대기 중인 이미지는 유지합니다.</small>
      </fieldset>
      <label>API key<input type="password" bind:value={settingsAPIKey} placeholder={settings.api_key_set ? '설정됨 — 변경할 때만 입력' : '선택 사항'} /></label>
      {#if settings.api_key_set}<label class="check"><input type="checkbox" bind:checked={clearAPIKey} /> 저장된 API key 제거</label>{/if}
      <p class="settings-help">Endpoint·모델·reasoning·시스템 프롬프트는 즉시 반영됩니다. Listen address와 DB 파일 변경은 재시작 후 반영됩니다.</p>
      {#if settingsNotice}<p class="settings-notice">{settingsNotice}</p>{/if}
      <div class="modal-actions"><button class="secondary" onclick={() => settingsOpen = false}>닫기</button><button class="primary" onclick={persistSettings}>저장</button></div>
    </div>
  </div>
{/if}
