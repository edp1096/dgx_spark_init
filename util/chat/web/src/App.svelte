<script>
  import { onMount, tick } from 'svelte';
  import SettingsModal from './components/SettingsModal.svelte';
  import Sidebar from './components/Sidebar.svelte';
  import ChatHeader from './components/ChatHeader.svelte';
  import Composer from './components/Composer.svelte';
  import MessageList from './components/MessageList.svelte';
  import { hydrateMessages, variantIndices as getVariantIndices, applyVariant as applyMessageVariant } from './lib/message-variants.js';
  import { createStreamHandlers } from './lib/chat-stream.js';
  import {
    listSessions, createSession, deleteSession, renameSession, listMessages, streamChat,
    getHealth, getModels, getConfig, retryMessage, editMessage as editChatMessage, uploadImage,
    setSessionGroup, listGroups, createGroup, renameGroup, moveGroup, deleteGroup,
  } from './api.js';

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
  let pendingImages = [];
  let uploadingImages = false;
  let imageDrafts = {};
  let imageInput;
  let dragActive = false;
  let dragResetTimer;
  let imageQueue = [];
  let queueProcessing = false;
  let uploadingSessionId = '';
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
      appearance = cfg.appearance || appearance;
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
      await streamChat(sessionId, content, attachments, selectedModel, reasoningEffort, webToolsEnabled, run.controller.signal,
        streamHandlersFor(run.messages[replyIndex], sessionId, run.messages));
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
      await editChatMessage(message.id, content, message.attachments || [], selectedModel, reasoningEffort, webToolsEnabled, run.controller.signal,
        streamHandlersFor(run.messages[replyIndex], sessionId, run.messages));
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
    return createStreamHandlers(message, () => publishMessages(sessionId, messageList));
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
  }

  async function applySavedSettings(next) {
    settings = next;
    reasoningEffort = settings.model.reasoning_effort || reasoningEffort;
    webToolsEnabled = settings.tools?.enabled ?? false;
    appearance = settings.appearance || appearance;
    await Promise.all([refreshModels(), refreshHealth()]);
    if (settings.model.default_model) selectedModel = settings.model.default_model;
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
      {controlsOpen}
      onToggleSidebar={toggleSidebar}
      onBeginTitleEdit={beginTitleEdit}
      onTitleKeydown={titleKeydown}
      onSaveTitle={saveTitle}
      onToggleControls={toggleControls}
      onCloseControls={closeControls}
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
    />
    {#if error}<div class="error">{error}</div>{/if}
    <Composer
      {pendingImages}
      {uploadingImages}
      {running}
      {activeId}
      bind:input
      bind:imageInput
      {dragActive}
      {reasoningEffort}
      {webToolsEnabled}
      onRemoveImage={removePendingImage}
      {onImageInputChange}
      {onKeydown}
      {onPaste}
      onStop={stop}
      onSend={send}
    />
  </main>
</div>

{#if dragActive}
  <div class="drop-overlay" role="region" aria-label="이미지 드롭 영역"><div><span>＋</span><strong>이미지를 여기에 놓으세요</strong><small>PNG, JPEG, WebP · 최대 6개</small></div></div>
{/if}

{#if settingsOpen && settings}
  <SettingsModal
    {settings}
    {models}
    keepImageIds={Object.values(imageDrafts).flat().map((item) => item.id)}
    onclose={() => settingsOpen = false}
    onsaved={applySavedSettings}
  />
{/if}
