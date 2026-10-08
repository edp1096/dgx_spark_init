import { createClientID } from './client-id.js';
export function createChatSessionController({
  loadMessages,
  hydrate = (messages) => messages,
  onActive = () => {},
  onMessages = () => {},
  onRuns = () => {},
  onError = () => {},
} = {}) {
  if (typeof loadMessages !== 'function') throw new Error('message loader is required');
  let activeId = '';
  let activation = 0, disposed = false;
  let currentMessages = [];
  let messageOwner = '';
  let runs = {};
  let cache = {};
  let errors = {};

  function publish(sessionId, nextMessages) {
    if (disposed) return;
    cache = { ...cache, [sessionId]: nextMessages };
    if (activeId === sessionId) {
      currentMessages = nextMessages;
      messageOwner = sessionId;
      onMessages(nextMessages, sessionId);
    }
  }

  function setError(sessionId, message) {
    if (disposed) return;
    errors = { ...errors, [sessionId]: message };
    if (activeId === sessionId) onError(message, sessionId);
  }

  async function activate(sessionId) {
    if (disposed) return [];
    const epoch = ++activation;
    if (activeId && activeId !== sessionId && messageOwner === activeId) cache = { ...cache, [activeId]: currentMessages };
    activeId = sessionId || '';
    onActive(activeId);
    if (!activeId) {
      currentMessages = [];
      messageOwner = '';
      onMessages([], '');
      onError('', '');
      return [];
    }
    let nextMessages = runs[activeId]?.messages || cache[activeId];
    if (!nextMessages) {
      const loaded = await loadMessages(sessionId);
      nextMessages = runs[sessionId]?.messages || cache[sessionId] || hydrate(loaded);
      if (!disposed && epoch === activation) cache = { ...cache, [sessionId]: nextMessages };
    }
    if (!disposed && epoch === activation && activeId === sessionId) {
      currentMessages = nextMessages;
      messageOwner = sessionId;
      onMessages(nextMessages, sessionId);
      onError(errors[sessionId] || '', sessionId);
    }
    return nextMessages;
  }

  function start(sessionId, messages, retryingIndex, extra = {}) {
    const run = { controller: new AbortController(), messages, retryingIndex, ...extra };
    runs = { ...runs, [sessionId]: run };
    onRuns(runs);
    publish(sessionId, messages);
    return run;
  }

  function finish(sessionId, run) {
    if (runs[sessionId] !== run) return false;
    const next = { ...runs };
    delete next[sessionId];
    runs = next;
    onRuns(runs);
    return true;
  }

  function remove(sessionId) {
    if (activeId === sessionId) activation++;
    const nextCache = { ...cache };
    delete nextCache[sessionId];
    cache = nextCache;
    const nextErrors = { ...errors };
    delete nextErrors[sessionId];
    errors = nextErrors;
    if (activeId === sessionId) {
      activeId = '';
      currentMessages = [];
      messageOwner = '';
      onActive('');
      onMessages([], '');
      onError('', '');
    }
  }

  async function sendInput(sessionId, content, send, saved = () => {}) {
    const run = runs[sessionId];
    if (disposed || !run || run.steeringSending) return false;
    if (!run.turnId) throw new Error('추가 입력을 받을 준비 중입니다. 잠시 후 전송하세요.');
    run.steeringSending = true; onRuns({ ...runs });
    try {
      if (!run.pendingInput || run.pendingInput.content !== content) run.pendingInput = { id:createClientID(), content };
      const entry = run.pendingInput;
      const result = await send(sessionId, run.turnId, entry.id, content);
      if (disposed || runs[sessionId] !== run) return false;
      saved(run, result);
      if (run.pendingInput === entry) run.pendingInput = null;
      return true;
    } finally {
      run.steeringSending = false;
      if (!disposed && runs[sessionId] === run) onRuns({ ...runs });
    }
  }
  function dispose() {
    if (disposed) return;
    disposed = true; activation++;
    for (const run of Object.values(runs)) run.controller.abort();
    runs = {}; cache = {}; errors = {};
  }
  return {
    sendInput, dispose,
    activate,
    publish,
    setError,
    start,
    finish,
    remove,
    abort: (sessionId = activeId) => runs[sessionId]?.controller.abort(),
    getRun: (sessionId = activeId) => runs[sessionId] || null,
    isRunning: (sessionId = activeId) => Boolean(runs[sessionId]),
    getMessages: (sessionId = activeId) => runs[sessionId]?.messages || cache[sessionId] || [],
  };
}
