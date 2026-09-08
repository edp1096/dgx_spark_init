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
  let currentMessages = [];
  let runs = {};
  let cache = {};
  let errors = {};

  function publish(sessionId, nextMessages) {
    cache = { ...cache, [sessionId]: nextMessages };
    if (activeId === sessionId) {
      currentMessages = nextMessages;
      onMessages(nextMessages, sessionId);
    }
  }

  function setError(sessionId, message) {
    errors = { ...errors, [sessionId]: message };
    if (activeId === sessionId) onError(message, sessionId);
  }

  async function activate(sessionId) {
    if (activeId && activeId !== sessionId) cache = { ...cache, [activeId]: currentMessages };
    activeId = sessionId || '';
    onActive(activeId);
    if (!activeId) {
      currentMessages = [];
      onMessages([], '');
      onError('', '');
      return [];
    }
    let nextMessages = runs[activeId]?.messages || cache[activeId];
    if (!nextMessages) {
      nextMessages = hydrate(await loadMessages(activeId));
      cache = { ...cache, [activeId]: nextMessages };
    }
    if (activeId === sessionId) {
      currentMessages = nextMessages;
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
    const nextCache = { ...cache };
    delete nextCache[sessionId];
    cache = nextCache;
    const nextErrors = { ...errors };
    delete nextErrors[sessionId];
    errors = nextErrors;
    if (activeId === sessionId) {
      activeId = '';
      currentMessages = [];
      onActive('');
      onMessages([], '');
      onError('', '');
    }
  }

  return {
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
