import {
  attachmentKind, isSupportedAttachmentFile, maxAttachmentBytes, maxImageBytes, maxMessageBytes,
} from './attachments.js';

const maxAttachments = 6;

export function createAttachmentController({ uploadFile, uploadURL, onState = () => {}, onError = () => {}, onQueueIdle = () => {} } = {}) {
  if (typeof uploadFile !== 'function' || typeof uploadURL !== 'function') throw new Error('attachment upload functions are required');
  let activeSessionId = '';
  let drafts = {};
  let queue = [];
  let processing = false;
  let uploadingSessionId = '';
  let sourceSessionId = '';

  function snapshot() {
    return {
      drafts: { ...drafts },
      pending: drafts[activeSessionId] || [],
      uploading: Boolean(activeSessionId) && (uploadingSessionId === activeSessionId || queue.some((item) => item.sessionId === activeSessionId)),
      sourceDownloading: Boolean(activeSessionId) && sourceSessionId === activeSessionId,
    };
  }

  function publish() { onState(snapshot()); }

  function setPending(sessionId, items) {
    drafts = { ...drafts, [sessionId]: items };
    publish();
  }

  function select(sessionId) {
    activeSessionId = sessionId || '';
    publish();
  }

  function discard(sessionId) {
    const next = { ...drafts };
    delete next[sessionId];
    drafts = next;
    queue = queue.filter((item) => item.sessionId !== sessionId);
    if (sourceSessionId === sessionId) sourceSessionId = '';
    publish();
  }

  function remove(id, { sessionId = activeSessionId, blocked = false } = {}) {
    if (!sessionId || blocked) return;
    setPending(sessionId, (drafts[sessionId] || []).filter((item) => item.id !== id));
  }

  function addFiles(files, { sessionId = activeSessionId, blocked = false } = {}) {
    const dropped = Array.from(files || []);
    const mediaFiles = dropped.filter(isSupportedAttachmentFile);
    if (!mediaFiles.length || blocked || !sessionId) {
      if (dropped.length && !blocked) onError(sessionId, '지원되는 이미지·음성·비디오 파일만 첨부할 수 있습니다.');
      return false;
    }
    const oversized = mediaFiles.find((file) => file.size > (attachmentKind(file) === 'image' ? maxImageBytes : maxAttachmentBytes));
    if (oversized) {
      onError(sessionId, `${oversized.name}: ${attachmentKind(oversized) === 'image' ? '이미지는 15MB' : '음성·비디오는 64MB'} 이하여야 합니다.`);
      return false;
    }
    const draft = drafts[sessionId] || [];
    const queued = queue.filter((item) => item.sessionId === sessionId);
    if (draft.length + queued.length + mediaFiles.length > maxAttachments) {
      onError(sessionId, `미디어는 한 메시지에 최대 ${maxAttachments}개까지 첨부할 수 있습니다.`);
      return false;
    }
    const totalBytes = draft.reduce((sum, item) => sum + (item.size || 0), 0)
      + queued.reduce((sum, item) => sum + (item.file.size || 0), 0)
      + mediaFiles.reduce((sum, file) => sum + (file.size || 0), 0);
    if (totalBytes > maxMessageBytes) {
      onError(sessionId, '한 메시지의 첨부 파일 합계는 96MB 이하여야 합니다.');
      return false;
    }
    drafts = { ...drafts, [sessionId]: draft };
    queue = [...queue, ...mediaFiles.map((file) => ({ file, sessionId }))];
    publish();
    processQueue();
    return true;
  }

  async function processQueue() {
    if (processing) return;
    processing = true;
    try {
      while (queue.length) {
        const item = queue[0];
        uploadingSessionId = item.sessionId;
        publish();
        onError(item.sessionId, '');
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 120000);
        try {
          const attachment = await uploadFile(item.file, controller.signal);
          if (Object.prototype.hasOwnProperty.call(drafts, item.sessionId)) {
            setPending(item.sessionId, [...(drafts[item.sessionId] || []), attachment]);
          }
        } catch (error) {
          onError(item.sessionId, error.name === 'AbortError' ? '미디어 업로드 시간이 초과되었습니다.' : error.message);
        } finally {
          clearTimeout(timeout);
          queue = queue.filter((queued) => queued !== item);
          uploadingSessionId = '';
          publish();
        }
      }
    } finally {
      processing = false;
      uploadingSessionId = '';
      publish();
      onQueueIdle();
    }
  }

  async function addURL(rawURL, { sessionId = activeSessionId, blocked = false } = {}) {
    if (!sessionId || blocked || snapshot().uploading) return false;
    try {
      const parsed = new URL(rawURL);
      if (!['http:', 'https:'].includes(parsed.protocol)) throw new Error('HTTP 또는 HTTPS 주소를 입력하세요.');
    } catch (error) {
      onError(sessionId, error.message === 'HTTP 또는 HTTPS 주소를 입력하세요.' ? error.message : '올바른 영상 주소를 입력하세요.');
      return false;
    }
    const draft = drafts[sessionId] || [];
    if (draft.length >= maxAttachments) {
      onError(sessionId, `미디어는 한 메시지에 최대 ${maxAttachments}개까지 첨부할 수 있습니다.`);
      return false;
    }
    uploadingSessionId = sessionId;
    sourceSessionId = sessionId;
    publish();
    onError(sessionId, '');
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 30 * 60 * 1000);
    try {
      const attachment = await uploadURL(rawURL, controller.signal);
      const current = drafts[sessionId] || [];
      const totalBytes = current.reduce((sum, item) => sum + (item.size || 0), 0) + (attachment.size || 0);
      if (current.length >= maxAttachments || totalBytes > maxMessageBytes) {
        onError(sessionId, '한 메시지의 첨부 파일 합계는 96MB 이하여야 합니다.');
        return false;
      }
      setPending(sessionId, [...current, attachment]);
      return true;
    } catch (error) {
      onError(sessionId, error.name === 'AbortError' ? 'URL 미디어 취득 시간이 초과되었습니다.' : error.message);
      return false;
    } finally {
      clearTimeout(timeout);
      uploadingSessionId = '';
      sourceSessionId = '';
      publish();
    }
  }

  return { select, discard, remove, addFiles, addURL, setPending, snapshot };
}
