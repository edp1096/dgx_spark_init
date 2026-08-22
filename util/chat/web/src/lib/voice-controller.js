import { beginContinuousVoice, isIgnorableVoiceTranscript } from './continuous-voice.js';
import { beginVoiceRecording, voiceFilename, voiceRecordingSupported } from './voice-recorder.js';

export function formatVoiceError(error) {
  if (error?.name === 'NotAllowedError' || error?.name === 'SecurityError') {
    return '마이크 권한이 거부되었습니다. 주소창의 사이트 권한에서 마이크를 허용하세요.';
  }
  if (error?.name === 'NotFoundError') return '사용 가능한 마이크를 찾지 못했습니다.';
  return error?.message || '마이크 녹음에 실패했습니다.';
}

export function createVoiceController({
  environment = () => globalThis,
  transcribe,
  getActiveSessionId = () => '',
  isSessionRunning = () => false,
  isUploading = () => false,
  filterFillers = () => true,
  hasInput = () => false,
  onTranscript = () => {},
  onAutoSend = () => {},
  onFocus = () => {},
  onBeforeContinuous = () => {},
  onUtteranceStart = () => {},
  onError = () => {},
  onState = () => {},
  beginRecording = beginVoiceRecording,
  beginContinuous = beginContinuousVoice,
  filenameForBlob = voiceFilename,
  shouldIgnore = isIgnorableVoiceTranscript,
} = {}) {
  if (typeof transcribe !== 'function') throw new Error('voice transcription function is required');
  let manualState = 'idle';
  let seconds = 0;
  let recording = null;
  let recordingSessionId = '';
  let timer = null;
  let continuousEnabled = false;
  let continuousState = 'off';
  let listener = null;
  let utterances = [];
  let queueProcessing = false;
  let pendingTranscripts = {};
  let autoSendPending = {};

  function publish() {
    onState({
      manualState,
      seconds,
      continuousEnabled,
      continuousState,
      queueCount: utterances.length + (queueProcessing ? 1 : 0),
    });
  }

  function supported() { return voiceRecordingSupported(environment()); }

  function deliverTranscript(sessionId, value) {
    const text = String(value || '').trim();
    if (!text) return;
    if (getActiveSessionId() === sessionId) {
      onTranscript(sessionId, text);
      return;
    }
    pendingTranscripts = {
      ...pendingTranscripts,
      [sessionId]: [pendingTranscripts[sessionId], text].filter(Boolean).join(' '),
    };
  }

  function consumePendingTranscript(sessionId) {
    const text = pendingTranscripts[sessionId] || '';
    if (text) {
      const next = { ...pendingTranscripts };
      delete next[sessionId];
      pendingTranscripts = next;
    }
    return text;
  }

  async function startManual() {
    const sessionId = getActiveSessionId();
    if (!sessionId || isSessionRunning(sessionId) || isUploading() || manualState !== 'idle' || continuousEnabled || ['requesting', 'stopping'].includes(continuousState)) return;
    if (!supported()) {
      onError(sessionId, '마이크는 HTTPS, localhost 또는 안전한 출처로 허용한 주소에서 사용할 수 있습니다.');
      return;
    }
    manualState = 'requesting';
    recordingSessionId = sessionId;
    onError(sessionId, '');
    publish();
    try {
      recording = await beginRecording(environment());
      seconds = 0;
      manualState = 'recording';
      clearInterval(timer);
      timer = setInterval(() => {
        seconds += 1;
        publish();
        if (seconds >= 300) stopManual();
      }, 1000);
    } catch (error) {
      onError(recordingSessionId, formatVoiceError(error));
      recording = null;
      manualState = 'idle';
      recordingSessionId = '';
    }
    publish();
  }

  async function stopManual() {
    if (manualState !== 'recording' || !recording) return;
    const activeRecording = recording;
    const sessionId = recordingSessionId;
    recording = null;
    manualState = 'transcribing';
    clearInterval(timer);
    publish();
    try {
      const blob = await activeRecording.stop();
      if (!blob.size) throw new Error('녹음된 음성이 없습니다.');
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 5 * 60 * 1000);
      try {
        const result = await transcribe(blob, filenameForBlob(blob.type), controller.signal);
        deliverTranscript(sessionId, result.text || '');
        onError(sessionId, '');
      } finally {
        clearTimeout(timeout);
      }
    } catch (error) {
      onError(sessionId, error?.name === 'AbortError' ? '음성 인식 시간이 초과되었습니다.' : formatVoiceError(error));
    } finally {
      manualState = 'idle';
      seconds = 0;
      recordingSessionId = '';
      publish();
      await flushAutoSend(sessionId);
      await onFocus(sessionId);
    }
  }

  async function toggleContinuous() {
    if (continuousEnabled || continuousState === 'error') {
      await stopContinuous();
      return;
    }
    const sessionId = getActiveSessionId();
    if (!sessionId || manualState !== 'idle' || ['requesting', 'stopping'].includes(continuousState)) return;
    if (!supported()) {
      onError(sessionId, '마이크는 HTTPS, localhost 또는 안전한 출처로 허용한 주소에서 사용할 수 있습니다.');
      return;
    }
    onBeforeContinuous();
    continuousEnabled = true;
    continuousState = 'requesting';
    publish();
    try {
      listener = await beginContinuous(environment(), {
        getContext: () => ({ sessionId: getActiveSessionId() }),
        onState: (state) => { continuousState = state; publish(); },
        onUtterance: (blob, context) => {
          onUtteranceStart();
          enqueueUtterance(blob, context?.sessionId || getActiveSessionId());
        },
        onError: (error) => {
          onError(getActiveSessionId(), formatVoiceError(error));
          continuousEnabled = false;
          continuousState = 'error';
          listener = null;
          publish();
        },
      });
      onError(sessionId, '');
    } catch (error) {
      onError(sessionId, formatVoiceError(error));
      continuousEnabled = false;
      listener = null;
      continuousState = 'off';
    }
    publish();
  }

  async function stopContinuous() {
    const activeListener = listener;
    continuousEnabled = false;
    listener = null;
    continuousState = 'stopping';
    publish();
    try { await activeListener?.stop(); }
    catch (error) { onError(getActiveSessionId(), formatVoiceError(error)); }
    finally { continuousState = 'off'; publish(); }
  }

  function enqueueUtterance(blob, sessionId) {
    if (!blob?.size || !sessionId) return;
    utterances = [...utterances, { blob, sessionId }];
    publish();
    processUtterances();
  }

  async function processUtterances() {
    if (queueProcessing) return;
    queueProcessing = true;
    publish();
    try {
      while (utterances.length) {
        const item = utterances[0];
        utterances = utterances.slice(1);
        publish();
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 5 * 60 * 1000);
        try {
          const result = await transcribe(item.blob, filenameForBlob(item.blob.type), controller.signal);
          const transcript = result.text || '';
          if (filterFillers() && shouldIgnore(transcript)) {
            onError(item.sessionId, '');
            continue;
          }
          deliverTranscript(item.sessionId, transcript);
          autoSendPending = { ...autoSendPending, [item.sessionId]: true };
          onError(item.sessionId, '');
          flushAutoSend(item.sessionId);
        } catch (error) {
          const message = error?.name === 'AbortError' ? '음성 인식 시간이 초과되었습니다.' : formatVoiceError(error);
          if (!/empty text|no audio|녹음된 음성이 없습니다/i.test(message)) onError(item.sessionId, message);
        } finally {
          clearTimeout(timeout);
        }
      }
    } finally {
      queueProcessing = false;
      publish();
    }
  }

  async function flushAutoSend(sessionId) {
    if (!autoSendPending[sessionId] || getActiveSessionId() !== sessionId || isSessionRunning(sessionId) || manualState !== 'idle') return;
    await Promise.resolve();
    if (!autoSendPending[sessionId] || getActiveSessionId() !== sessionId || isSessionRunning(sessionId) || !hasInput(sessionId)) return;
    autoSendPending = { ...autoSendPending, [sessionId]: false };
    await onAutoSend(sessionId);
  }

  function clearAutoSend(sessionId) {
    autoSendPending = { ...autoSendPending, [sessionId]: false };
  }

  function setPlaybackPaused(paused) {
    if (continuousEnabled && listener) listener.setPaused(paused);
  }

  async function dispose() {
    clearInterval(timer);
    if (recording) await recording.stop().catch(() => {});
    recording = null;
    await stopContinuous();
  }

  publish();
  return {
    supported,
    startManual,
    stopManual,
    toggleContinuous,
    stopContinuous,
    consumePendingTranscript,
    flushAutoSend,
    clearAutoSend,
    hasAutoSendPending: (sessionId) => Boolean(autoSendPending[sessionId]),
    setPlaybackPaused,
    dispose,
  };
}
