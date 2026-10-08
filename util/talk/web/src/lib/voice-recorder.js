const candidateMIMEs = [
  'audio/webm;codecs=opus',
  'audio/ogg;codecs=opus',
  'audio/mp4',
  'audio/webm',
];

export function voiceRecordingSupported(scope = globalThis) {
  return Boolean(scope?.isSecureContext && scope?.navigator?.mediaDevices?.getUserMedia && scope?.MediaRecorder);
}

export function preferredVoiceMIME(MediaRecorderClass = globalThis.MediaRecorder) {
  if (!MediaRecorderClass) return '';
  if (typeof MediaRecorderClass.isTypeSupported !== 'function') return '';
  return candidateMIMEs.find((mime) => MediaRecorderClass.isTypeSupported(mime)) || '';
}

export function voiceFilename(mimeType = '') {
  if (mimeType.includes('wav')) return 'voice.wav';
  if (mimeType.includes('ogg')) return 'voice.ogg';
  if (mimeType.includes('mp4')) return 'voice.m4a';
  return 'voice.webm';
}

export async function beginVoiceRecording(scope = globalThis) {
  if (!voiceRecordingSupported(scope)) throw new Error('이 브라우저에서는 마이크 녹음을 사용할 수 없습니다.');
  const stream = await scope.navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });
  const mimeType = preferredVoiceMIME(scope.MediaRecorder);
  let recorder;
  try {
    recorder = mimeType ? new scope.MediaRecorder(stream, { mimeType }) : new scope.MediaRecorder(stream);
  } catch (error) {
    stream.getTracks().forEach((track) => track.stop());
    throw error;
  }
  const chunks = [];
  let stopped = false;
  const completed = new Promise((resolve, reject) => {
    recorder.ondataavailable = (event) => { if (event.data?.size) chunks.push(event.data); };
    recorder.onerror = (event) => {
      stream.getTracks().forEach((track) => track.stop());
      reject(event.error || new Error('마이크 녹음에 실패했습니다.'));
    };
    recorder.onstop = () => {
      stream.getTracks().forEach((track) => track.stop());
      const type = recorder.mimeType || mimeType || chunks[0]?.type || 'audio/webm';
      resolve(new scope.Blob(chunks, { type }));
    };
  });
  recorder.start(500);
  return {
    stop() {
      if (!stopped && recorder.state !== 'inactive') {
        stopped = true;
        recorder.stop();
      }
      return completed;
    },
  };
}
