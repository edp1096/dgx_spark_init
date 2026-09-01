import { PCMStreamPlayer } from './pcm-player.js';

export function createSpeechController({
  stream,
  PlayerClass = PCMStreamPlayer,
  onStatus = () => {},
  onPlaybackChange = () => {},
  onError = () => {},
} = {}) {
  if (typeof stream !== 'function') throw new Error('speech stream function is required');
  let active = null;
  let loadingKey = '';
  let playingKey = '';

  function publishStatus() {
    onStatus({ loadingKey, playingKey });
  }

  function release(session) {
    if (active !== session) return;
    active = null;
    loadingKey = '';
    playingKey = '';
    publishStatus();
    if (session.playbackStarted) onPlaybackChange(false);
    session.resolveDone();
  }

  function stop() {
    const session = active;
    if (!session) return;
    session.stopped = true;
    session.controller.abort();
    session.player?.stop();
    release(session);
  }

  function create(key, sessionId) {
    const session = {
      key,
      sessionId,
      controller: new AbortController(),
      player: null,
      queue: [],
      processing: false,
      closed: false,
      stopped: false,
      playbackStarted: false,
      resolveDone: () => {},
      done: null,
    };
    session.done = new Promise((resolve) => { session.resolveDone = resolve; });
    active = session;
    return session;
  }

  function enqueue(session, text) {
    const value = String(text || '').trim();
    if (!value || session.stopped || session.closed || active !== session) return;
    session.queue.push(value);
    if (!session.player) {
      loadingKey = session.key;
      publishStatus();
    }
    process(session);
  }

  function close(session) {
    if (session.stopped || active !== session) return session.done;
    session.closed = true;
    process(session);
    return session.done;
  }

  async function process(session) {
    if (session.processing || session.stopped || active !== session) return;
    session.processing = true;
    try {
      while (session.queue.length && !session.stopped) {
        const text = session.queue.shift();
        await stream(text, session.controller.signal, async (bytes, sampleRate) => {
          if (session.stopped || active !== session) return;
          if (!session.player) {
            session.player = new PlayerClass({
              sampleRate,
              onStart: () => {
                if (active !== session) return;
                loadingKey = '';
                playingKey = session.key;
                publishStatus();
                session.playbackStarted = true;
                onPlaybackChange(true);
              },
            });
          }
          await session.player.append(bytes);
        });
      }
      if (session.closed && !session.stopped) {
        if (session.player) await session.player.finish();
        release(session);
      }
    } catch (error) {
      if (!session.stopped && error?.name !== 'AbortError') {
        onError(session.sessionId, error?.message || '답변 음성 생성에 실패했습니다.');
      }
      session.player?.stop();
      release(session);
    } finally {
      session.processing = false;
    }
  }

  return {
    create,
    enqueue,
    close,
    stop,
    isCurrent: (session) => active === session,
    status: () => ({ loadingKey, playingKey }),
  };
}
