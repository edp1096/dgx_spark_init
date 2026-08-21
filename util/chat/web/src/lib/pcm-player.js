function joinBytes(left, right) {
  if (!left.length) return right;
  const joined = new Uint8Array(left.length + right.length);
  joined.set(left);
  joined.set(right, left.length);
  return joined;
}

export function resolveSpeechSeed(configuredSeed, cryptoProvider = globalThis.crypto) {
  const configured = Number(configuredSeed);
  if (Number.isInteger(configured) && configured >= 0 && configured <= 2147483647) return configured;
  if (!cryptoProvider?.getRandomValues) throw new Error('음성 seed를 생성할 수 없습니다.');
  const random = new Uint32Array(1);
  cryptoProvider.getRandomValues(random);
  return random[0] & 0x7fffffff;
}

export function pcm16LEToFloat32(bytes) {
  const sampleCount = Math.floor(bytes.byteLength / 2);
  const samples = new Float32Array(sampleCount);
  const view = new DataView(bytes.buffer, bytes.byteOffset, sampleCount * 2);
  for (let index = 0; index < sampleCount; index += 1) {
    const value = view.getInt16(index * 2, true);
    samples[index] = value < 0 ? value / 32768 : value / 32767;
  }
  return samples;
}

export class PCMStreamPlayer {
  constructor({ sampleRate = 24000, AudioContextClass, onStart } = {}) {
    const Context = AudioContextClass || globalThis.AudioContext || globalThis.webkitAudioContext;
    if (!Context) throw new Error('이 브라우저는 스트리밍 음성 재생을 지원하지 않습니다.');
    this.context = new Context({ sampleRate });
    this.sampleRate = sampleRate;
    this.onStart = onStart;
    this.pending = new Uint8Array(0);
    this.sources = new Set();
    this.nextTime = 0;
    this.started = false;
    this.stopped = false;
    this.finishing = false;
    this.finishResolve = null;
    this.blockBytes = Math.floor(sampleRate / 10) * 2;
  }

  async append(bytes) {
    if (this.stopped || !bytes?.length) return;
    if (this.context.state === 'suspended') await this.context.resume();
    this.pending = joinBytes(this.pending, bytes);
    const ready = this.pending.length - (this.pending.length % this.blockBytes);
    if (ready <= 0) return;
    const playable = this.pending.slice(0, ready);
    this.pending = this.pending.slice(ready);
    for (let offset = 0; offset < playable.length; offset += this.blockBytes) {
      this.schedule(playable.subarray(offset, offset + this.blockBytes));
    }
  }

  schedule(bytes) {
    if (this.stopped || bytes.length < 2) return;
    const samples = pcm16LEToFloat32(bytes);
    const buffer = this.context.createBuffer(1, samples.length, this.sampleRate);
    buffer.copyToChannel(samples, 0);
    const source = this.context.createBufferSource();
    source.buffer = buffer;
    source.connect(this.context.destination);
    source.onended = () => {
      this.sources.delete(source);
      if (this.finishing && this.sources.size === 0) {
        this.finishResolve?.();
        this.finishResolve = null;
      }
    };
    const startAt = Math.max(this.nextTime, this.context.currentTime + 0.04);
    source.start(startAt);
    this.nextTime = startAt + buffer.duration;
    this.sources.add(source);
    if (!this.started) {
      this.started = true;
      this.onStart?.();
    }
  }

  async finish() {
    if (this.stopped) return;
    if (this.pending.length >= 2) {
      const evenLength = this.pending.length - (this.pending.length % 2);
      this.schedule(this.pending.subarray(0, evenLength));
    }
    this.pending = new Uint8Array(0);
    this.finishing = true;
    if (this.sources.size > 0) {
      await new Promise((resolve) => { this.finishResolve = resolve; });
    }
    if (!this.stopped) await this.context.close();
  }

  stop() {
    if (this.stopped) return;
    this.stopped = true;
    this.pending = new Uint8Array(0);
    for (const source of this.sources) {
      try { source.stop(); } catch { /* already stopped */ }
    }
    this.sources.clear();
    this.finishResolve?.();
    this.finishResolve = null;
    this.context.close().catch(() => {});
  }
}
