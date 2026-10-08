import { Stretch } from '@soundtouchjs/core';

// Feed one continuous WSOLA pipeline across network and sentence boundaries.
// AudioBufferSource.playbackRate would also raise/lower the speaker's pitch.
export class PCMTempo {
  constructor({ sampleRate = 24000, speakRate = 1 } = {}) {
    this.rate = Number.isFinite(speakRate) && speakRate >= 0.5 && speakRate <= 2 ? speakRate : 1;
    this.inputFrames = 0;
    this.outputFrames = 0;
    this.stretch = this.rate === 1 ? null : new Stretch({
      sampleRate,
      createBuffers: true,
    });
    if (this.stretch) this.stretch.tempo = this.rate;
  }

  append(samples) {
    this.inputFrames += samples.length;
    if (!this.stretch) return samples;
    this.feed(samples);
    return this.drain(Math.floor(this.inputFrames / this.rate) - this.outputFrames);
  }

  feed(samples) {
    // SoundTouch accepts interleaved stereo; the TTS stream is mono.
    const stereo = new Float32Array(samples.length * 2);
    for (let i = 0; i < samples.length; i += 1) {
      stereo[2 * i] = stereo[2 * i + 1] = samples[i];
    }
    this.stretch.inputBuffer.putSamples(stereo);
    this.stretch.process();
  }

  drain(limit) {
    const frames = Math.min(this.stretch.outputBuffer.frameCount, Math.max(0, limit));
    const stereo = new Float32Array(frames * 2);
    this.stretch.outputBuffer.extract(stereo, 0, frames);
    this.stretch.outputBuffer.receive(frames);
    const mono = new Float32Array(frames);
    for (let i = 0; i < frames; i += 1) mono[i] = stereo[2 * i];
    this.outputFrames += frames;
    return mono;
  }

  finish() {
    if (!this.stretch) return new Float32Array(0);
    const remaining = Math.max(0, Math.round(this.inputFrames / this.rate) - this.outputFrames);
    // Flush WSOLA's lookahead and overlap history, retaining only the real
    // stream's target duration. Padding is never counted as input speech.
    const padding = new Float32Array(this.stretch.inputChunkSize);
    while (this.stretch.outputBuffer.frameCount < remaining) this.feed(padding);
    const tail = this.drain(remaining);
    this.stretch = null;
    return tail;
  }
}
