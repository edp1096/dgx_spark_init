import { framesForDuration } from './videoTiming.js'
import { nearestAvailableVideoKeyframeFrame, normalizeVideoTiming, videoKeyframeCapacity } from './videoWorkflow.js'

export class VideoTimelineController {
  constructor(actions, urls = globalThis.URL) {
    this.actions = actions
    this.urls = urls
  }

  normalizeImage(image) {
    if (!image) return null
    if (image.server || image.file) return image
    return { file: image, name: image.name, size: image.size, lastModified: image.lastModified, preview: this.urls.createObjectURL(image), server: false }
  }

  releaseImage(image) {
    if (image?.preview?.startsWith('blob:')) this.urls.revokeObjectURL(image.preview)
  }

  imagePreview(image) { return image?.preview || image?.url || '' }

  setConditionImage(target, image) {
    const normalized = this.normalizeImage(image)
    this.actions.setPromptMessage('')
    if (target === 'start') {
      this.releaseImage(this.actions.getStartImage())
      this.actions.setStartImage(normalized)
      this.actions.resetEnhancement()
      return normalized
    }
    if (target === 'end') {
      this.releaseImage(this.actions.getEndImage())
      this.actions.setEndImage(normalized)
      return normalized
    }
    if (!target.startsWith('keyframe:')) return null
    const id = Number(target.split(':')[1])
    this.actions.setKeyframes(this.actions.getKeyframes().map((keyframe) => {
      if (keyframe.id !== id) return keyframe
      this.releaseImage(keyframe.image)
      return { ...keyframe, image: normalized }
    }))
    return normalized
  }

  capacity() {
    return videoKeyframeCapacity(this.actions.getDuration(), this.actions.getFPS())
  }

  nearestFrame(rawFrame, excludeID = null) {
    return nearestAvailableVideoKeyframeFrame({
      rawFrame,
      excludeID,
      seconds: this.actions.getDuration(),
      fps: this.actions.getFPS(),
      keyframes: this.actions.getKeyframes()
    })
  }

  addKeyframe() {
    const keyframes = this.actions.getKeyframes()
    if (keyframes.length >= this.capacity()) return null
    this.actions.setPromptMessage('')
    const count = keyframes.length + 1
    const fps = Math.max(1, Number(this.actions.getFPS()) || 1)
    const lastFrame = framesForDuration(this.actions.getDuration(), fps) - 1
    const frame = this.nearestFrame(Math.round(lastFrame * count / (count + 1)))
    if (frame == null) return null
    const keyframe = { id: this.actions.allocateKeyframeID(), image: null, time: frame / fps, strength: 1 }
    this.actions.setKeyframes([...keyframes, keyframe])
    return keyframe
  }

  normalizeTiming() {
    const normalized = normalizeVideoTiming({
      seconds: this.actions.getDuration(),
      fps: this.actions.getFPS(),
      keyframes: this.actions.getKeyframes(),
      audioClips: this.actions.getAudioClips()
    })
    this.actions.setDuration(normalized.duration)
    this.actions.setKeyframes(normalized.keyframes)
    this.actions.setAudioClips(normalized.audioClips)
    return normalized
  }

  removeKeyframe(id) {
    this.actions.setPromptMessage('')
    const keyframes = this.actions.getKeyframes()
    this.releaseImage(keyframes.find((keyframe) => keyframe.id === id)?.image)
    this.actions.setKeyframes(keyframes.filter((keyframe) => keyframe.id !== id))
  }

  updateKeyframe(id, field, value) {
    this.actions.setPromptMessage('')
    this.actions.setKeyframes(this.actions.getKeyframes().map((keyframe) => keyframe.id === id ? { ...keyframe, [field]: Number(value) } : keyframe))
  }

  moveKeyframe(id, rawTime) {
    const fps = Math.max(1, Number(this.actions.getFPS()) || 1)
    const frame = this.nearestFrame(Math.round(Number(rawTime) * fps), id)
    if (frame != null) this.updateKeyframe(id, 'time', frame / fps)
  }

  clearConditioning() {
    this.actions.setPromptMessage('')
    this.releaseImage(this.actions.getStartImage())
    this.releaseImage(this.actions.getEndImage())
    this.actions.getKeyframes().forEach((keyframe) => this.releaseImage(keyframe.image))
    this.actions.setStartImage(null)
    this.actions.setEndImage(null)
    this.actions.setEndStrength(1)
    this.actions.setKeyframes([])
  }

  removeAudio(id) {
    this.actions.setAudioClips(this.actions.getAudioClips().filter((clip) => clip.id !== id))
  }

  clearAudio() { this.actions.setAudioClips([]) }

  moveAudio(id, rawStart) {
    const clips = this.actions.getAudioClips()
    const clip = clips.find((item) => item.id === id)
    if (!clip) return
    const duration = Math.max(0, Number(clip.duration) || 0)
    const endLimit = Math.max(0, this.actions.getDuration() - Math.min(duration, this.actions.getDuration()))
    const desired = Math.min(endLimit, Math.max(0, Number(rawStart) || 0))
    const others = clips.filter((item) => item.id !== id).sort((a, b) => a.start - b.start)
    let start = desired
    for (const other of others) {
      const otherEnd = Number(other.start) + Number(other.duration || 0)
      if (start < otherEnd && start + duration > Number(other.start)) {
        start = desired >= Number(other.start) ? Math.min(endLimit, otherEnd) : Math.max(0, Number(other.start) - duration)
      }
    }
    this.actions.setAudioClips(clips.map((item) => item.id === id ? { ...item, start: Math.round(start * 100) / 100 } : item))
  }

  toggleAudio(job) {
    const current = this.actions.getAudioClips().find((clip) => clip.job.id === job.id)
    if (current) this.removeAudio(current.id)
    else this.actions.sendAudioToVideo(job)
  }

  destroy() { this.clearConditioning() }
}
