import { durationFromFrames, framesForDuration } from './videoTiming.js'

function browserAudioDuration(url) {
  return new Promise((resolve) => {
    const probe = new Audio(url)
    const finish = (value) => { probe.src = ''; resolve(value) }
    probe.onloadedmetadata = () => finish(Number.isFinite(probe.duration) ? probe.duration : 0)
    probe.onerror = () => finish(0)
  })
}

export class MediaTransferController {
  constructor(actions, options = {}) {
    this.actions = actions
    this.probeAudioDuration = options.probeAudioDuration || browserAudioDuration
  }

  switchTo(tab) {
    this.actions.switchTab(tab, false)
  }

  async usePickedVideoFrame(file, target, sourceTime, sourceDuration) {
    if (target === 'start' || target === 'end') {
      this.actions.setVideoConditionImage(target, file)
    } else {
      const keyframes = this.actions.getVideoKeyframes()
      if (keyframes.length >= 8) throw new Error('키프레임은 최대 8개까지 추가할 수 있습니다.')
      const form = this.actions.getVideoForm()
      const fps = Math.max(1, Number(form.fps) || 1)
      const lastFrame = framesForDuration(this.actions.getVideoDuration(), fps) - 1
      const ratio = sourceDuration > 0 ? Math.min(1, Math.max(0, sourceTime / sourceDuration)) : 0.5
      const frame = this.actions.nearestAvailableVideoKeyframeFrame(Math.round(lastFrame * ratio))
      if (frame == null) throw new Error('현재 길이에 더 추가할 수 있는 키프레임 위치가 없습니다.')
      this.actions.setVideoKeyframes([...keyframes, {
        id: this.actions.allocateVideoKeyframeID(),
        image: this.actions.normalizeVideoImage(file),
        time: frame / fps,
        strength: 1
      }])
    }
    this.switchTo('video')
  }

  sendVideoToRecognition(job) {
    if (!job?.output_url || job.status !== 'completed') return
    this.actions.setRecognitionSourceVideoJob(job)
    this.actions.setRecognitionFile(null)
    this.actions.clearRecognitionFileInput()
    this.actions.setRecognitionForm({
      ...this.actions.getRecognitionForm(),
      source: 'video_job',
      url: ''
    })
    this.actions.resetRecognitionOptions()
    this.switchTo('recognition')
  }

  clearRecognitionSourceVideo() {
    this.actions.setRecognitionSourceVideoJob(null)
    this.actions.setRecognitionForm({ ...this.actions.getRecognitionForm(), source: 'url' })
  }

  loadVideoJobSettings(job) {
    if (!job) return
    this.actions.clearVideoConditioning()
    const config = this.actions.getConfig()
    const current = this.actions.getVideoForm()
    const form = {
      ...current,
      prompt: job.prompt || '',
      width: Number(job.params?.width) || config?.video?.default_width || 768,
      height: Number(job.params?.height) || config?.video?.default_height || 512,
      fps: Number(job.params?.fps) || config?.video?.default_fps || 24,
      seed: Number.isFinite(Number(job.params?.seed)) ? Number(job.params.seed) : -1,
      image_strength: Number(job.params?.image_strength) || 1
    }
    this.actions.setVideoForm(form)
    this.actions.setVideoDuration(durationFromFrames(job.params?.num_frames || config?.video?.default_frames || 121, form.fps))
    const saved = Array.isArray(job.params?.audio_clips) ? job.params.audio_clips : []
    const legacyID = job.params?.audio_source_job_id || ''
    const clips = (saved.length ? saved : (legacyID ? [{ source_job_id: legacyID, start: 0 }] : []))
      .map((clip) => {
        const sourceJob = this.actions.getSpeechJobs().find((item) => item.id === clip.source_job_id)
        return sourceJob ? {
          id: this.actions.allocateVideoAudioClipID(),
          job: sourceJob,
          start: Number(clip.start) || 0,
          duration: Number(clip.duration) || 0
        } : null
      })
      .filter(Boolean)
    this.actions.setVideoAudioClips(clips)
    this.actions.resetVideoEnhancement()
    this.switchTo('video')
  }

  async sendAudioToVideo(job) {
    if (!job?.output_url || job.status !== 'completed') return
    let clips = this.actions.getVideoAudioClips()
    if (clips.some((clip) => clip.job.id === job.id)) {
      this.switchTo('video')
      return
    }
    if (clips.length >= 8) {
      this.actions.setError('음성은 최대 8개까지 배치할 수 있습니다.')
      return
    }
    const lastEnd = clips.reduce((end, clip) => Math.max(end, Number(clip.start) + Number(clip.duration || 0)), 0)
    const start = Math.min(19.99, lastEnd)
    if (start >= 19.99 && clips.length) {
      this.actions.setError('20초 안에 새 음성을 배치할 공간이 없습니다.')
      return
    }
    const clipID = this.actions.allocateVideoAudioClipID()
    clips = [...clips, { id: clipID, job, start, duration: 0 }]
    this.actions.setVideoAudioClips(clips)
    try {
      const duration = Math.max(0, await this.probeAudioDuration(job.output_url))
      let cursor = 0
      clips = this.actions.getVideoAudioClips().map((clip) => clip.id === clipID ? { ...clip, duration } : clip).map((clip) => {
        const packedStart = Math.max(Number(clip.start) || 0, cursor)
        cursor = packedStart + (Number(clip.duration) || 0)
        return { ...clip, start: Math.min(19.99, Math.round(packedStart * 100) / 100) }
      })
      this.actions.setVideoAudioClips(clips)
      if (duration > 0) {
        this.actions.setVideoDuration(this.actions.snapVideoDuration(Math.min(20, Math.max(this.actions.getVideoDuration(), cursor))))
        this.actions.normalizeVideoTiming()
      }
    } catch {}
    this.switchTo('video')
  }
}
