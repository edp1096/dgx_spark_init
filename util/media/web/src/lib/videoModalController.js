import { writable } from 'svelte/store'
import { outputLabels } from './catalogs.js'
import {
  captionLanguage,
  isAudioMedia,
  mediaSummary,
  recognitionLanguageLabel,
  videoJobDuration,
} from './mediaPresentation.js'
import { formatDuration } from './videoTiming.js'

const initialState = {
  video: null,
  subtitle: null,
  subtitleRegenerateJob: null,
  regeneratingSubtitleJob: '',
  audio: null,
  recognitionVideoPickerOpen: false,
  audioPickerOpen: false,
  timelineEditorOpen: false,
  imagePickerTarget: '',
  remoteImageTarget: '',
  framePickerSource: null,
  upscaleSource: null,
  upscaleBusy: false
}

export class VideoModalController {
  constructor(actions) {
    this.actions = actions
    this.current = { ...initialState }
    this.state = writable(this.current)
    this.state.subscribe((value) => this.current = value)
  }

  setState(patch) {
    this.state.update((value) => ({ ...value, ...patch }))
  }

  showVideo(job) {
    if (!job?.output_url) return
    const captionJob = this.actions.getRecognitionJobs().find((item) => item.status === 'completed' && item.params?.source_job_id === job.id && item.caption_url)
    const details = [
      `${job.params?.width || '—'}×${job.params?.height || '—'}`,
      formatDuration(videoJobDuration(job)),
      Number(job.params?.fps) > 0 ? `${job.params.fps} fps` : '',
      job.params?.seed >= 0 ? `seed ${job.params.seed}` : ''
    ].filter(Boolean)
    this.setState({
      subtitle: null,
      video: {
        jobID: job.id,
        src: job.output_url,
        title: '생성 영상',
        detail: details.join(' · '),
        prompt: job.prompt,
        captionSrc: captionJob?.caption_url ? `${captionJob.caption_url}?v=${encodeURIComponent(captionJob.updated_at || '')}` : '',
        captionLabel: captionJob?.params?.translation_mode === 'none' ? '원문' : captionJob?.params?.target_language || '번역',
        canLoadSettings: job.params?.mode !== 'upscale',
        canSelectFrames: true,
        thumbnails: { url: `/api/jobs/${job.id}/video-preview.jpg`, number: 50, column: 10, width: 160, height: 90, scale: 1 }
      }
    })
  }

  showSubtitle(job) {
    if (!job || (!job.media_url && !job.params?.text && !job.outputs && !job.output_url)) return
    const outputs = job.outputs
      ? Object.entries(job.outputs).map(([kind, url]) => ({ label: outputLabels[kind] || kind, url }))
      : job.output_url ? [{ label: '결과 열기', url: job.output_url }] : []
    const details = [
      job.params?.detected_language || recognitionLanguageLabel(job.params?.language),
      job.params?.segments ? `${job.params.segments}구간` : '',
      job.params?.media ? mediaSummary(job) : ''
    ].filter(Boolean)
    this.setState({
      video: null,
      subtitle: {
        jobID: job.id,
        mediaSrc: job.media_url,
        audio: isAudioMedia(job),
        captionSrc: job.caption_url ? `${job.caption_url}?v=${encodeURIComponent(job.updated_at || '')}` : '',
        captionLang: captionLanguage(job),
        captionLabel: job.params?.translation_mode === 'none' ? '원문' : job.params?.target_language || '번역',
        transcript: job.params?.text,
        prompt: job.prompt,
        detail: details.join(' · '),
        canSelectFrames: Boolean(job.media_url && !isAudioMedia(job)),
        outputs
      }
    })
  }

  openSubtitleRegenerate(job) {
    if (!job || job.kind !== 'recognition' || job.status !== 'completed') return
    this.setState({ subtitle: null, subtitleRegenerateJob: job })
  }

  async regenerateSubtitle(options) {
    const job = this.current.subtitleRegenerateJob
    if (!job || this.current.regeneratingSubtitleJob) return
    this.setState({ regeneratingSubtitleJob: job.id })
    this.actions.setError('')
    try {
      await this.actions.regenerateSubtitle(job.id, options)
      this.setState({ subtitleRegenerateJob: null })
    } catch (cause) {
      this.actions.setError(cause.message)
    } finally {
      this.setState({ regeneratingSubtitleJob: '' })
    }
  }

  openFramePicker(job) {
    if (!job) return
    const recognition = job.kind === 'recognition'
    const src = recognition ? job.media_url : job.output_url
    if (!src || (recognition && isAudioMedia(job))) return
    this.setState({
      video: null,
      subtitle: null,
      framePickerSource: {
        jobID: job.id,
        src,
        title: recognition ? `받아쓰기 원본 · ${job.prompt}` : `생성 영상 · ${job.prompt}`,
        duration: recognition ? Number(job.params?.media?.duration) || 0 : videoJobDuration(job)
      }
    })
  }

  openUpscale(job) {
    if (!job) return
    const recognition = job.kind === 'recognition'
    if (recognition && (!job.media_url || isAudioMedia(job))) return
    if (!recognition && (job.status !== 'completed' || !job.output_url)) return
    const media = recognition ? job.params?.media || {} : job.params || {}
    this.setState({
      video: null,
      subtitle: null,
      upscaleSource: {
        jobID: job.id,
        title: recognition ? `받아쓰기 원본 · ${job.prompt}` : `생성 영상 · ${job.prompt}`,
        width: Number(media.width) || 0,
        height: Number(media.height) || 0,
        duration: recognition ? Number(media.duration) || 0 : videoJobDuration(job)
      }
    })
  }

  showUpscaleSource(job) {
    const sourceID = job?.params?.source_job_id
    if (!sourceID) return
    const source = this.actions.getJobs().find((item) => item.id === sourceID)
    if (!source) return
    if (source.kind === 'recognition') this.showSubtitle(source)
    else this.showVideo(source)
  }

  async submitUpscale(options) {
    const source = this.current.upscaleSource
    if (!source || this.current.upscaleBusy) return
    this.setState({ upscaleBusy: true })
    this.actions.setError('')
    try {
      await this.actions.submitUpscale(source.jobID, options)
      this.setState({ upscaleSource: null })
    } catch (cause) {
      this.actions.setError(cause.message || '영상 업스케일 작업을 추가하지 못했습니다.')
    } finally {
      this.setState({ upscaleBusy: false })
    }
  }

  showAudio(job) {
    if (!job?.output_url) return
    this.setState({
      audio: {
        src: job.output_url,
        detail: [job.params?.speaker, job.params?.language, job.params?.seed >= 0 ? `seed ${job.params.seed}` : ''].filter(Boolean).join(' · '),
        prompt: job.prompt,
        instructions: job.params?.instructions || '',
        jobID: job.id
      }
    })
  }

  sendVideoToRecognition(job) {
    if (!job?.output_url) return
    this.setState({ video: null, recognitionVideoPickerOpen: false })
    this.actions.sendVideoToRecognition(job)
  }

  loadVideoSettings(job) {
    if (!job) return
    this.setState({ video: null })
    this.actions.loadVideoSettings(job)
  }

  sendAudioToVideo(job) {
    if (!job?.output_url) return
    this.setState({ audio: null })
    return this.actions.sendAudioToVideo(job)
  }

  selectRecentImage(job, target = this.current.imagePickerTarget) {
    if (!job?.output_url || !target) return
    this.actions.setVideoConditionImage(target, {
      server: true,
      ref: `${job.id}:output:0`,
      url: job.output_url,
      name: `결과 ${job.id.slice(0, 8)}.png`,
      role: 'output'
    })
    this.setState({ imagePickerTarget: '' })
  }

  selectRemoteImage(file, target = this.current.remoteImageTarget) {
    if (file && target) this.actions.setVideoConditionImage(target, file)
  }

  conditionTitle(target, suffix = ' 선택') {
    if (target === 'start') return `시작 이미지${suffix}`
    if (target === 'end') return `마지막 이미지${suffix}`
    const id = Number(String(target).split(':')[1])
    const index = this.actions.getVideoKeyframes().findIndex((keyframe) => keyframe.id === id)
    return `키프레임 ${index >= 0 ? index + 1 : ''} 이미지${suffix}`
  }
}
