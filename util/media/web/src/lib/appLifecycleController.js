import { durationFromFrames } from './videoTiming.js'

export function loadMediaPreferences(storage, pageSizeOptionsFor, defaults) {
  const views = {
    subtitle: storage.getItem('media-subtitle-view') === 'list' ? 'list' : 'gallery',
    image: storage.getItem('media-image-view') === 'list' ? 'list' : 'gallery',
    video: storage.getItem('media-video-view') === 'list' ? 'list' : 'gallery',
    speech: storage.getItem('media-speech-view') === 'list' ? 'list' : 'gallery'
  }
  const pageSizes = { ...defaults.pageSizes }
  const sortOrders = { ...defaults.sortOrders }
  for (const key of Object.keys(pageSizes)) {
    const storedSize = Number(storage.getItem(`media-${key}-page-size`))
    if (pageSizeOptionsFor(key).includes(storedSize)) pageSizes[key] = storedSize
    const storedOrder = storage.getItem(`media-${key}-sort-order`)
    if (storedOrder === 'asc' || storedOrder === 'desc') sortOrders[key] = storedOrder
  }
  return { views, pageSizes, sortOrders }
}

export function runtimeDefaults(config, currentOptions, validImageModes) {
  const checkpoint = config.image.default_checkpoint || 'official'
  const currentSampling = currentOptions.sampling_preset || 'default'
  const samplingPreset = checkpoint.startsWith('moody-') ? 'moody' : currentSampling === 'moody' ? 'default' : currentSampling
  return {
    imageForm: {
      width: config.image.default_width,
      height: config.image.default_height,
      mode: validImageModes.includes(config.image.default_mode) ? config.image.default_mode : 'create'
    },
    speechForm: { language: config.speech.default_language, speaker: config.speech.default_speaker },
    recognitionForm: {
      language: config.recognition.default_language,
      output_formats: [...config.recognition.default_output_formats],
      translation_mode: config.recognition.default_translation_mode,
      target_language: config.recognition.default_translation_language
    },
    videoForm: { width: config.video.default_width, height: config.video.default_height, fps: config.video.default_fps },
    videoDuration: durationFromFrames(config.video.default_frames, config.video.default_fps),
    enhanceEnabled: config.prompt_enhancement.default_enabled,
    options: {
      ...currentOptions,
      checkpoint,
      sampling_preset: samplingPreset,
      prompt_enhancer: Boolean(config.image.default_prompt_enhancer),
      ...(checkpoint === 'official' ? {} : { filter_mode: 'off', filter_strength: 0 })
    }
  }
}

export class AppLifecycleController {
  constructor({ api, storage, timers = globalThis, actions }) {
    this.api = api
    this.storage = storage
    this.timers = timers
    this.actions = actions
    this.timerIDs = []
  }

  start() {
    this.actions.applyPreferences(loadMediaPreferences(this.storage, this.actions.pageSizeOptionsFor, this.actions.preferenceDefaults()))
    this.api.config().then((config) => this.actions.applyConfig(config)).catch((cause) => this.actions.setError(cause.message))
    this.actions.refreshUserLoras()
    this.actions.refreshJobs()
    this.actions.refreshSystemUsage()
    this.actions.refreshVideoModels()
    this.actions.refreshImageModels()
    this.timerIDs = [
      this.timers.setInterval(this.actions.refreshJobs, 1500),
      this.timers.setInterval(this.actions.refreshSystemUsage, 5000),
      this.timers.setInterval(() => {
        if (this.actions.shouldRefreshModels()) {
          this.actions.refreshVideoModels()
          this.actions.refreshImageModels()
        }
      }, 3000),
      this.timers.setInterval(() => this.actions.setProgressClock(Date.now()), 1000)
    ]
    return () => this.stop()
  }

  stop() {
    this.timerIDs.forEach((id) => this.timers.clearInterval(id))
    this.timerIDs = []
  }
}
