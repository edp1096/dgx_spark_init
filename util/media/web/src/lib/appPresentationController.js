import { buildAssistantState } from './assistantState.js'
import {
  imageGenerationProgress,
  recognitionProgressPercent,
  recognitionProgressText,
  recognitionProgressTiming,
  speechGenerationProgress,
  videoGenerationProgress
} from './jobProgress.js'

export class AppPresentationController {
  constructor({ engineCatalog, actions }) {
    this.engineCatalog = engineCatalog
    this.actions = actions
  }

  state() { return this.actions.getState() }
  activeJobs() { return this.state().jobs.filter((job) => job.status === 'queued' || job.status === 'running') }
  imageProgress(job) { const state = this.state(); return imageGenerationProgress(job, state.jobs, state.progressClock) }
  videoProgress(job) { const state = this.state(); return videoGenerationProgress(job, state.jobs, state.progressClock) }
  speechProgress(job) { const state = this.state(); return speechGenerationProgress(job, state.jobs, state.progressClock) }
  recognitionText(job) { return recognitionProgressText(job, this.state().jobs) }
  recognitionTiming(job) { return recognitionProgressTiming(job, this.state().progressClock) }
  recognitionPercent(job) { return recognitionProgressPercent(job) }

  enginePresentation() {
    const state = this.state()
    const statuses = this.engineCatalog.map(([key, label]) => ({ key, label, online: state.engineStates[key] === 'online' }))
    const online = statuses.filter((item) => item.online).length
    const aggregate = online === statuses.length ? 'healthy' : online === 0 ? 'down' : 'degraded'
    return { statuses, aggregate, label: aggregate === 'healthy' ? '전체 정상' : aggregate === 'degraded' ? '일부 장애' : '전체 장애' }
  }

  assistantState() {
    const state = this.state()
    return buildAssistantState({
      tab: state.tab, busy: state.busy,
      imageForm: state.imageForm, imageEnhanceEnabled: state.imageEnhanceEnabled, activeKreaModuleLabels: state.activeKreaModuleLabels,
      videoForm: state.videoForm, videoDurationSeconds: state.videoDurationSeconds, videoEnhanceEnabled: state.videoEnhanceEnabled,
      videoImage: state.videoImage, videoEndImage: state.videoEndImage, videoAudioJob: state.videoAudioJob,
      videoAudioClips: state.videoAudioClips, videoKeyframes: state.videoKeyframes,
      speechForm: state.speechForm, recognitionForm: state.recognitionForm, recognitionFile: state.recognitionFile,
      recognitionSourceVideoJob: state.recognitionSourceVideoJob, pagedImageJobs: state.pagedImageJobs,
      imagePage: state.listPages.image, imagePageSize: state.listPageSizes.image
    })
  }
}
