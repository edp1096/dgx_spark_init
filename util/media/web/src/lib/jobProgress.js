import { compactElapsed, formatBytes } from './mediaPresentation.js'

import { a2vBaselineEstimateSeconds, imageGenerationDistance, imageGenerationKey, imageGenerationWork, imageJobDurationSeconds, percentile, speechGenerationKey, speechGenerationWork, videoAccelerationProfile, videoGenerationDistance, videoGenerationDurationSeconds, videoGenerationKey, videoGenerationPipeline, videoGenerationWork } from './generationMetrics.js'

import { videoUpscaleEstimateSeconds } from '../videoUpscaleEta.js'
import { runtimePhasePresentation } from './runtimePhasePresentation.js'


export function modelPreparationProgress(job, jobs, progressClock) {
    const params = job.params || {}
    if (job.status !== 'running' || params.stage !== 'model-preparing') return null
    const started = Date.parse(params.model_prepare_started_at || params.stage_started_at || params.started_at || job.updated_at || job.created_at || 0)
    const elapsedSeconds = Number.isFinite(started) ? Math.max(0, (progressClock - started) / 1000) : 0
    const profile = params.model_prepare_profile || params.model_plan?.profile || ''
    const observed = percentile((jobs || [])
      .filter((item) => item.id !== job.id && item.params?.model_prepare_profile === profile && Number(item.params?.model_prepare_seconds) > 0)
      .sort((left, right) => Date.parse(right.updated_at || 0) - Date.parse(left.updated_at || 0))
      .slice(0, 12)
      .map((item) => Number(item.params.model_prepare_seconds)), .6)
    const estimateSeconds = Math.max(1, observed || Number(params.model_prepare_estimate_seconds) || 30)
    const percent = Math.min(94, Math.max(3, elapsedSeconds / estimateSeconds * 100))
    const remainingSeconds = Math.max(0, estimateSeconds - elapsedSeconds)
    const label = params.model_prepare_label || params.model_plan?.label || '필요 모델 탑재'
    const runtime = runtimePhasePresentation(job)
    const finishTime = new Date(progressClock + remainingSeconds * 1000).toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit', second: '2-digit', hourCycle: 'h23' })
    return {
      label: runtime?.label || `${label} · ${Math.round(percent)}%`,
      percent: runtime?.progress > 0 ? Math.min(99, runtime.progress) : percent,
      elapsed: `${Math.round(elapsedSeconds)}/${Math.round(estimateSeconds)}초`,
      eta: runtime?.detail || (remainingSeconds > 0 ? `${finishTime} 준비 예상` : '모델 탑재 확인 중')
    }
  }



export function imageGenerationEstimateSeconds(job, jobs) {
    const key = imageGenerationKey(job)
    const exactJobs = jobs
      .filter((item) => item.kind === 'image' && item.status === 'completed' && item.id !== job.id && imageGenerationKey(item) === key)
      .sort((left, right) => Date.parse(right.updated_at || 0) - Date.parse(left.updated_at || 0))
      .slice(0, 12)
    const exact = exactJobs.map(imageJobDurationSeconds).filter((seconds) => seconds > 0)
    const jobStarted = Date.parse(job.params?.started_at || job.created_at || 0)
    const previousFinished = Date.parse(exactJobs[0]?.updated_at || 0)
    // ComfyUI lazily loads the text encoder, DiT checkpoint and VAE. A run
    // immediately following the same pipeline is usually warm; after an idle
    // interval it may have to load all three again. Mixing both populations
    // produced estimates such as 17s for an 86s cold run.
    const warmWindowSeconds = 3 * 60
    const secondsSincePrevious = (jobStarted - previousFinished) / 1000
    const likelyWarm = Number.isFinite(secondsSincePrevious) && secondsSincePrevious >= 0 && secondsSincePrevious <= warmWindowSeconds
    const exactMedian = percentile(exact, .5)
    const warmExact = exact.filter((seconds) => seconds <= exactMedian)
    const warmObserved = percentile(warmExact, .5)
    const coldObserved = percentile(exact, .85)
    const observed = likelyWarm ? (warmObserved || coldObserved) : coldObserved
    if (observed) return Math.max(3, observed)
    const params = job.params || {}
    const mode = params.mode || 'create'
    const comparable = jobs
      .filter((item) => item.kind === 'image' && item.status === 'completed' && item.id !== job.id && (item.params?.mode || 'create') === mode)
      .sort((left, right) => imageGenerationDistance(job, left) - imageGenerationDistance(job, right))
      .slice(0, 12)
    const normalized = comparable.map((item) => imageJobDurationSeconds(item) / imageGenerationWork(item))
    const rate = percentile(normalized, .6)
    if (rate) return Math.max(3, rate * imageGenerationWork(job))
    const megapixels = Math.max(.25, (Number(params.width) || 1024) * (Number(params.height) || 1024) / 1_000_000)
    if (mode === 'garment_extract') return Math.max(3, 3 * megapixels)
    if (mode === 'upscale') return Math.max(20, 16 * megapixels * Math.max(1, Number(params.upscale_scale) || 2) / 2)
    return Math.max(8, 4 + imageGenerationWork(job) * 1.15)
  }

export function imageGenerationProgress(job, jobs, progressClock) {
    const created = Date.parse(job.created_at || 0)
    if (job.status === 'queued') {
      const elapsed = Number.isFinite(created) ? (progressClock - created) / 1000 : 0
      const position = generationQueuePosition(job, jobs)
      const engineWait = job.params?.stage === 'waiting_upscale_engine' || job.params?.stage === 'waiting_video_engine'
      const planned = job.params?.model_plan?.label ? ` · ${job.params.model_plan.label} 예정` : ''
      return { label: engineWait ? '업스케일러 대기' : position ? `대기 ${position}번째` : '대기 중', percent: 2, elapsed: `대기 ${compactElapsed(elapsed)}`, eta: (engineWait ? '이전 SeedVR2 작업 종료 후 시작' : '앞선 작업 종료 후 시작') + planned }
    }
    const preparing = modelPreparationProgress(job, jobs, progressClock)
    if (preparing) return preparing
    const started = Date.parse(job.params?.generation_started_at || job.params?.started_at || job.updated_at || job.created_at || 0)
    const elapsedSeconds = Number.isFinite(started) ? Math.max(0, (progressClock - started) / 1000) : 0
    let estimateSeconds = imageGenerationEstimateSeconds(job, jobs)
    // If a supposedly warm request crosses its estimate, stop claiming that
    // it is merely finishing. The process was evicted or had to reload, so use
    // the conservative observed bound while it continues.
    if (elapsedSeconds > estimateSeconds) {
      const exact = jobs
        .filter((item) => item.kind === 'image' && item.status === 'completed' && item.id !== job.id && imageGenerationKey(item) === imageGenerationKey(job))
        .map(imageJobDurationSeconds)
      estimateSeconds = Math.max(estimateSeconds, percentile(exact, .85))
    }
    const remainingSeconds = estimateSeconds - elapsedSeconds
    const percent = Math.min(94, Math.max(5, elapsedSeconds / estimateSeconds * 100))
    const finishTime = new Date(progressClock + Math.max(0, remainingSeconds) * 1000).toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit', second: '2-digit', hourCycle: 'h23' })
    const timing = `${Math.round(elapsedSeconds)}/${Math.round(estimateSeconds)}초`
    const runtime = runtimePhasePresentation(job)
    return {
      label: runtime?.label || (remainingSeconds > 0 ? `${Math.round(percent)}%` : '마무리 중'),
      percent: runtime?.progress > 0 ? Math.max(5, runtime.progress) : percent,
      elapsed: timing,
      eta: runtime?.detail || (remainingSeconds > 0 ? `${finishTime} 완료 예상` : '')
    }
  }

export function videoGenerationEstimateSeconds(job, jobs) {
    const pipeline = videoGenerationPipeline(job)
    // SeedVR2 is deterministic enough to model its execution phases directly.
    // Never copy a completed upscale job's wall time: it may contain queue
    // waiting, an application restart or a duplicated retry.
    if (pipeline === 'upscale') return videoUpscaleEstimateSeconds(job.params || {})
    const exact = jobs
      .filter((item) => item.kind === 'video' && item.status === 'completed' && item.id !== job.id && videoGenerationKey(item) === videoGenerationKey(job))
      .slice(0, 12)
      .map(videoGenerationDurationSeconds)
    const exactObserved = percentile(exact, .6)
    if (exactObserved) return Math.max(10, exactObserved)
    const candidates = jobs
      .filter((item) => item.kind === 'video' && item.status === 'completed' && item.id !== job.id && videoGenerationPipeline(item) === pipeline)
    const sameAcceleration = candidates.filter((item) => videoAccelerationProfile(item) === videoAccelerationProfile(job))
    const comparable = (sameAcceleration.length ? sameAcceleration : candidates)
      .sort((left, right) => videoGenerationDistance(job, left) - videoGenerationDistance(job, right))
      .slice(0, 20)
    // A2V uses the 22B dev model for a 30-step low-resolution pass followed
    // by a 3-step full-resolution pass.  It must never inherit the much faster
    // distilled T2V/I2V rate. Different resolutions scale non-linearly, so
    // completed runs calibrate the GB10 curve rather than using seconds/pixel.
    if (pipeline === 'a2v') {
      const corrections = comparable.map((item) => videoGenerationDurationSeconds(item) / a2vBaselineEstimateSeconds(item))
      const correction = percentile(corrections, .6) || 1.1
      return a2vBaselineEstimateSeconds(job) * Math.max(.8, Math.min(1.5, correction))
    }
    const normalized = comparable.map((item) => videoGenerationDurationSeconds(item) / videoGenerationWork(item))
    const rate = percentile(normalized, .6)
    if (rate) return Math.max(10, rate * videoGenerationWork(job))
    return Math.max(30, videoGenerationWork(job) * 2.5)
  }

export function videoGenerationProgress(job, jobs, progressClock) {
    const created = Date.parse(job.created_at || 0)
    if (job.status === 'queued') {
      const elapsed = Number.isFinite(created) ? (progressClock - created) / 1000 : 0
      const position = generationQueuePosition(job, jobs)
      const planned = job.params?.model_plan?.label ? ` · ${job.params.model_plan.label} 예정` : ''
      return { label: position ? `대기 ${position}번째` : '대기 중', percent: 2, elapsed: `대기 ${compactElapsed(elapsed)}`, eta: '앞선 작업 종료 후 시작' + planned }
    }
    const preparing = modelPreparationProgress(job, jobs, progressClock)
    if (preparing) return preparing
    const started = Date.parse(job.params?.generation_started_at || job.params?.started_at || job.updated_at || job.created_at || 0)
    const elapsedSeconds = Number.isFinite(started) ? Math.max(0, (progressClock - started) / 1000) : 0
    const estimateSeconds = videoGenerationEstimateSeconds(job, jobs)
    const remainingSeconds = estimateSeconds - elapsedSeconds
    const percent = Math.min(94, Math.max(5, elapsedSeconds / estimateSeconds * 100))
    const finishTime = new Date(progressClock + Math.max(0, remainingSeconds) * 1000).toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit', second: '2-digit', hourCycle: 'h23' })
    const phaseSwapped = job.params?.model_plan?.phase_swapped === true
    const runtime = runtimePhasePresentation(job)
    return {
      label: runtime?.label || (remainingSeconds > 0 ? `${phaseSwapped ? '단계별 모델 적재·생성 · ' : ''}${Math.round(percent)}%` : '마무리 중'),
      percent: runtime?.progress > 0 ? Math.max(5, runtime.progress) : percent,
      elapsed: `${Math.round(elapsedSeconds)}/${Math.round(estimateSeconds)}초`,
      eta: runtime?.detail || (remainingSeconds > 0 ? `${finishTime} 완료 예상` : '')
    }
  }

export function speechGenerationEstimateSeconds(job, jobs) {
    const exact = jobs
      .filter((item) => item.kind === 'speech' && item.status === 'completed' && item.id !== job.id && speechGenerationKey(item) === speechGenerationKey(job))
      .slice(0, 12)
      .map(imageJobDurationSeconds)
    const exactObserved = percentile(exact, .6)
    if (exactObserved) return Math.max(1, exactObserved)
    const params = job.params || {}
    const comparable = jobs
      .filter((item) => item.kind === 'speech' && item.status === 'completed' && item.id !== job.id)
      .sort((left, right) => {
        const a = left.params || {}, b = right.params || {}
        const leftScore = Math.abs(Math.log(speechGenerationWork(job) / speechGenerationWork(left))) + ((params.language || 'Auto') === (a.language || 'Auto') ? 0 : .8) + ((params.speaker || '') === (a.speaker || '') ? 0 : .25) + (Boolean(params.instructions) === Boolean(a.instructions) ? 0 : .2)
        const rightScore = Math.abs(Math.log(speechGenerationWork(job) / speechGenerationWork(right))) + ((params.language || 'Auto') === (b.language || 'Auto') ? 0 : .8) + ((params.speaker || '') === (b.speaker || '') ? 0 : .25) + (Boolean(params.instructions) === Boolean(b.instructions) ? 0 : .2)
        return leftScore - rightScore
      })
      .slice(0, 12)
    const rates = comparable.map((item) => imageJobDurationSeconds(item) / speechGenerationWork(item))
    const rate = percentile(rates, .6)
    return Math.max(1, rate ? rate * speechGenerationWork(job) : .12 * speechGenerationWork(job))
  }

export function speechGenerationProgress(job, jobs, progressClock) {
    if (job.status === 'queued') {
      const created = Date.parse(job.created_at || 0)
      const elapsed = Number.isFinite(created) ? (progressClock - created) / 1000 : 0
      const position = generationQueuePosition(job, jobs)
      const planned = job.params?.model_plan?.label ? ` · ${job.params.model_plan.label} 예정` : ''
      return { label: position ? `대기 ${position}번째` : '대기 중', percent: 2, elapsed: `대기 ${compactElapsed(elapsed)}`, eta: '앞선 작업 종료 후 시작' + planned }
    }
    const preparing = modelPreparationProgress(job, jobs, progressClock)
    if (preparing) return preparing
    const started = Date.parse(job.params?.generation_started_at || job.params?.started_at || job.updated_at || job.created_at || 0)
    const elapsedSeconds = Number.isFinite(started) ? Math.max(0, (progressClock - started) / 1000) : 0
    const estimateSeconds = speechGenerationEstimateSeconds(job, jobs)
    const remainingSeconds = estimateSeconds - elapsedSeconds
    const runtime = runtimePhasePresentation(job)
    return {
      label: runtime?.label || (remainingSeconds > 0 ? `${Math.min(94, Math.max(5, Math.round(elapsedSeconds / estimateSeconds * 100)))}%` : '마무리 중'),
      percent: runtime?.progress > 0 ? Math.max(5, runtime.progress) : Math.min(94, Math.max(5, elapsedSeconds / estimateSeconds * 100)),
      elapsed: `${Math.round(elapsedSeconds)}/${Math.round(estimateSeconds)}초`,
      eta: runtime?.detail || (remainingSeconds > 0 ? `${new Date(progressClock + remainingSeconds * 1000).toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit', second: '2-digit', hourCycle: 'h23' })} 완료 예상` : '')
    }
  }

export function recognitionProgressText(job, jobs) {
    const params = job.params || {}
    if (job.status === 'cancelled') return '중지됨'
	if (job.status === 'queued') {
	  const position = recognitionQueuePosition(job, jobs)
	  return position ? `대기 ${position}번째 · 앞선 작업 완료 후 자동 시작` : '대기 중'
	}
    if (params.stage === 'model-preparing') {
      const label = params.model_prepare_label || params.model_plan?.label || '음성 인식 모델 준비'
      return label
    }
    const runtime = runtimePhasePresentation(job)
    if (runtime && runtime.phase !== 'completed') return runtime.detail ? `${runtime.label} · ${runtime.detail}` : runtime.label
    if (params.stage === 'media') {
      const labels = {
        starting: '미디어 준비 시작 중', resuming: '저장된 원본에서 작업 재개 중', receiving: '파일 전송 중', resolving: '영상 페이지 분석 중',
        storing: '미디어 저장·재생 형식 정리 중', extracting_audio: '음성 추출·분할 중', complete: '미디어 준비 마무리 중'
      }
      if (params.media_stage === 'downloading') {
        const percent = Number(params.media_percent) || 0
        const amount = params.media_total_bytes ? ` · ${formatBytes(params.media_downloaded_bytes)} / ${formatBytes(params.media_total_bytes)}` : ''
        const eta = params.media_eta_seconds ? ` · 약 ${params.media_eta_seconds}초 남음` : ''
        return `미디어 다운로드 ${percent.toFixed(1)}%${amount}${eta}`
      }
      return labels[params.media_stage] || '미디어 준비 중'
    }
    if (params.stage === 'recognition') return params.segments ? `음성 인식 ${params.progress || 0}/${params.segments} 구간` : '음성 인식 준비 중'
    if (params.stage === 'translation') return `자막 번역 ${params.translation_progress || 0}/${params.translation_total || 0} 배치`
    if (params.stage === 'finalizing') return '자막 파일 생성 중'
    return job.status
  }

export function recognitionProgressTiming(job, progressClock) {
    const params = job.params || {}
    if (job.status !== 'running') return ''
    const stageStarted = Date.parse(params.stage_started_at || params.started_at || job.created_at || 0)
    const elapsed = Number.isFinite(stageStarted) ? Math.max(0, (progressClock - stageStarted) / 1000) : 0
    if (params.stage === 'model-preparing') {
      const estimate = Math.max(1, Number(params.model_prepare_estimate_seconds) || 5)
      return `${compactElapsed(elapsed)}/${compactElapsed(estimate)}`
    }
    if (params.stage === 'media' && params.media_stage === 'downloading' && Number(params.media_eta_seconds) > 0) {
      return `${compactElapsed(elapsed)}/${compactElapsed(elapsed + Number(params.media_eta_seconds))}`
    }
    let done = 0, total = 0
    if (params.stage === 'recognition') {
      done = Number(params.progress) || 0
      total = Number(params.segments) || 0
    } else if (params.stage === 'translation') {
      done = Number(params.translation_progress) || 0
      total = Number(params.translation_total) || 0
    }
    if (done > 0 && total > 0) {
      const estimate = elapsed * total / done
      return `${compactElapsed(elapsed)}/${compactElapsed(Math.max(elapsed, estimate))}`
    }
    return elapsed > 0 ? `${compactElapsed(elapsed)} 경과` : ''
  }

export function recognitionProgressPercent(job) {
    const params = job.params || {}
	if (job.status === 'queued') return 2
    if (params.stage === 'model-preparing') {
      const started = Date.parse(params.model_prepare_started_at || params.stage_started_at || 0)
      const elapsed = Number.isFinite(started) ? Math.max(0, (Date.now() - started) / 1000) : 0
      return Math.min(94, Math.max(3, elapsed * 100 / Math.max(1, Number(params.model_prepare_estimate_seconds) || 5)))
    }
    if (params.stage === 'media' && params.media_stage === 'downloading') return Math.min(100, Math.max(0, Number(params.media_percent) || 0))
    if (params.stage === 'recognition' && params.segments) return Math.min(100, (Number(params.progress) || 0) * 100 / params.segments)
    if (params.stage === 'translation' && params.translation_total) return Math.min(100, (Number(params.translation_progress) || 0) * 100 / params.translation_total)
    return 0
  }

export function recognitionQueuePosition(job, jobs) {
	const queued = jobs
	  .filter((item) => item.kind === 'recognition' && item.status === 'queued')
	  .sort((a, b) => {
		const left = Date.parse(a.params?.queued_at || a.created_at || 0)
		const right = Date.parse(b.params?.queued_at || b.created_at || 0)
		return left - right || String(a.id).localeCompare(String(b.id))
	  })
	const index = queued.findIndex((item) => item.id === job.id)
	return index < 0 ? 0 : index + 1
  }

export function generationQueuePosition(job, jobs) {
	const queued = jobs
	  .filter((item) => ['image', 'video', 'speech'].includes(item.kind) && item.status === 'queued')
	  .sort((a, b) => {
		const left = Date.parse(a.params?.queued_at || a.created_at || 0)
		const right = Date.parse(b.params?.queued_at || b.created_at || 0)
		return left - right || String(a.id).localeCompare(String(b.id))
	  })
	const index = queued.findIndex((item) => item.id === job.id)
	return index < 0 ? 0 : index + 1
  }
