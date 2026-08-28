import { upscaleFrameWork } from '../videoUpscaleEta.js'



export function imageGenerationKey(job) {
    const params = job.params || {}
    const mode = params.mode || 'create'
    const steps = Number(params.steps) || (mode === 'detail_enhance' ? 10 : 8)
    const sampler = params.sampler || (mode === 'detail_enhance' ? 'er_sde' : 'euler')
    const modules = ['identity', 'depth', 'vision', 'style_reference', 'nk2e', 'anypaint'].filter((name) => params[name]).join('+') || 'base'
    const checkpoint = params.checkpoint || params.model || 'official'
    const textEncoder = params.text_encoder || params.encoder || 'default'
    const loraCount = (Array.isArray(params.user_loras) ? params.user_loras.length : 0) + (Array.isArray(params.styles) ? params.styles.length : 0)
    const references = imageReferenceCount(job)
    const sequence = params.sequence_strategy === 'storyboard' ? 'single' : (params.sequence_strategy || 'single')
    return `${mode}|${checkpoint}|${textEncoder}|${sampler}|${params.scheduler || 'simple'}|${steps}|${Number(params.width) || 1024}x${Number(params.height) || 1024}|${modules}|refs:${references}|lora:${loraCount}|vae:${params.vae_mode || params.detail_vae || 'default'}|filter:${params.filter_mode || 'default'}|sequence:${sequence}`
  }

export function imageReferenceCount(job) {
    const params = job.params || {}
    const explicit = Number(params.references)
    if (Number.isFinite(explicit)) return explicit
    return ['identity_reference_count', 'style_reference_count', 'vision_count', 'garment_reference_count']
      .reduce((total, key) => total + (Number(params[key]) || 0), 0)
  }

export function imageGenerationWork(job) {
    const params = job.params || {}
    const mode = params.mode || 'create'
    const megapixels = Math.max(.1, (Number(params.width) || 1024) * (Number(params.height) || 1024) / 1_000_000)
    if (mode === 'garment_extract') return megapixels
    if (mode === 'upscale') return megapixels * Math.max(1, Number(params.upscale_scale) || 2)
    const steps = Number(params.steps) || (mode === 'detail_enhance' ? 10 : 8)
    const modules = ['identity', 'depth', 'vision', 'style_reference', 'nk2e', 'anypaint'].filter((name) => params[name]).length
    const references = imageReferenceCount(job)
    const loras = (Array.isArray(params.user_loras) ? params.user_loras.length : 0) + (Array.isArray(params.styles) ? params.styles.length : 0)
    const sequencePasses = params.sequence_strategy === 'major' && params.sequence_previous_job_id ? 2.35 : 1
    return megapixels * steps * (1 + modules * .28 + references * .08 + loras * .04) * sequencePasses
  }

export function imageGenerationDistance(left, right) {
    const a = left.params || {}, b = right.params || {}
    const ratio = Math.abs(Math.log(imageGenerationWork(left) / imageGenerationWork(right)))
    const mode = (a.mode || 'create') === (b.mode || 'create') ? 0 : 8
    const model = (a.checkpoint || a.model || 'official') === (b.checkpoint || b.model || 'official') ? 0 : 3
    const leftSequence = a.sequence_strategy === 'storyboard' ? 'single' : (a.sequence_strategy || 'single')
    const rightSequence = b.sequence_strategy === 'storyboard' ? 'single' : (b.sequence_strategy || 'single')
    const sequence = leftSequence === rightSequence ? 0 : 2
    const moduleMismatch = ['identity', 'depth', 'vision', 'style_reference', 'nk2e', 'anypaint'].reduce((total, key) => total + (Boolean(a[key]) === Boolean(b[key]) ? 0 : .8), 0)
    return ratio + mode + model + sequence + moduleMismatch + Math.abs(imageReferenceCount(left) - imageReferenceCount(right)) * .15
  }

export function imageJobDurationSeconds(job) {
    const started = Date.parse(job.params?.generation_started_at || job.params?.started_at || job.created_at || 0)
    const completed = Date.parse(job.updated_at || 0)
    if (!Number.isFinite(started) || !Number.isFinite(completed) || completed <= started) return 0
    return (completed - started) / 1000 + Math.max(0, Number(job.params?.sequence_draft_seconds) || 0)
  }

export function percentile(values, fraction = .5) {
    const sorted = values.filter((value) => Number.isFinite(value) && value > 0).sort((a, b) => a - b)
    if (!sorted.length) return 0
    const position = Math.max(0, Math.min(sorted.length - 1, Math.ceil(sorted.length * fraction) - 1))
    return sorted[position]
  }

export function videoGenerationKey(job) {
    const params = job.params || {}
    if (params.mode === 'upscale') return `upscale|${params.model || params.upscale_engine || 'default'}|${params.source_width || 0}x${params.source_height || 0}|${params.upscale_scale || 2}|frames:${upscaleFrameWork(params).sourceFrames}|fps:${Number(params.fps) || 24}|batch:${params.batch_size || 5}|overlap:${params.temporal_overlap ?? 1}`
    const pipeline = videoGenerationPipeline(job)
    const conditions = videoConditionCount(job)
    return `${pipeline}|${params.model || 'ltx'}|${params.acceleration_requested || params.acceleration || 'legacy'}|${Number(params.width) || 0}x${Number(params.height) || 0}|${Number(params.num_frames) || 0}|${Number(params.fps) || 24}|conditions:${conditions}|motion:${params.motion_lora_enabled ? Number(params.motion_lora_strength) || .5 : 0}`
  }

export function videoConditionCount(job) {
    const params = job.params || {}
    return Array.isArray(params.video_conditions) ? params.video_conditions.length : Number(params.keyframes) || (params.image ? 1 : 0)
  }

export function videoAccelerationProfile(job) {
    const params = job.params || {}
    return params.acceleration_requested || params.acceleration || 'legacy'
  }

export function videoGenerationPipeline(job) {
    const params = job.params || {}
    if (params.mode === 'upscale') return 'upscale'
    if (params.mode === 'a2v' || params.stage === 'a2v' || params.audio) return 'a2v'
    return params.image ? 'i2v' : 't2v'
  }

export function videoGenerationWork(job) {
    const params = job.params || {}
    if (params.mode === 'upscale') {
      const scale = Math.max(1, Number(params.upscale_scale) || 2)
      const sourcePixels = Math.max(.1, (Number(params.source_width) || 768) * (Number(params.source_height) || 512) / 1_000_000)
      const outputPixels = Number(params.width) > 0 && Number(params.height) > 0
        ? Math.max(.1, Number(params.width) * Number(params.height) / 1_000_000)
        : sourcePixels * scale * scale
      return outputPixels * upscaleFrameWork(params).processedFrames
    }
    const base = Math.max(.1, (Number(params.width) || 768) * (Number(params.height) || 512) / 1_000_000) * Math.max(9, Number(params.num_frames) || 97)
    const conditions = videoConditionCount(job)
    // Multi-frame conditioning adds encoding work, but the two GB10 samples
    // with 2 and 8 conditions differed by only ~1.2% per extra image. Most
    // denoising work is shared, so treating every condition as another 5% of
    // the whole diffusion pass substantially under-estimated sparse jobs.
    const conditioningCost = 1 + Math.max(0, conditions - 1) * .012
    const motionCost = params.motion_lora_enabled ? 1.03 : 1
    return base * conditioningCost * motionCost
  }

export function videoGenerationDistance(left, right) {
    const a = left.params || {}, b = right.params || {}
    const pipeline = videoGenerationPipeline(left) === videoGenerationPipeline(right) ? 0 : 10
    const work = Math.abs(Math.log(videoGenerationWork(left) / videoGenerationWork(right)))
    const conditions = Math.abs(videoConditionCount(left) - videoConditionCount(right)) * .18
    const motion = Boolean(a.motion_lora_enabled) === Boolean(b.motion_lora_enabled) ? 0 : .8
    // Jobs predating acceleration metadata ran a different LTX path. Do not
    // silently classify them as current `auto` samples: that mixed legacy
    // 768x512 timings into SM121 estimates and caused the 328s vs 391s miss.
    const acceleration = videoAccelerationProfile(left) === videoAccelerationProfile(right) ? 0 : 4
    const model = (a.model || 'ltx') === (b.model || 'ltx') ? 0 : 2
    return pipeline + work + conditions + motion + acceleration + model
  }

export function videoGenerationDurationSeconds(job) {
    const started = Date.parse(job.params?.generation_started_at || job.params?.started_at || job.created_at || 0)
    const completed = Date.parse(job.updated_at || 0)
    if (!Number.isFinite(started) || !Number.isFinite(completed) || completed <= started) return 0
    return (completed - started) / 1000
  }

export function a2vBaselineEstimateSeconds(job) {
    const relativeWork = Math.max(1, videoGenerationWork(job) / (.256 * .256 * 9))
    const stageOne = 30 * 1.92 * Math.pow(relativeWork, .47)
    const stageTwo = 3 * .57 * Math.pow(relativeWork, .62)
    return Math.max(140, 70 + stageOne + stageTwo)
  }

export function speechGenerationKey(job) {
    const params = job.params || {}
    const lengthBand = Math.max(1, Math.round((job.prompt || '').length / 40))
    return `${params.model || 'qwen3-tts'}|${params.language || 'Auto'}|${params.speaker || 'default'}|instructions:${params.instructions ? 1 : 0}|length:${lengthBand}`
  }

export function speechGenerationWork(job) {
    const params = job.params || {}
    const characters = Math.max(8, (job.prompt || '').length)
    return Math.pow(characters, .88) * (params.instructions ? 1.05 : 1)
  }

export function orderedJobs(items, order) {
    return order === 'asc' ? [...items].reverse() : items
  }
