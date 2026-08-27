import { languageCodes, recognitionLanguages } from './catalogs.js'

import { formatDuration } from './videoTiming.js'



export function recognitionLanguageLabel(language) {
    return recognitionLanguages.find(([value]) => value === language)?.[1] || language
  }

export function captionLanguage(job) {
    const language = job.params?.translation_mode === 'none'
      ? job.params?.detected_language || job.params?.language
      : job.params?.target_language
    return languageCodes[language] || 'und'
  }

export function formatBytes(value) {
    const bytes = Number(value) || 0
    if (bytes >= 1073741824) return `${(bytes / 1073741824).toFixed(2)} GB`
    if (bytes >= 1048576) return `${(bytes / 1048576).toFixed(1)} MB`
    if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${bytes} B`
  }

export function isAudioMedia(job) {
    const media = job.params?.media || {}
    return media.media_type === 'audio' || String(media.content_type || '').startsWith('audio/')
  }

export function mediaSummary(job) {
    const media = job.params?.media
    if (!media) return ''
    const dimensions = !isAudioMedia(job) && media.width && media.height ? `${media.width}×${media.height} · ` : ''
    return `${dimensions}${formatDuration(media.duration)} · ${formatBytes(media.size)}`
  }

export function subtitleTranslationWarnings(job) {
    return Array.isArray(job.params?.translation_warnings) ? job.params.translation_warnings : []
  }

export function subtitleTranslationWarningText(job) {
    return subtitleTranslationWarnings(job).map((warning) => {
      const segment = Number(warning?.segment) || '?'
      return `${segment}번 자막\n원문: ${warning?.source || '(내용 없음)'}\n${warning?.reason || '번역하지 못해 원문을 유지했습니다.'}`
    }).join('\n\n')
  }

export function videoJobDuration(job) {
    if (job.params?.mode === 'upscale') return Math.max(0, Number(job.params?.duration) || 0)
    return (Math.max(1, Number(job.params?.num_frames) || 1) - 1) / Math.max(1, Number(job.params?.fps) || 1)
  }

export function videoFPSLabel(job) {
    return Number(job.params?.fps) > 0 ? ` · ${job.params.fps} fps` : ''
  }

export function videoAccelerationLabel(job) {
    if (job.params?.mode === 'upscale') {
      const scale = Number(job.params?.upscale_scale) || 2
      return `SeedVR2 ${scale.toFixed(2).replace(/\.?0+$/, '')}×`
    }
    const actual = job.params?.acceleration
    if (actual === 'cute_sm121+exact-adaln') return 'SOL Attn'
    if (actual === 'cute_sm121') return 'SOL Attn'
    if (actual === 'dense') return 'Dense'
    return job.params?.acceleration_requested === 'auto' ? '자동' : ''
  }

export function imageModuleSummary(job) {
    const params = job.params || {}
    if (params.mode !== 'create') return ''
    const modules = []
    if (params.identity && !params.sequence_total) modules.push('Identity')
    if (params.depth) modules.push('Depth')
    if (params.styles?.length || params.style) modules.push(`LoRA${params.styles?.length > 1 ? ` ×${params.styles.length}` : ''}`)
    if (params.user_loras?.length) modules.push(`사용자 LoRA${params.user_loras.length > 1 ? ` ×${params.user_loras.length}` : ''}`)
    if (params.style_reference) modules.push('Style Ref')
    if (params.vision) modules.push('Vision')
    if (params.nk2e) modules.push(params.nk2e_mode === 'canny' ? 'NK2E Canny' : 'NK2E Edit')
    if (params.anypaint) modules.push(params.anypaint_mask ? 'Inpaint' : 'Outpaint')
    if (params.sequence_total) modules.push(`연속 ${params.sequence_index}/${params.sequence_total}`)
    return modules.length ? ` · ${modules.join(' + ')}` : ''
  }

export function imageSamplingSummary(job) {
    const params = job.params || {}
    if (!params.sampler && !params.scheduler && !params.steps) return ''
    return `${params.sampler || '—'} / ${params.scheduler || '—'} · ${params.steps || '—'} steps`
  }

export function compactElapsed(seconds) {
    const value = Math.max(0, Math.round(Number(seconds) || 0))
    if (value < 60) return `${value}초`
    const minutes = Math.floor(value / 60)
    const remainder = value % 60
    if (minutes < 60) return remainder ? `${minutes}분 ${remainder}초` : `${minutes}분`
    const hours = Math.floor(minutes / 60)
    return `${hours}시간 ${minutes % 60}분`
  }

export function imagePromptModalText(job) {
    const enhanced = job.params?.generated_edit_prompt || job.params?.enhanced_prompt || job.params?.source_enhanced_prompt
    if (!enhanced) return job.prompt || ''
    return `원문\n${job.prompt || ''}\n\n실제 생성 프롬프트\n${enhanced}`
  }

export function videoPromptModalText(job) {
    const original = job.prompt || ''
    const enhanced = job.params?.enhanced_prompt || job.params?.source_enhanced_prompt
    if (!enhanced || enhanced.trim() === original.trim()) return original
    return `원문\n${original}\n\n실제 생성 프롬프트\n${enhanced}`
  }

