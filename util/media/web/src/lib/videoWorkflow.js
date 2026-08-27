import { durationFromFrames, framesForDuration } from './videoTiming.js'

export function videoResolutionPresetID(form, presets) {
  return presets.find((preset) => preset.width === Number(form.width) && preset.height === Number(form.height))?.id || 'custom'
}

export function videoStage2TokenCount(width, height, seconds, fps) {
  const frames = framesForDuration(seconds, fps)
  return (Math.floor((frames - 1) / 8) + 1) * Math.floor(Number(width) / 32) * Math.floor(Number(height) / 32)
}

export function videoAccelerationPreview({ acceleration, tokens }) {
  if (acceleration === 'dense') return 'Dense · 가속 꺼짐'
  if (tokens < 32000) return `Dense · ${tokens.toLocaleString()} tokens`
  return `SOL Attn · ${tokens.toLocaleString()} tokens`
}

export function videoInputKey(image) {
  if (!image) return ''
  return image.server ? image.ref : `${image.name}:${image.size}:${image.lastModified}`
}

export function videoEnhancementCurrent({ enhanced, source, prompt, imageKey, currentImageKey }) {
  return enhanced.trim() !== '' && source === prompt.trim() && imageKey === currentImageKey
}

export function videoEnhancementActive({ enabled, image, visionEnabled }) {
  return enabled && !(image && !visionEnabled)
}

export function videoKeyframeCapacity(seconds, fps) {
  return Math.max(0, Math.min(8, framesForDuration(seconds, fps) - 2))
}

export function nearestAvailableVideoKeyframeFrame({ rawFrame, excludeID = null, seconds, fps, keyframes }) {
  const lastFrame = framesForDuration(seconds, fps) - 1
  if (lastFrame <= 1) return null
  const occupied = new Set(keyframes.filter((item) => item.id !== excludeID).map((item) => Math.round(Number(item.time) * fps)))
  const desired = Math.min(lastFrame - 1, Math.max(1, Math.round(Number(rawFrame) || 1)))
  if (!occupied.has(desired)) return desired
  for (let distance = 1; distance < lastFrame; distance++) {
    const right = desired + distance
    const left = desired - distance
    if (right < lastFrame && !occupied.has(right)) return right
    if (left > 0 && !occupied.has(left)) return left
  }
  return null
}

export function normalizeVideoTiming({ seconds, fps, keyframes, audioClips }) {
  const normalizedFPS = Math.max(1, Number(fps) || 1)
  let frames = framesForDuration(seconds, normalizedFPS)
  while (frames - 2 < keyframes.length) frames += 8
  const duration = durationFromFrames(frames, normalizedFPS)
  const lastFrame = frames - 1
  const occupied = new Set()
  const normalizedKeyframes = keyframes.map((keyframe) => {
    const desired = Math.min(lastFrame - 1, Math.max(1, Math.round(Number(keyframe.time) * normalizedFPS)))
    let frame = desired
    if (occupied.has(frame)) {
      for (let distance = 1; distance < lastFrame; distance++) {
        const right = desired + distance
        const left = desired - distance
        if (right < lastFrame && !occupied.has(right)) { frame = right; break }
        if (left > 0 && !occupied.has(left)) { frame = left; break }
      }
    }
    occupied.add(frame)
    return { ...keyframe, time: frame / normalizedFPS }
  })
  const normalizedAudioClips = audioClips.map((clip) => ({
    ...clip,
    start: Math.min(Math.max(0, duration - Math.min(Number(clip.duration) || 0, duration)), Math.max(0, Number(clip.start) || 0))
  }))
  return { duration, keyframes: normalizedKeyframes, audioClips: normalizedAudioClips }
}

export function videoConditioningDisabledReason({ audioSelected, a2vReady, seconds, fps, keyframes }) {
  if (audioSelected && !a2vReady) return 'A2V 모델을 준비 중입니다. 설정 · 연결에서 진행 상태를 확인하세요.'
  const frameCount = framesForDuration(seconds, fps)
  const finalTime = (frameCount - 1) / Number(fps || 1)
  const occupied = new Set()
  for (let index = 0; index < keyframes.length; index++) {
    const keyframe = keyframes[index]
    if (!keyframe.image) continue
    const frame = Math.round(Number(keyframe.time) * Number(fps))
    if (!(Number(keyframe.time) > 0 && Number(keyframe.time) < finalTime) || frame <= 0 || frame >= frameCount - 1) return `키프레임 ${index + 1} 위치를 시작과 마지막 사이로 지정하세요.`
    if (occupied.has(frame)) return '같은 프레임 위치에 키프레임을 두 개 배치할 수 없습니다.'
    occupied.add(frame)
  }
  return ''
}
