import { snapDimension } from './videoTiming.js'

function completedImage(env, index) {
  const job = env.getImageJobs()[Number(index) - 1]
  if (!job?.output_url || job.status !== 'completed') return null
  return {
    server: true,
    ref: `${job.id}:output:0`,
    url: job.output_url,
    name: `생성 이미지 #${index}`,
    role: 'output',
  }
}

export function applyAssistantActionList(actions = [], env) {
  for (const action of actions) {
    if (action.type === 'navigate' && action.tab) env.switchTab(action.tab)
    else if (action.type === 'show_results' && action.tab) env.switchTab(action.tab, true)
    else if (action.type === 'open_modules') {
      env.switchTab('image')
      env.setFeatureModulesOpen(true)
    } else if (action.type === 'set_image') {
      env.switchTab('image')
      env.setImageForm({
        ...env.getImageForm(),
        ...(action.prompt != null ? { prompt: action.prompt } : {}),
        ...(action.width >= 256 ? { width: snapDimension(action.width, 8, 256, 2048) } : {}),
        ...(action.height >= 256 ? { height: snapDimension(action.height, 8, 256, 2048) } : {}),
        ...(action.seed != null ? { seed: Math.round(Number(action.seed)) } : {}),
      })
      if (action.enhance_enabled != null) env.setImageEnhanceEnabled(action.enhance_enabled)
      env.setImageResolutionMode('custom')
      env.resetImageEnhancement()
    } else if (action.type === 'set_video') {
      env.switchTab('video')
      env.setVideoForm({
        ...env.getVideoForm(),
        ...(action.prompt != null ? { prompt: action.prompt } : {}),
        ...(action.width >= 256 ? { width: snapDimension(action.width, 64, 256, 1920) } : {}),
        ...(action.height >= 256 ? { height: snapDimension(action.height, 64, 256, 1920) } : {}),
        ...(action.fps > 0 ? { fps: action.fps } : {}),
        ...(action.seed != null ? { seed: Math.round(Number(action.seed)) } : {}),
      })
      if (action.duration > 0) {
        env.setVideoDurationSeconds(env.snapVideoDuration(action.duration))
        env.normalizeVideoTiming()
      }
      if (action.enhance_enabled != null) env.setVideoEnhanceEnabled(action.enhance_enabled)
      env.resetVideoEnhancement()
    } else if (action.type === 'set_speech') {
      env.switchTab('speech')
      env.setSpeechForm({
        ...env.getSpeechForm(),
        ...(action.text != null ? { text: action.text } : {}),
        ...(action.instructions != null ? { instructions: action.instructions } : {}),
        ...(action.language ? { language: action.language } : {}),
        ...(action.speaker ? { speaker: action.speaker } : {}),
        ...(action.seed != null ? { seed: Math.round(Number(action.seed)) } : {}),
      })
    } else if (action.type === 'set_recognition') {
      env.switchTab('recognition')
      env.setRecognitionForm({
        ...env.getRecognitionForm(),
        ...(action.context != null ? { context: action.context } : {}),
        ...(action.language ? { language: action.language } : {}),
        ...(action.translation_mode ? { translation_mode: action.translation_mode } : {}),
        ...(action.target_language ? { target_language: action.target_language } : {}),
      })
    } else if (action.type === 'set_module' && action.module in env.getKreaModules()) {
      env.switchTab('image')
      const desired = action.enabled !== false
      if (Boolean(env.getKreaModules()[action.module]) !== desired) env.toggleKreaModule(action.module)
      if (desired && action.module === 'identity' && action.preset != null) env.applyIdentityPreset(action.preset)
      env.setFeatureModulesOpen(true)
    } else if (action.type === 'set_recent_image') {
      const image = completedImage(env, action.image_index)
      if (!image) continue
      if (action.target === 'vision' || action.target === 'styleReference') env.addKreaRefObjects(action.target, [image])
      else if (['identity', 'identityReference', 'depth', 'nk2e', 'anypaint'].includes(action.target)) env.setKreaImage(action.target, image)
      env.switchTab('image')
      env.setFeatureModulesOpen(true)
    } else if (action.type === 'set_outpaint') {
      const image = completedImage(env, action.image_index)
      if (!image) continue
      env.setKreaModules(Object.fromEntries(Object.keys(env.getKreaModules()).map((name) => [name, name === 'anypaint'])))
      env.setKreaImage('anypaint', image)
      env.setKreaImage('anypaintMask', null)
      env.setKreaOptions({
        ...env.getKreaOptions(),
        outpaint_left: Number(action.outpaint_left) || 0,
        outpaint_top: Number(action.outpaint_top) || 0,
        outpaint_right: Number(action.outpaint_right) || 0,
        outpaint_bottom: Number(action.outpaint_bottom) || 0,
      })
      env.setImageForm({ ...env.getImageForm(), mode: 'create', prompt: '' })
      env.resetImageEnhancement()
      env.switchTab('image')
      env.setFeatureModulesOpen(true)
    }
  }
}
