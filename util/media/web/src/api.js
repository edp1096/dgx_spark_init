async function checked(response) {
  if (!response.ok) throw new Error((await response.text()) || `HTTP ${response.status}`)
  return response.json()
}

async function checkedBlob(response) {
  if (!response.ok) throw new Error((await response.text()) || `HTTP ${response.status}`)
  return response.blob()
}

export const api = {
  config: () => fetch('/api/config').then(checked),
  saveConfig: (config) => fetch('/api/config', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config)
  }).then(checked),
  jobs: () => fetch('/api/jobs').then(checked),
  tags: () => fetch('/api/tags').then(checked),
  updateJobTags: (id, tags) => fetch(`/api/jobs/${encodeURIComponent(id)}/tags`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ tags })
  }).then(checked),
  imageInputs: (id) => fetch(`/api/jobs/${encodeURIComponent(id)}/inputs`).then(checked),
  imageEXIF: (id) => fetch(`/api/jobs/${encodeURIComponent(id)}/exif`).then(checked),
  engines: () => fetch('/api/engines').then(checked),
  system: () => fetch('/api/system').then(checked),
  videoModels: () => fetch('/api/video/models').then(checked),
  prepareVideoModels: (hfToken) => fetch('/api/video/models/prepare', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ hf_token: hfToken })
  }).then(checked),
  imageCheckpoints: () => fetch('/api/image/checkpoints').then(checked),
  prepareImageCheckpoints: (civitaiToken, hfToken, variants) => fetch('/api/image/checkpoints/prepare', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ civitai_token: civitaiToken, hf_token: hfToken, variants })
  }).then(checked),
  convertImageCheckpointsNVFP4: (civitaiToken, variants, removeBF16Sources = false) => fetch('/api/image/checkpoints/convert-nvfp4', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ civitai_token: civitaiToken, variants, remove_bf16_sources: removeBF16Sources })
  }).then(checked),
  assistantChat: (payload) => fetch('/api/assistant/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  }).then(checked),
  remoteImage: async (url) => {
    const response = await fetch('/api/images/fetch', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ url })
    })
    if (!response.ok) throw new Error((await response.text()) || `HTTP ${response.status}`)
    return {
      blob: await response.blob(),
      filename: response.headers.get('X-Image-Filename') || 'url-image.png'
    }
  },
  deleteJob: (id) => fetch(`/api/jobs/${encodeURIComponent(id)}`, { method: 'DELETE' }).then(async (response) => {
    if (!response.ok) throw new Error((await response.text()) || `HTTP ${response.status}`)
  }),
  cancelJob: (id) => fetch(`/api/jobs/${encodeURIComponent(id)}/cancel`, { method: 'POST' }).then(checked),
  retryJob: (id) => fetch(`/api/jobs/${encodeURIComponent(id)}/retry`, { method: 'POST' }).then(checked),
  deleteFinishedJobs: () => fetch('/api/jobs', { method: 'DELETE' }).then(checked),
  image: (form) => fetch('/api/jobs/image', { method: 'POST', body: form }).then(checked),
  upscaleImage: (id, options = { scale: 2, seed: -1 }) => fetch(`/api/jobs/${encodeURIComponent(id)}/upscale`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(options)
  }).then(checked),
  upscaleVideo: (id, options = { scale: 2, seed: -1, batch_size: 5, temporal_overlap: 1 }) => fetch(`/api/jobs/${encodeURIComponent(id)}/video-upscale`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(options)
  }).then(checked),
  detailEnhanceImage: (id, options = { strength: 1, seed: -1, vae: 'wan' }) => fetch(`/api/jobs/${encodeURIComponent(id)}/detail-enhance`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(options)
  }).then(checked),
  garmentExtract: (form) => fetch('/api/jobs/garment-extract', { method: 'POST', body: form }).then(checked),
  faceSwap: (form) => fetch('/api/jobs/face-swap', { method: 'POST', body: form }).then(checked),
  speech: (form) => fetch('/api/jobs/speech', { method: 'POST', body: form }).then(checked),
  recognition: (form) => fetch('/api/jobs/recognition', { method: 'POST', body: form }).then(checked),
  regenerateSubtitle: (id, options) => fetch(`/api/jobs/${encodeURIComponent(id)}/subtitle-regenerate`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(options)
  }).then(checked),
  mediaOptions: (url) => fetch('/api/media/options', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({ url })
  }).then(checked),
  storage: () => fetch('/api/storage').then(checked),
  cleanupTemporaryStorage: () => fetch('/api/storage/temp', { method: 'DELETE' }).then(checked),
  video: (form) => fetch('/api/jobs/video', { method: 'POST', body: form }).then(checked),
  enhancePrompt: (form) => fetch('/api/prompts/enhance', { method: 'POST', body: form }).then(checked),
	describeSequenceCharacter: (form) => fetch('/api/prompts/character-description', { method: 'POST', body: form }).then(checked),
	createSequenceCharacterSheet: (form) => fetch('/api/images/character-sheet', { method: 'POST', body: form }).then(checkedBlob),
	sequenceCharacterSheetStatus: (operationID) => fetch(`/api/images/character-sheet/status?operation_id=${encodeURIComponent(operationID)}`).then(checked),
	planImageSequence: (payload) => fetch('/api/prompts/sequence-plan', {
		method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
	}).then(checked),
	randomPromptWildcard: (variant = 'no_camera') => fetch(`/api/prompts/wildcard?variant=${encodeURIComponent(variant)}`).then(checked),
  loraStatus: () => fetch('/api/lora/status').then(checked),
  saveLoraTokens: (civitaiToken, hfToken) => fetch('/api/lora/tokens', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ civitai_token: civitaiToken, hf_token: hfToken })
  }).then(checked),
  userLoras: () => fetch('/api/lora').then(checked),
  importUserLora: (request) => fetch('/api/lora/import', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(request)
  }).then(checked),
  uploadUserLora: (form) => fetch('/api/lora/upload', { method: 'POST', body: form }).then(checked),
  updateUserLora: (filename, request) => fetch(`/api/lora/${encodeURIComponent(filename)}`, {
    method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(request)
  }).then(checked),
  updateUserLoraPreview: (filename, form) => fetch(`/api/lora/${encodeURIComponent(filename)}/preview`, { method: 'PUT', body: form }).then(checked),
  deleteUserLoraPreview: (filename) => fetch(`/api/lora/${encodeURIComponent(filename)}/preview`, { method: 'DELETE' }),
  deleteUserLora: (filename) => fetch(`/api/lora/${encodeURIComponent(filename)}`, { method: 'DELETE' })
}
