async function checked(response) {
  if (!response.ok) throw new Error((await response.text()) || `HTTP ${response.status}`)
  return response.json()
}

export const api = {
  config: () => fetch('/api/config').then(checked),
  saveConfig: (config) => fetch('/api/config', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config)
  }).then(checked),
  jobs: () => fetch('/api/jobs').then(checked),
  imageInputs: (id) => fetch(`/api/jobs/${encodeURIComponent(id)}/inputs`).then(checked),
  imageEXIF: (id) => fetch(`/api/jobs/${encodeURIComponent(id)}/exif`).then(checked),
  engines: () => fetch('/api/engines').then(checked),
  system: () => fetch('/api/system').then(checked),
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
  detailEnhanceImage: (id, options = { strength: 1, seed: -1, vae: 'wan' }) => fetch(`/api/jobs/${encodeURIComponent(id)}/detail-enhance`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(options)
  }).then(checked),
  speech: (form) => fetch('/api/jobs/speech', { method: 'POST', body: form }).then(checked),
  recognition: (form) => fetch('/api/jobs/recognition', { method: 'POST', body: form }).then(checked),
  mediaOptions: (url) => fetch('/api/media/options', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({ url })
  }).then(checked),
  storage: () => fetch('/api/storage').then(checked),
  cleanupTemporaryStorage: () => fetch('/api/storage/temp', { method: 'DELETE' }).then(checked),
  video: (form) => fetch('/api/jobs/video', { method: 'POST', body: form }).then(checked),
  enhancePrompt: (form) => fetch('/api/prompts/enhance', { method: 'POST', body: form }).then(checked),
  loraDatasets: () => fetch('/api/lora/datasets').then(checked),
  createLoraDataset: (name) => fetch('/api/lora/datasets', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name })
  }).then(checked),
  deleteLoraDataset: (name) => fetch(`/api/lora/datasets/${encodeURIComponent(name)}`, { method: 'DELETE' }),
  uploadLoraImages: (name, form) => fetch(`/api/lora/datasets/${encodeURIComponent(name)}/images`, { method: 'POST', body: form }).then(checked),
  saveLoraCaption: (dataset, filename, caption) => fetch(`/api/lora/datasets/${encodeURIComponent(dataset)}/images/${encodeURIComponent(filename)}/caption`, {
    method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ caption })
  }).then(checked),
  deleteLoraImage: (dataset, filename) => fetch(`/api/lora/datasets/${encodeURIComponent(dataset)}/images/${encodeURIComponent(filename)}`, { method: 'DELETE' }),
  loraJobs: () => fetch('/api/lora/jobs').then(checked),
  startLoraTraining: (request) => fetch('/api/lora/jobs', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(request)
  }).then(checked),
  cancelLoraTraining: (id) => fetch(`/api/lora/jobs/${encodeURIComponent(id)}/cancel`, { method: 'POST' }).then(checked),
  userLoras: () => fetch('/api/lora/loras').then(checked),
  deleteUserLora: (filename) => fetch(`/api/lora/loras/${encodeURIComponent(filename)}`, { method: 'DELETE' })
}
