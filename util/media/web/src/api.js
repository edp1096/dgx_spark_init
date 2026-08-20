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
  engines: () => fetch('/api/engines').then(checked),
  deleteJob: (id) => fetch(`/api/jobs/${encodeURIComponent(id)}`, { method: 'DELETE' }).then(async (response) => {
    if (!response.ok) throw new Error((await response.text()) || `HTTP ${response.status}`)
  }),
  cancelJob: (id) => fetch(`/api/jobs/${encodeURIComponent(id)}/cancel`, { method: 'POST' }).then(checked),
  deleteFinishedJobs: () => fetch('/api/jobs', { method: 'DELETE' }).then(checked),
  image: (form) => fetch('/api/jobs/image', { method: 'POST', body: form }).then(checked),
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
  enhancePrompt: (form) => fetch('/api/prompts/enhance', { method: 'POST', body: form }).then(checked)
}
