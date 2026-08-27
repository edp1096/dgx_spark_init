function defaultCreateObjectURL(file) {
  return URL.createObjectURL(file)
}

function defaultRevokeObjectURL(url) {
  URL.revokeObjectURL(url)
}

export function isImageFile(file) {
  return Boolean(file?.type?.startsWith('image/'))
}

export function mediaInputPreview(input) {
  return input?.preview || input?.url || ''
}

export function normalizeMediaInput(input, createObjectURL = defaultCreateObjectURL) {
  if (!input) return null
  if (input.server) return { ...input, preview: mediaInputPreview(input) }
  if (input.file && input.preview) return input
  const file = input.file || input
  return {
    ...((typeof input === 'object') ? input : {}),
    file,
    name: input.name || file.name || 'image',
    preview: input.preview || createObjectURL(file),
    server: false
  }
}

export function normalizeImageFiles(files, createObjectURL = defaultCreateObjectURL) {
  return [...(files || [])]
    .filter(isImageFile)
    .map((file) => normalizeMediaInput(file, createObjectURL))
}

export function releaseMediaInput(input, revokeObjectURL = defaultRevokeObjectURL) {
  const preview = mediaInputPreview(input)
  if (preview.startsWith('blob:')) revokeObjectURL(preview)
}

export function appendMediaInputs(current, incoming, limit, options = {}) {
  const createObjectURL = options.createObjectURL || defaultCreateObjectURL
  const revokeObjectURL = options.revokeObjectURL || defaultRevokeObjectURL
  const normalized = [...(incoming || [])].filter(Boolean).map((input) => normalizeMediaInput(input, createObjectURL))
  const combined = [...(current || []), ...normalized]
  combined.slice(limit).forEach((input) => releaseMediaInput(input, revokeObjectURL))
  return combined.slice(0, limit)
}

export function clearMediaInputs(current, revokeObjectURL = defaultRevokeObjectURL) {
  ;[...(current || [])].forEach((input) => releaseMediaInput(input, revokeObjectURL))
  return []
}

export function removeMediaInput(current, index, revokeObjectURL = defaultRevokeObjectURL) {
  const list = [...(current || [])]
  releaseMediaInput(list[index], revokeObjectURL)
  return list.filter((_, itemIndex) => itemIndex !== index)
}

export function replaceMediaInput(current, next, options = {}) {
  const createObjectURL = options.createObjectURL || defaultCreateObjectURL
  const revokeObjectURL = options.revokeObjectURL || defaultRevokeObjectURL
  if (current && current !== next) releaseMediaInput(current, revokeObjectURL)
  return normalizeMediaInput(next, createObjectURL)
}
