const visualContextCues = ['프롬프트', 'prompt', '시작', '마지막', '키프레임', '장면', '영상', '이어', '전환', '움직']

export function needsVideoVisualContext(message) {
  const text = String(message || '').toLowerCase()
  return visualContextCues.some((cue) => text.includes(cue))
}

export function loadBrowserImage(src) {
  return new Promise((resolve, reject) => {
    const image = new Image()
    image.onload = () => resolve(image)
    image.onerror = () => reject(new Error('선택한 영상 이미지를 읽지 못했습니다.'))
    image.src = src
  })
}

export async function createVideoVisualContext({ message, conditions, imageURL, loadImage = loadBrowserImage }) {
  if (!needsVideoVisualContext(message) || !conditions.length) return null
  const loaded = []
  for (const condition of conditions) {
    try {
      loaded.push({ ...condition, bitmap: await loadImage(imageURL(condition.image)) })
    } catch (_) {}
  }
  if (!loaded.length) return null

  const columns = loaded.length === 1 ? 1 : 2
  const cellWidth = loaded.length === 1 ? 640 : 420
  const cellHeight = loaded.length === 1 ? 480 : 315
  const rows = Math.ceil(loaded.length / columns)
  const canvas = document.createElement('canvas')
  canvas.width = cellWidth * columns
  canvas.height = cellHeight * rows
  const context = canvas.getContext('2d')
  context.fillStyle = '#0d1115'
  context.fillRect(0, 0, canvas.width, canvas.height)
  loaded.forEach((item, index) => {
    const left = index % columns * cellWidth
    const top = Math.floor(index / columns) * cellHeight
    const padding = 6
    const sourceWidth = item.bitmap.naturalWidth || item.bitmap.width
    const sourceHeight = item.bitmap.naturalHeight || item.bitmap.height
    const scale = Math.min((cellWidth - padding * 2) / sourceWidth, (cellHeight - padding * 2) / sourceHeight)
    const width = sourceWidth * scale
    const height = sourceHeight * scale
    context.drawImage(item.bitmap, left + (cellWidth - width) / 2, top + (cellHeight - height) / 2, width, height)
    const label = `${item.label} · ${item.detail}`
    context.font = '700 18px sans-serif'
    const labelWidth = context.measureText(label).width + 22
    context.fillStyle = 'rgba(8,12,15,.86)'
    context.fillRect(left + 12, top + 12, labelWidth, 34)
    context.fillStyle = '#d9f5b7'
    context.fillText(label, left + 23, top + 35)
  })
  return {
    kind: 'video_conditioning',
    image_url: canvas.toDataURL('image/jpeg', 0.84),
    labels: loaded.map((item) => `${item.label} ${item.detail}`),
  }
}
