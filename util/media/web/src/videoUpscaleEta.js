// Deterministic SeedVR2 video-upscale estimator for the DGX Spark GB10.
//
// These coefficients describe the actual SeedVR2 execution phases instead of
// copying a previous job's wall time. They are calibrated conservatively from
// a clean 121-frame, 768x512 -> 1632x1088 run with the 3B FP8 model and tiled
// VAE. Queueing, retries and application downtime are deliberately absent.

const VAE_TILE_SIZE = 1024
const VAE_TILE_OVERLAP = 128

function positiveNumber(value, fallback) {
  const number = Number(value)
  return Number.isFinite(number) && number > 0 ? number : fallback
}

function tileAxisCount(length) {
  const size = Math.max(1, Number(length) || 1)
  if (size <= VAE_TILE_SIZE) return 1
  return Math.ceil((size - VAE_TILE_SIZE) / (VAE_TILE_SIZE - VAE_TILE_OVERLAP)) + 1
}

export function upscaleFrameWork(params = {}) {
  const fps = positiveNumber(params.fps, 24)
  const explicitFrames = Math.round(Number(params.num_frames) || 0)
  const sourceFrames = Math.max(1, explicitFrames > 0
    ? explicitFrames
    : Math.round(Math.max(0.04, positiveNumber(params.duration, 5)) * fps))
  const batch = Math.max(1, Math.round(positiveNumber(params.batch_size, 5)))
  let overlap = Math.max(0, Math.round(Number(params.temporal_overlap) || 0))
  if (overlap >= batch) overlap = 0
  const step = overlap > 0 ? batch - overlap : batch
  let processedFrames = 0
  let batches = 0
  for (let index = 0; index < sourceFrames; index += step) {
    const count = Math.min(batch, sourceFrames - index)
    if (index > 0 && count <= overlap) break
    processedFrames += count
    batches += 1
  }
  return { sourceFrames, processedFrames, batches, batch, overlap }
}

export function videoUpscaleEstimate(params = {}) {
  const scale = Math.max(1, positiveNumber(params.upscale_scale, 2))
  const sourceWidth = positiveNumber(params.source_width, 768)
  const sourceHeight = positiveNumber(params.source_height, 512)
  const width = positiveNumber(params.width, sourceWidth * scale)
  const height = positiveNumber(params.height, sourceHeight * scale)
  const outputMegapixels = Math.max(.1, width * height / 1_000_000)
  const frames = upscaleFrameWork(params)
  const processedMegapixelFrames = outputMegapixels * frames.processedFrames
  const sourceMegapixelFrames = outputMegapixels * frames.sourceFrames
  const spatialTiles = tileAxisCount(width) * tileAxisCount(height)
  const processedTileFrames = spatialTiles * frames.processedFrames

  // GB10 phase model. Per-batch costs preserve the real trade-off between a
  // small temporal batch and duplicate overlap work; VAE costs also account
  // for the 1024px/128px-overlap spatial tiling used by this service.
  const encodeSeconds = 5 + .35 * processedMegapixelFrames + .20 * processedTileFrames
  const ditSeconds = 40 + .80 * processedMegapixelFrames + 2 * frames.batches
  const decodeSeconds = 15 + .85 * processedMegapixelFrames + .435 * processedTileFrames
  const postprocessSeconds = 5 + .06 * sourceMegapixelFrames
  const transferSeconds = 8 + .10 * sourceMegapixelFrames
  const totalSeconds = encodeSeconds + ditSeconds + decodeSeconds + postprocessSeconds + transferSeconds

  return {
    totalSeconds: Math.max(30, totalSeconds),
    encodeSeconds,
    ditSeconds,
    decodeSeconds,
    postprocessSeconds,
    transferSeconds,
    spatialTiles,
    outputMegapixels,
    ...frames
  }
}

export function videoUpscaleEstimateSeconds(params = {}) {
  return videoUpscaleEstimate(params).totalSeconds
}
