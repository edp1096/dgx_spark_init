import test from 'node:test'
import assert from 'node:assert/strict'
import { MediaTransferController } from './mediaTransferController.js'

function harness() {
  const state = {
    tab: '', form: { prompt: '', width: 768, height: 512, fps: 24, seed: -1, image_strength: 1 },
    duration: 5, keyframes: [], clips: [], recognitionForm: { source: 'url', url: 'https://example.com' },
    recognitionJob: null, recognitionFile: {}, error: ''
  }
  let keyframeID = 1
  let clipID = 1
  const actions = {
    switchTab: (tab) => state.tab = tab,
    setVideoConditionImage: (target, image) => state[target] = image,
    getVideoKeyframes: () => state.keyframes,
    setVideoKeyframes: (value) => state.keyframes = value,
    getVideoForm: () => state.form,
    setVideoForm: (value) => state.form = value,
    getVideoDuration: () => state.duration,
    setVideoDuration: (value) => state.duration = value,
    nearestAvailableVideoKeyframeFrame: (frame) => frame,
    allocateVideoKeyframeID: () => keyframeID++,
    normalizeVideoImage: (file) => ({ file, normalized: true }),
    setRecognitionSourceVideoJob: (job) => state.recognitionJob = job,
    setRecognitionFile: (file) => state.recognitionFile = file,
    clearRecognitionFileInput: () => state.fileInputCleared = true,
    getRecognitionForm: () => state.recognitionForm,
    setRecognitionForm: (value) => state.recognitionForm = value,
    resetRecognitionOptions: () => state.recognitionReset = true,
    clearVideoConditioning: () => state.conditioningCleared = true,
    getConfig: () => ({ video: { default_width: 1280, default_height: 704, default_fps: 24, default_frames: 121 } }),
    getSpeechJobs: () => state.speechJobs || [],
    allocateVideoAudioClipID: () => clipID++,
    getVideoAudioClips: () => state.clips,
    setVideoAudioClips: (value) => state.clips = value,
    resetVideoEnhancement: () => state.enhancementReset = true,
    setError: (value) => state.error = value,
    snapVideoDuration: (value) => value,
    normalizeVideoTiming: () => state.timingNormalized = true
  }
  return { state, actions }
}

test('video frames and completed videos route to their destination tabs', async () => {
  const { state, actions } = harness()
  const controller = new MediaTransferController(actions)
  await controller.usePickedVideoFrame({ name: 'frame.png' }, 'keyframe', 2.5, 5)
  assert.equal(state.keyframes[0].time, 2.5)
  assert.equal(state.tab, 'video')
  controller.sendVideoToRecognition({ id: 'video-1', output_url: '/video.mp4', status: 'completed' })
  assert.equal(state.recognitionJob.id, 'video-1')
  assert.equal(state.recognitionForm.source, 'video_job')
  assert.equal(state.tab, 'recognition')
})

test('audio transfer packs clips and grows video duration', async () => {
  const { state, actions } = harness()
  const controller = new MediaTransferController(actions, { probeAudioDuration: async () => 3.25 })
  await controller.sendAudioToVideo({ id: 'audio-1', output_url: '/audio.wav', status: 'completed' })
  assert.equal(state.clips[0].duration, 3.25)
  assert.equal(state.duration, 5)
  assert.equal(state.timingNormalized, true)
  assert.equal(state.tab, 'video')
})
