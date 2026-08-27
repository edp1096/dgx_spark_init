import test from 'node:test'
import assert from 'node:assert/strict'
import { ImageModalController } from './imageModalController.js'
import { VideoModalController } from './videoModalController.js'

test('image picker routing uses the visible target instead of stale controller state', () => {
  const selected = []
  const controller = new ImageModalController({
    getImageJobs: () => [],
    getSequencePrompts: () => [],
    setSequencePrompts: () => {},
    setSequenceBase: () => {},
    resetSequence: () => {},
    addKreaRefObjects: () => {},
    addIdentityReferenceObjects: () => {},
    addKreaRefs: () => {},
    addIdentityReferences: () => {},
    setKreaImage: (target, image) => selected.push({ target, image }),
    setNK2EPreprocessed: () => {},
    setError: () => {}
  })
  const identityUI = { primary: '편집할 원본', secondary: '보조 참조' }
  assert.equal(controller.presetTitle(identityUI, 'depth'), '자세·구도 프리셋 선택')
  controller.selectRecent({ id: 'abc123456', output_url: '/saved.png' }, 'depth')
  assert.equal(selected[0].target, 'depth')
  assert.equal(selected[0].image.ref, 'abc123456:output:0')
})

test('video picker routing accepts its current visible keyframe target', () => {
  const selected = []
  const controller = new VideoModalController({
    getJobs: () => [],
    getRecognitionJobs: () => [],
    getVideoKeyframes: () => [{ id: 7 }],
    setVideoConditionImage: (target, image) => selected.push({ target, image }),
    setError: () => {},
    regenerateSubtitle: async () => {},
    submitUpscale: async () => {},
    sendVideoToRecognition: () => {},
    loadVideoSettings: () => {},
    sendAudioToVideo: () => {}
  })
  controller.selectRecentImage({ id: 'video-image', output_url: '/image.png' }, 'keyframe:7')
  assert.equal(selected[0].target, 'keyframe:7')
  assert.equal(controller.conditionTitle('keyframe:7'), '키프레임 1 이미지 선택')
})
