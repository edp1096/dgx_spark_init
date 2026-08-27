import test from 'node:test'
import assert from 'node:assert/strict'
import { buildImageGenerationForm, buildVideoGenerationForm } from './generationRequests.js'

test('image requests preserve server references and enabled module settings', () => {
  const form = buildImageGenerationForm({
    imageForm: { prompt: 'visible prompt', width: 1024, height: 1024, mode: 'create' },
    prompt: 'effective prompt', originalPrompt: 'visible prompt', parentJobID: 'parent-1', sequence: null,
    modules: { identity: true, depth: false, style: true, userLora: false, vision: false, styleReference: false, nk2e: false, anypaint: false },
    options: { steps: 10, checkpoint: 'official', filter_mode: 'balanced', filter_strength: 1, prompt_enhancer: false, prompt_enhancer_strength: 1, prompt_text_scale: 1.75, sampling_preset: 'default', identity_strength: 1, ref_boost: 4, source_ref_boost: 1, grounding_px: 768, strict_mask_grow: 0, strict_mask_feather: 0, vae_mode: 'default', identity_fit_mode: 'fit', identity_model: 'convrot', identity_encoder: 'heretic' },
    identity: { preset: 'tryon', hasUserPrompt: false, preserveItems: ['identity'], preserveCustom: '', image: { server: true, ref: 'job:image:0' }, references: [{ server: true, ref: 'job:clothes:0' }], mask: null, strictMask: null },
    depth: { image: null }, styles: [{ name: 'retroanime', strength: 1 }], userLoras: [], visionImages: [], styleReferenceImages: [], nk2e: { image: null, preprocessed: false }, anypaint: { image: null, mask: null }, references: []
  })
  assert.equal(form.get('prompt'), 'effective prompt')
  assert.equal(form.get('original_prompt'), 'visible prompt')
  assert.equal(form.get('reuse_identity_image'), 'job:image:0')
  assert.equal(form.get('reuse_identity_reference'), 'job:clothes:0')
  assert.equal(form.get('identity_auto_prompt'), 'true')
  assert.equal(form.get('identity_user_prompt'), 'false')
  assert.deepEqual(JSON.parse(form.get('styles')), [{ name: 'retroanime', strength: 1 }])
})

test('video requests keep audio timing and only populated keyframes', () => {
  const form = buildVideoGenerationForm({
    videoForm: { prompt: 'raw', width: 768, height: 512, fps: 24 }, prompt: 'enhanced', originalPrompt: 'raw', numFrames: 121,
    startImage: { server: true, ref: 'image:start:0' }, endImage: null, endStrength: 0.8,
    audioClips: [{ job: { id: 'audio-1' }, start: 1.25, duration: 2.5 }],
    keyframes: [{ image: null, time: 1, strength: 1 }, { image: { server: true, ref: 'image:key:0' }, time: 2.5, strength: 0.7 }]
  })
  assert.equal(form.get('prompt'), 'enhanced')
  assert.equal(form.get('reuse_start_image'), 'image:start:0')
  assert.equal(form.get('audio_count'), '1')
  assert.equal(form.get('audio_start_0'), '1.25')
  assert.equal(form.get('keyframe_count'), '1')
  assert.equal(form.get('reuse_keyframe_image_0'), 'image:key:0')
})
