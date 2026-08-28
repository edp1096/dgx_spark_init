import test from 'node:test'
import assert from 'node:assert/strict'
import { get } from 'svelte/store'
import { ImageSequenceController, imageSequenceBlockedMessage } from './imageSequenceController.js'

test('storyboard owns independent scenes and ordering', () => {
  const controller = new ImageSequenceController()
  controller.setEntryMode('scenes')
  controller.updatePrompt(0, 'first')
  controller.updatePrompt(1, 'second')
  controller.addScene()
  controller.updatePrompt(2, 'third')
  controller.moveScene(2, -1)
  assert.deepEqual(get(controller.state).prompts, ['first', 'third', 'second'])
  controller.removeScene(1)
  assert.deepEqual(get(controller.state).prompts, ['first', 'second'])
  controller.destroy()
})

test('story and direct examples switch input mode', () => {
  const controller = new ImageSequenceController()
  controller.applyStoryExample()
  let state = get(controller.state)
  assert.equal(state.entryMode, 'story')
  assert.equal(state.sceneCount, 5)
  assert.match(state.storyIdea, /60대 남성 사진가/)
  controller.applySceneExample()
  state = get(controller.state)
  assert.equal(state.entryMode, 'scenes')
  assert.equal(state.prompts.length, 4)
  controller.destroy()
})

test('storyboard compatibility allows full T2I resolution and reports actual blockers', () => {
  const base = { mode: 'create', modules: {}, moduleReason: '', width: 2048, height: 2048 }
  assert.equal(imageSequenceBlockedMessage(base), '')
  assert.match(imageSequenceBlockedMessage({ ...base, modules: { depth: true } }), /자세·구도/)
  assert.match(imageSequenceBlockedMessage({ ...base, mode: 'edit' }), /새 이미지 생성/)
  assert.match(imageSequenceBlockedMessage({ ...base, checkpoint: 'moody-v7', hasReIDReference: true }), /공식 Krea/)
})

test('direct scene planning stores standalone prompts and titles', async () => {
  const controller = new ImageSequenceController()
  controller.setPrompts(['market clue', 'rooftop meeting'])
  const api = {
    planImageSequence: async ({ prompts }) => {
      assert.deepEqual(prompts, ['market clue', 'rooftop meeting'])
      return {
        shared_prompt: 'same detective and noir style',
        canonical_prompt_en: 'A detective with a short black bob and navy coat.',
        scenes: [
          { original_prompt: prompts[0], scene_title: '시장', enhanced_prompt: 'A detective finds a clue in a rainy market.' },
          { original_prompt: prompts[1], scene_title: '옥상', enhanced_prompt: 'The same detective meets a suspect on a rooftop.' }
        ]
      }
    }
  }
  await controller.plan(api)
  const state = get(controller.state)
  assert.deepEqual(state.sceneTitles, ['시장', '옥상'])
  assert.equal(state.enhancedPrompts[1], 'The same detective meets a suspect on a rooftop.')
  assert.equal(state.sharedPrompt, 'same detective and noir style')
  assert.equal(state.canonicalPrompt, 'A detective with a short black bob and navy coat.')
  controller.destroy()
})

test('editing the Korean continuity bible invalidates English prompts and sends it on replanning', async () => {
  const controller = new ImageSequenceController()
  controller.setPrompts(['시장 장면', '옥상 장면'])
  controller.setSharedPrompt('주황색 로봇의 둥근 얼굴과 파란 눈을 유지한다.')
  let received
  const api = {
    planImageSequence: async (payload) => {
      received = payload
      return {
        shared_prompt: payload.shared_prompt,
        canonical_prompt_en: 'A compact orange robot with a round faceplate and two blue eyes.',
        scenes: payload.prompts.map((prompt, index) => ({ original_prompt: prompt, scene_title: `장면 ${index + 1}`, enhanced_prompt: `Stable robot. Scene ${index + 1}.` }))
      }
    }
  }
  await controller.plan(api)
  const state = get(controller.state)
  assert.equal(received.shared_prompt, '주황색 로봇의 둥근 얼굴과 파란 눈을 유지한다.')
  assert.equal(state.enhancedPrompts.length, 2)
  assert.match(state.canonicalPrompt, /round faceplate/)
  controller.setSharedPrompt('주황색 로봇의 사각 얼굴과 초록 눈을 유지한다.')
  assert.equal(get(controller.state).enhancedPrompts.length, 0)
  controller.destroy()
})

test('story planning replaces the outline with reviewable scenes', async () => {
  const controller = new ImageSequenceController()
  controller.setStoryIdea('로봇이 주인을 찾는다')
  controller.setSceneCount(3)
  const api = {
    planImageSequence: async (payload) => {
      assert.deepEqual(payload, { outline: '로봇이 주인을 찾는다', scene_count: 3 })
      return {
        shared_prompt: '같은 로봇',
        scenes: [1, 2, 3].map((index) => ({ original_prompt: `장면 ${index} 설명`, scene_title: `장면 ${index}`, enhanced_prompt: `Scene ${index}` }))
      }
    }
  }
  await controller.plan(api)
  const state = get(controller.state)
  assert.equal(state.entryMode, 'scenes')
  assert.deepEqual(state.prompts, ['장면 1 설명', '장면 2 설명', '장면 3 설명'])
  assert.deepEqual(state.enhancedPrompts, ['Scene 1', 'Scene 2', 'Scene 3'])
  controller.destroy()
})

test('visual character analysis is locked into the sequence plan payload', async () => {
  const controller = new ImageSequenceController()
  controller.setPrompts(['시장에 들어간다', '옥상에 도착한다'])
  controller.addCharacter()
  controller.addCharacterResult(0, { id: 'image-job-123456', output_url: '/api/outputs/character.png' })
  const api = {
    describeSequenceCharacter: async (form) => {
      assert.equal(form.get('name'), '등장인물 1')
      assert.equal(form.get('reuse_images'), 'image-job-123456:output:0')
      assert.deepEqual(JSON.parse(form.get('locked_traits')), ['face', 'hair', 'body', 'outfit', 'mechanical'])
      return {
        name_ko: '로', name_en: 'Rho', description_ko: '주황색 배달 로봇',
        canonical_prompt_en: 'Rho is a compact orange delivery robot with a cream circular faceplate.',
        observations: { mechanical_geometry: '둥근 얼굴판' }
      }
    },
    planImageSequence: async (payload) => {
      assert.deepEqual(payload.locked_characters, [{
        id: 'character_1', name_ko: '등장인물 1', name_en: 'Rho', description_ko: '주황색 배달 로봇',
        prompt_en: 'Rho is a compact orange delivery robot with a cream circular faceplate.'
      }])
      return {
        shared_prompt: '주황색 배달 로봇', canonical_prompt_en: payload.locked_characters[0].prompt_en,
        scenes: payload.prompts.map((prompt, index) => ({ original_prompt: prompt, scene_title: `장면 ${index + 1}`, enhanced_prompt: `${payload.locked_characters[0].prompt_en} Scene ${index + 1}.` }))
      }
    }
  }
  await controller.analyzeCharacter(0, api)
  let state = get(controller.state)
  assert.equal(state.characters[0].nameEN, 'Rho')
  assert.equal(state.characters[0].observations.mechanical_geometry, '둥근 얼굴판')
  await controller.plan(api)
  state = get(controller.state)
  assert.match(state.enhancedPrompts[1], /compact orange delivery robot/)
  controller.destroy()
})

test('character preparation keeps an explicit ReID anchor and invalidates analysis when traits change', () => {
  const controller = new ImageSequenceController()
  controller.addCharacter()
  controller.addCharacterResult(0, { id: 'first-image-job', output_url: '/api/outputs/first.png' })
  controller.addCharacterResult(0, { id: 'second-image-job', output_url: '/api/outputs/second.png' })
  controller.setCharacterReIDReference(0, 1)
  let state = get(controller.state)
  assert.equal(state.characters[0].reidReferenceIndex, 1)
  assert.match(controller.reidReference().url, /second\.png/)
  controller.setCharacterCanonicalPrompt(0, 'approved identity')
  controller.toggleCharacterTrait(0, 'accessories')
  state = get(controller.state)
  assert.equal(state.characters[0].lockedTraits.accessories, true)
  assert.equal(state.characters[0].canonicalPromptEN, '')
  controller.removeCharacterReference(0, 0)
  assert.equal(get(controller.state).characters[0].reidReferenceIndex, 0)
  controller.destroy()
})
