import { writable } from 'svelte/store'
import { appendMediaInputs, clearMediaInputs, mediaInputPreview, normalizeImageFiles, removeMediaInput } from './mediaInputs.js'

const storyExample = {
  idea: '비 오는 서울에서 오래된 사진의 장소를 찾아가는 60대 남성 사진가의 하루. 같은 얼굴과 백발, 낡은 갈색 가죽 재킷을 유지하고 장면마다 장소와 행동만 바꾼다.',
  count: 5
}

const sceneExample = [
  '같은 60대 남성 사진가가 비 내리는 서울 골목에서 오래된 필름 카메라를 점검하는 장면',
  '같은 남성이 붐비는 재래시장에서 낡은 사진과 건물들을 비교하는 장면',
  '같은 남성이 해질 무렵 한강 다리 위에서 카메라를 들어 사진을 찍는 장면',
  '같은 남성이 밤의 옥상에서 찾던 장소를 발견하고 조용히 미소 짓는 장면'
]

export const sequenceCharacterTraitChoices = [
  ['face', '얼굴'], ['hair', '머리'], ['body', '체형'], ['outfit', '의상'],
  ['accessories', '액세서리'], ['mechanical', '기계 구조']
]

const defaultCharacterTraits = () => ({ face: true, hair: true, body: true, outfit: true, accessories: false, mechanical: true })

function newCharacter(counter, overrides = {}) {
  return {
    id: `character_${counter}`, name: `등장인물 ${counter}`, nameKO: '', nameEN: '', references: [],
    reidReferenceIndex: 0, lockedTraits: defaultCharacterTraits(),
    quadViewCandidate: null, quadViewGenerating: false, quadViewError: '', quadViewStartedAt: 0, quadViewProgress: null,
    turntableFrames: [], turntableSelection: [],
    descriptionKO: '', canonicalPromptEN: '', observations: {}, analyzing: false, error: '',
    ...overrides
  }
}

function initialState() {
  return {
    entryMode: 'story', storyIdea: '', sceneCount: 5,
    prompts: ['', ''], enhancedPrompts: [], sceneTitles: [], sharedPrompt: '', canonicalPrompt: '', sharedPromptEdited: false,
    characters: [], characterCounter: 0,
    planSignature: '', planning: false, planError: ''
  }
}

function lockedCharacters(state) {
  return state.characters
    .filter((character) => character.canonicalPromptEN.trim())
    .map((character) => ({
      id: character.id,
      name_ko: character.name.trim() || character.nameKO.trim(),
      name_en: character.nameEN.trim(),
      description_ko: character.descriptionKO.trim(),
      prompt_en: character.canonicalPromptEN.trim()
    }))
}

function base64ImageFile(frame, characterName) {
  const binary = atob(String(frame.data || ''))
  const bytes = new Uint8Array(binary.length)
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index)
  const extension = frame.mime_type === 'image/png' ? 'png' : 'jpg'
  return new File([bytes], `${characterName || 'character'}-${frame.direction}.${extension}`, { type: frame.mime_type || 'image/jpeg' })
}

export class ImageSequenceController {
  constructor() {
    this.current = initialState()
    this.state = writable(this.current)
    this.unsubscribe = this.state.subscribe((value) => this.current = value)
  }

  setState(patch) { this.state.update((value) => ({ ...value, ...patch })) }

  signature() {
    return JSON.stringify({
      entryMode: this.current.entryMode,
      storyIdea: this.current.storyIdea.trim(),
      sceneCount: this.current.sceneCount,
      prompts: this.current.prompts.map((value) => value.trim()),
      sharedPrompt: this.current.sharedPromptEdited ? this.current.sharedPrompt.trim() : '',
      lockedCharacters: lockedCharacters(this.current)
    })
  }

  invalidatePlan(patch = {}, keepEditedShared = true) {
    const keepShared = keepEditedShared && this.current.sharedPromptEdited
    this.setState({
      ...patch, enhancedPrompts: [], sceneTitles: [], canonicalPrompt: '',
      sharedPrompt: keepShared ? this.current.sharedPrompt : '', sharedPromptEdited: keepShared,
      planSignature: '', planError: ''
    })
  }

  reset(prompts = ['', '']) {
	this.current.characters.forEach((character) => { clearMediaInputs(character.references); clearMediaInputs(character.turntableFrames || []); clearMediaInputs(character.quadViewCandidate ? [character.quadViewCandidate] : []) })
    this.setState({
      ...initialState(),
      entryMode: prompts.some((value) => value.trim()) ? 'scenes' : 'story',
      prompts: [...prompts]
    })
  }

  setEntryMode(entryMode) {
    if (entryMode !== 'story' && entryMode !== 'scenes') return
    this.invalidatePlan({ entryMode }, false)
  }

  setStoryIdea(storyIdea) { this.invalidatePlan({ storyIdea }, false) }

  setSceneCount(sceneCount) {
    this.invalidatePlan({ sceneCount: Math.max(2, Math.min(12, Number(sceneCount) || 5)) }, false)
  }

  setPrompts(prompts) { this.invalidatePlan({ entryMode: 'scenes', prompts: [...prompts] }, false) }

  setSharedPrompt(sharedPrompt) {
    this.setState({
      sharedPrompt, sharedPromptEdited: true, enhancedPrompts: [], canonicalPrompt: '',
      planSignature: '', planError: ''
    })
  }

  addCharacter() {
    if (this.current.characters.length >= 4) return
    const counter = this.current.characterCounter + 1
    this.invalidatePlan({
      characterCounter: counter,
      characters: [...this.current.characters, newCharacter(counter)]
    })
  }

  addCharacterExample(index, file, example) {
    const character = this.current.characters[index]
    if (!character || character.references.length >= 6 || !file) return
    const reference = normalizeImageFiles([file])[0]
    if (!reference) return
    const mechanical = example?.kind === 'toy'
    const patch = {
      name: character.name.startsWith('등장인물 ') ? (example?.name || character.name) : character.name,
      references: appendMediaInputs(character.references, [reference], 6),
      lockedTraits: example?.lockedTraits || { face: true, hair: !mechanical, body: true, outfit: !mechanical, accessories: false, mechanical },
      canonicalPromptEN: String(example?.canonicalPromptEN || '').trim(),
      descriptionKO: String(example?.descriptionKO || '').trim(), observations: {}, error: ''
    }
    const characters = this.current.characters.map((item, itemIndex) => itemIndex === index ? { ...item, ...patch } : item)
    const related = { characters }
    if (this.current.entryMode === 'story' && !this.current.storyIdea.trim() && example?.story) {
      related.storyIdea = example.story
      related.sceneCount = example.scenes?.length || 4
    } else if (this.current.entryMode === 'scenes' && this.current.prompts.every((prompt) => !prompt.trim()) && example?.scenes?.length) {
      related.prompts = [...example.scenes]
    }
    this.invalidatePlan(related)
  }

  removeCharacter(index) {
    const removed = this.current.characters[index]
    if (!removed) return
    clearMediaInputs(removed.references)
    clearMediaInputs(removed.turntableFrames || [])
    clearMediaInputs(removed.quadViewCandidate ? [removed.quadViewCandidate] : [])
    this.invalidatePlan({ characters: this.current.characters.filter((_, itemIndex) => itemIndex !== index) })
  }

  updateCharacter(index, patch, invalidate = true) {
    const characters = this.current.characters.map((character, itemIndex) => itemIndex === index ? { ...character, ...patch } : character)
    if (invalidate) this.invalidatePlan({ characters })
    else this.setState({ characters })
  }

  setCharacterName(index, name) { this.updateCharacter(index, { name }) }
  setCharacterDescription(index, descriptionKO) { this.updateCharacter(index, { descriptionKO }) }
  setCharacterCanonicalPrompt(index, canonicalPromptEN) { this.updateCharacter(index, { canonicalPromptEN }) }

  setCharacterReIDReference(index, referenceIndex) {
    const character = this.current.characters[index]
    if (!character || referenceIndex < 0 || referenceIndex >= character.references.length) return
    this.updateCharacter(index, { reidReferenceIndex: referenceIndex })
  }

  toggleCharacterTrait(index, trait) {
    if (!sequenceCharacterTraitChoices.some(([value]) => value === trait)) return
    const character = this.current.characters[index]
    if (!character) return
    this.updateCharacter(index, {
      lockedTraits: { ...defaultCharacterTraits(), ...character.lockedTraits, [trait]: !character.lockedTraits?.[trait] },
      canonicalPromptEN: '', descriptionKO: '', observations: {}, error: ''
    })
  }

  addCharacterFiles(index, files) {
    const character = this.current.characters[index]
    if (!character) return
    const references = appendMediaInputs(character.references, normalizeImageFiles(files), 6)
    this.updateCharacter(index, { references, reidReferenceIndex: Math.min(character.reidReferenceIndex || 0, Math.max(0, references.length - 1)), canonicalPromptEN: '', descriptionKO: '', observations: {}, error: '' })
  }

  addCharacterResult(index, job) {
    const character = this.current.characters[index]
    if (!character?.references || !job?.id || !job?.output_url) return
    const reference = {
      server: true, ref: `${job.id}:output:0`, url: job.output_url, preview: job.output_url,
      name: `결과 #${job.id.slice(0, 8)}`
    }
    const references = appendMediaInputs(character.references, [reference], 6)
    this.updateCharacter(index, { references, reidReferenceIndex: Math.min(character.reidReferenceIndex || 0, Math.max(0, references.length - 1)), canonicalPromptEN: '', descriptionKO: '', observations: {}, error: '' })
  }

  removeCharacterReference(index, referenceIndex) {
    const character = this.current.characters[index]
    if (!character) return
    const references = removeMediaInput(character.references, referenceIndex)
    let reidReferenceIndex = character.reidReferenceIndex || 0
    if (referenceIndex < reidReferenceIndex) reidReferenceIndex -= 1
    else if (referenceIndex === reidReferenceIndex) reidReferenceIndex = 0
    this.updateCharacter(index, {
      references, reidReferenceIndex: Math.min(reidReferenceIndex, Math.max(0, references.length - 1)),
      canonicalPromptEN: '', descriptionKO: '', observations: {}, error: ''
    })
  }

  characterPreview(reference) { return mediaInputPreview(reference) }

  async generateCharacterSheet(index, api) {
    const character = this.current.characters[index]
    const anchor = character?.references?.[Math.min(character.reidReferenceIndex || 0, Math.max(0, character.references.length - 1))]
    if (!anchor) throw new Error('먼저 ReID 대표 이미지를 선택하세요.')
    const operationID = `character-turntable-${Date.now()}-${Math.random().toString(16).slice(2)}`
    let polling = true
    let progressTimer = 0
    const pollProgress = async () => {
      if (!polling) return
      try {
        const status = await api.sequenceCharacterTurntableStatus(operationID)
        const operation = status?.operation
        if (operation?.operation_id === operationID) {
          const currentProgress = Number(this.current.characters[index]?.quadViewProgress?.progress || 0)
          const nextProgress = Number(operation.progress || 0)
          if (nextProgress >= currentProgress) this.updateCharacter(index, { quadViewProgress: operation }, false)
        }
      } catch (_) {
        // The generation request may reach the engine just after the first poll.
      }
      if (polling) progressTimer = setTimeout(pollProgress, 400)
    }
    this.updateCharacter(index, {
      quadViewGenerating: true, quadViewError: '', quadViewStartedAt: Date.now(),
      quadViewProgress: { operation_id: operationID, phase: 'preparing', detail: '참조 이미지 전달·엔진 대기', progress: 0.03 }
    }, false)
    try {
      const form = new FormData()
      form.append('operation_id', operationID)
      if (anchor.server) form.append('reuse_image', anchor.ref)
      else form.append('image', anchor.file)
      pollProgress()
      const result = await api.createSequenceCharacterTurntable(form)
      const candidates = normalizeImageFiles(result.frames.map((frame) => base64ImageFile(frame, character.name)))
        .map((candidate, frameIndex) => ({ ...candidate, direction: result.frames[frameIndex].direction, frameIndex: result.frames[frameIndex].frame_index }))
      clearMediaInputs(character.quadViewCandidate ? [character.quadViewCandidate] : [])
      clearMediaInputs(character.turntableFrames || [])
      this.updateCharacter(index, {
        quadViewCandidate: null, turntableFrames: candidates, turntableSelection: [1, 2, 4, 6, 7],
        quadViewGenerating: false, quadViewError: '', quadViewProgress: null
      }, false)
      return candidates
    } catch (error) {
      this.updateCharacter(index, { quadViewGenerating: false, quadViewError: error.message || String(error), quadViewProgress: null }, false)
      throw error
    } finally {
      polling = false
      clearTimeout(progressTimer)
    }
  }

  approveCharacterSheet(index) {
    const character = this.current.characters[index]
    if (!character?.turntableFrames?.length) return
    const available = Math.max(0, 6 - character.references.length)
    const chosen = (character.turntableSelection || []).slice(0, available).map((frameIndex) => character.turntableFrames[frameIndex]).filter(Boolean)
    const references = appendMediaInputs(character.references, chosen, 6)
    const approved = new Set(chosen)
    clearMediaInputs(character.turntableFrames.filter((frame) => !approved.has(frame)))
    this.updateCharacter(index, {
      references, turntableFrames: [], turntableSelection: [], quadViewCandidate: null,
      canonicalPromptEN: '', descriptionKO: '', observations: {}, quadViewError: '', error: ''
    })
  }

  discardCharacterSheet(index) {
    const character = this.current.characters[index]
    if (!character?.turntableFrames?.length && !character?.quadViewCandidate) return
    clearMediaInputs(character.turntableFrames || [])
    clearMediaInputs([character.quadViewCandidate])
    this.updateCharacter(index, { turntableFrames: [], turntableSelection: [], quadViewCandidate: null, quadViewError: '' }, false)
  }

  toggleCharacterTurntableFrame(index, frameIndex) {
    const character = this.current.characters[index]
    if (!character?.turntableFrames?.[frameIndex]) return
    const selected = new Set(character.turntableSelection || [])
    if (selected.has(frameIndex)) selected.delete(frameIndex)
    else if (selected.size < Math.max(0, 6 - character.references.length)) selected.add(frameIndex)
    this.updateCharacter(index, { turntableSelection: [...selected].sort((a, b) => a - b) }, false)
  }

  async analyzeCharacter(index, api) {
    const character = this.current.characters[index]
    if (!character?.references?.length) throw new Error('캐릭터 시트나 인물 이미지를 한 장 이상 선택하세요.')
    this.updateCharacter(index, { analyzing: true, error: '' }, false)
    try {
      const form = new FormData()
      form.append('name', character.name.trim())
      form.append('locked_traits', JSON.stringify(Object.entries({ ...defaultCharacterTraits(), ...character.lockedTraits }).filter(([, enabled]) => enabled).map(([trait]) => trait)))
      character.references.forEach((reference) => {
        if (reference.server) form.append('reuse_images', reference.ref)
        else if (reference.file) form.append('images', reference.file)
      })
      const result = await api.describeSequenceCharacter(form)
      this.updateCharacter(index, {
        nameKO: String(result.name_ko || '').trim(), nameEN: String(result.name_en || '').trim(),
        descriptionKO: String(result.description_ko || '').trim(),
        canonicalPromptEN: String(result.canonical_prompt_en || '').trim(),
        observations: result.observations || {}, analyzing: false, error: ''
      })
      return result
    } catch (error) {
      this.updateCharacter(index, { analyzing: false, error: error.message || String(error) }, false)
      throw error
    }
  }

  characterReadinessMessage() {
    const pending = this.current.characters.find((character) => character.references.length && !character.canonicalPromptEN.trim())
    if (!pending) return ''
    const name = pending.name || '등장인물'
    if (pending.quadViewGenerating) return `${name}의 360° 외형 자료를 생성하고 있습니다. 완료될 때까지 기다리세요.`
    if (pending.turntableFrames?.length) {
      const selected = pending.turntableSelection?.length || 0
      return `${name}: 생성된 방향 이미지에서 사용할 자료를 고른 뒤 아래의 “선택 ${selected}장 승인 · 외형 분석”을 누르세요.`
    }
    if (pending.analyzing) return `${name}의 외형 고정 문구를 만들고 있습니다.`
    return `${name}: 캐릭터 준비에서 “이미지에서 외형 고정 문구 만들기”를 누르세요.`
  }

  reidReference() {
    const character = this.current.characters[0]
    if (!character?.references?.length) return null
    return character.references[Math.min(character.reidReferenceIndex || 0, character.references.length - 1)] || character.references[0]
  }

  applyStoryExample(reference = null, example = null) {
    this.current.characters.forEach((character) => {
      clearMediaInputs(character.references)
      clearMediaInputs(character.turntableFrames || [])
      clearMediaInputs(character.quadViewCandidate ? [character.quadViewCandidate] : [])
    })
    const sample = example?.story ? example : { name: '연화', story: storyExample.idea, scenes: Array(storyExample.count).fill('') }
    const characters = reference ? [newCharacter(1, {
      name: sample.name, references: normalizeImageFiles([reference]),
      lockedTraits: sample.lockedTraits || { face: true, hair: true, body: true, outfit: true, accessories: false, mechanical: false },
      descriptionKO: String(sample.descriptionKO || '').trim(), canonicalPromptEN: String(sample.canonicalPromptEN || '').trim()
    })] : []
    this.setState({ ...initialState(), entryMode: 'story', storyIdea: sample.story, sceneCount: sample.scenes.length, characters, characterCounter: characters.length })
  }

  applySceneExample(reference = null, example = null) {
    this.current.characters.forEach((character) => {
      clearMediaInputs(character.references)
      clearMediaInputs(character.turntableFrames || [])
      clearMediaInputs(character.quadViewCandidate ? [character.quadViewCandidate] : [])
    })
    const sample = example?.scenes?.length ? example : { name: '연화', scenes: sceneExample }
    const characters = reference ? [newCharacter(1, {
      name: sample.name, references: normalizeImageFiles([reference]),
      lockedTraits: sample.lockedTraits || { face: true, hair: true, body: true, outfit: true, accessories: false, mechanical: false },
      descriptionKO: String(sample.descriptionKO || '').trim(), canonicalPromptEN: String(sample.canonicalPromptEN || '').trim()
    })] : []
    this.setState({ ...initialState(), entryMode: 'scenes', prompts: [...sample.scenes], characters, characterCounter: characters.length })
  }

  addScene() {
    if (this.current.prompts.length >= 12) return
    this.invalidatePlan({ entryMode: 'scenes', prompts: [...this.current.prompts, ''] })
  }

  removeScene(index) {
    if (this.current.prompts.length <= 2) return
    this.invalidatePlan({ entryMode: 'scenes', prompts: this.current.prompts.filter((_, itemIndex) => itemIndex !== index) })
  }

  moveScene(index, direction) {
    const destination = index + direction
    if (destination < 0 || destination >= this.current.prompts.length) return
    const prompts = [...this.current.prompts]
    ;[prompts[index], prompts[destination]] = [prompts[destination], prompts[index]]
    this.invalidatePlan({ entryMode: 'scenes', prompts })
  }

  updatePrompt(index, prompt) {
    this.invalidatePlan({
      entryMode: 'scenes',
      prompts: this.current.prompts.map((value, itemIndex) => itemIndex === index ? prompt : value)
    })
  }

  canPlan() {
    if (this.current.entryMode === 'story') return Boolean(this.current.storyIdea.trim())
    return this.current.prompts.length >= 2 && this.current.prompts.every((value) => value.trim())
  }

  async plan(api) {
    const signature = this.signature()
    if (this.current.planSignature === signature && this.current.enhancedPrompts.length === this.current.prompts.length) return this.current
    if (!this.canPlan()) throw new Error(this.current.entryMode === 'story' ? '이야기나 주제를 입력하세요.' : '모든 장면의 내용을 입력하세요.')
    const characterMessage = this.characterReadinessMessage()
    if (characterMessage) throw new Error(characterMessage)
    const storyMode = this.current.entryMode === 'story'
    this.setState({ planning: true, planError: '' })
    try {
      const characterPayload = lockedCharacters(this.current)
      const common = {
        ...(this.current.sharedPromptEdited ? { shared_prompt: this.current.sharedPrompt.trim() } : {}),
        ...(characterPayload.length ? { locked_characters: characterPayload } : {})
      }
      const result = await api.planImageSequence(storyMode
        ? { outline: this.current.storyIdea.trim(), scene_count: this.current.sceneCount, ...common }
        : { prompts: this.current.prompts.map((value) => value.trim()), ...common })
      const scenes = Array.isArray(result.scenes) ? result.scenes : []
      const expected = storyMode ? this.current.sceneCount : this.current.prompts.length
      if (scenes.length !== expected) throw new Error('장면 계획의 개수가 맞지 않습니다.')
      const prompts = storyMode
        ? scenes.map((scene) => String(scene.original_prompt || scene.change_summary || '').trim())
        : [...this.current.prompts]
      if (prompts.some((value) => !value)) throw new Error('장면 설명이 비어 있습니다.')
      const planSignature = JSON.stringify({
        entryMode: 'scenes', storyIdea: this.current.storyIdea.trim(), sceneCount: prompts.length,
        prompts: prompts.map((value) => value.trim()), sharedPrompt: this.current.sharedPromptEdited ? this.current.sharedPrompt.trim() : '',
        lockedCharacters: characterPayload
      })
      this.setState({
        entryMode: 'scenes', prompts,
        enhancedPrompts: scenes.map((scene) => scene.enhanced_prompt),
        sceneTitles: scenes.map((scene, index) => scene.scene_title || scene.change_summary || `장면 ${index + 1}`),
        sharedPrompt: result.shared_prompt || '', canonicalPrompt: result.canonical_prompt_en || '', planSignature,
        sceneCount: prompts.length, planning: false, planError: ''
      })
      return this.current
    } catch (error) {
      this.setState({ planning: false, planError: error.message || String(error) })
      throw error
    }
  }

  destroy() {
    this.current.characters.forEach((character) => { clearMediaInputs(character.references); clearMediaInputs(character.turntableFrames || []); clearMediaInputs(character.quadViewCandidate ? [character.quadViewCandidate] : []) })
    this.unsubscribe?.()
  }
}

export function imageSequenceBlockedMessage({ mode, modules, moduleReason, checkpoint = 'official-int8', hasReIDReference = false }) {
  if (mode !== 'create') return '다중 장면은 새 이미지 생성에서만 사용할 수 있습니다.'
  const incompatible = [
    [modules.identity, '원본 수정'], [modules.depth, '자세·구도'],
    [modules.vision, '내용·구도 참조'], [modules.styleReference, '스타일 참조'],
    [modules.nk2e, '편집·윤곽'], [modules.anypaint, '부분 수정·확장']
  ].filter(([enabled]) => enabled).map(([, label]) => label)
  if (incompatible.length) return `${incompatible.join(' · ')} 모듈을 끈 뒤 사용할 수 있습니다.`
  if (hasReIDReference && checkpoint !== 'official-int8' && checkpoint !== 'official') return 'ReID 외형 고정은 공식 Krea 체크포인트에서만 사용할 수 있습니다.'
  if (moduleReason) return moduleReason
  return ''
}
