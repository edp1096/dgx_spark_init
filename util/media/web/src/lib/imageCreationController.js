import { buildImageGenerationForm } from './generationRequests.js'
import {
  identityHasExtraUserPrompt,
  identityPreserveDefaults,
  imageDisabledReason,
  imageEnhancementActive,
  imageEnhancementCurrent,
  implicitModulePrompt,
  kreaModuleDisabledReason,
  rawImagePrompt
} from './imageWorkflow.js'
import { snapDimension } from './videoTiming.js'

const emptyModules = () => ({
  identity: false,
  depth: false,
  style: false,
  userLora: false,
  vision: false,
  styleReference: false,
  nk2e: false,
  anypaint: false
})

export class ImageCreationController {
  constructor({ api, catalogs, actions }) {
    this.api = api
    this.catalogs = catalogs
    this.actions = actions
  }

  state() { return this.actions.getState() }

  implicitPrompt(state = this.state()) {
    return implicitModulePrompt({ modules: state.modules, identityPreset: state.identityPreset, anypaintImage: state.anypaintImage, anypaintMask: state.anypaintMask, options: state.options })
  }

  hasExtraIdentityPrompt(state = this.state()) {
    return identityHasExtraUserPrompt({ enteredPrompt: state.form.prompt, implicitPrompt: this.implicitPrompt(state) })
  }

  rawPrompt(state = this.state()) {
    return rawImagePrompt({ enteredPrompt: state.form.prompt, implicitPrompt: this.implicitPrompt(state), modules: state.modules, identityPreset: state.identityPreset, identityPreserveCustom: state.identityPreserveCustom })
  }

  looksStructured(value = this.rawPrompt()) {
    const text = String(value || '').trim()
    if (!text || (text[0] !== '{' && text[0] !== '[')) return false
    try { JSON.parse(text); return true } catch { return false }
  }

  enhancementActive(state = this.state()) {
    const prompt = this.rawPrompt(state)
    return imageEnhancementActive({
      enabled: state.enhanceEnabled,
      prompt,
      structured: this.looksStructured(prompt),
      identityTryonWithoutUserPrompt: state.modules.identity && state.identityPreset === 'tryon' && !this.hasExtraIdentityPrompt(state)
    })
  }

  enhancementCurrent(state = this.state()) {
    return imageEnhancementCurrent({ enhanced: state.enhancedPrompt, source: state.enhancedSource, current: this.rawPrompt(state) })
  }

  resetEnhancement() {
    this.actions.patch({ enhancedPrompt: '', enhancedSource: '' })
  }

  preserveDefaults(preset, state = this.state()) {
    return identityPreserveDefaults(preset, this.catalogs.defaultIdentityPreserveItems, state.modules.depth)
  }

  setPreserveItems(items) {
    const allowed = this.catalogs.identityPreserveCatalog.map((item) => item.id)
    this.actions.patch({ identityPreserveItems: allowed.filter((id) => items.includes(id)) })
    this.resetEnhancement()
  }

  togglePreserveItem(id) {
    const state = this.state()
    if (state.modules.depth && (id === 'pose' || id === 'composition')) return
    this.setPreserveItems(state.identityPreserveItems.includes(id)
      ? state.identityPreserveItems.filter((item) => item !== id)
      : [...state.identityPreserveItems, id])
  }

  applyIdentityPreset(value) {
    const state = this.state()
    const ui = this.catalogs.identityPresetUI[value] || this.catalogs.identityPresetUI['']
    if (!ui.showSecondary) this.actions.setKreaImage('identityReference', null)
    const prompts = {
      restage: 'Place the same person in a new scene and pose as described',
      sheet: 'Create a clean 2x2 character sheet on a plain background: front view upper-left, three-quarter view upper-right, left profile lower-left, and back view lower-right',
      tryon: '',
      replace: 'Replace only the selected object or region as described',
      faceSwap: 'Replace only the face of the person in Image One with the face from Image Two',
      headSwap: 'Replace the entire head of the person in Image One with the head from Image Two',
      personSwap: 'Replace the entire person in Image One with the person from Image Two'
    }
    const patch = {
      identityPreset: value,
      identityPreserveItems: identityPreserveDefaults(value, this.catalogs.defaultIdentityPreserveItems, state.modules.depth),
      identityPreserveCustom: ''
    }
    if (value in prompts) patch.form = { ...state.form, prompt: prompts[value] }
    this.actions.patch(patch)
    this.resetEnhancement()
  }

  moduleDisabledReason(state = this.state()) {
    const ui = this.catalogs.identityPresetUI[state.identityPreset] || this.catalogs.identityPresetUI['']
    return kreaModuleDisabledReason({
      modules: state.modules,
      identityUI: ui,
      identityImage: state.identityImage,
      identityReference: state.identityReferences[0] || null,
      depthImage: state.depthImage,
      visionImages: state.visionImages,
      styleReferenceImages: state.styleReferenceImages,
      styleSelections: state.styleSelections,
      userLoraSelections: state.userLoraSelections,
      nk2eImage: state.nk2eImage,
      anypaintImage: state.anypaintImage,
      anypaintMask: state.anypaintMask,
      options: state.options
    })
  }

  disabledReason(state = this.state()) {
    return imageDisabledReason({ busy: state.busy, prompt: this.rawPrompt(state), imageForm: state.form, references: state.references, moduleReason: this.moduleDisabledReason(state) })
  }

  samplingPreset(checkpoint, current = 'default') {
    if (checkpoint?.startsWith('moody-')) return 'moody'
    return current === 'moody' ? 'default' : current
  }

  applySmartResolution() {
    const state = this.state()
    if (state.resolutionMode !== 'smart') return
    const ratio = this.catalogs.imageAspectRatios.find((item) => item[0] === state.aspectRatio)?.[1] || 1
    const pixels = Number(state.megapixels) * 1024 * 1024
    const width = Math.sqrt(pixels * ratio)
    const height = width / ratio
    const multiple = state.modules.anypaint ? 16 : 8
    this.actions.patch({ form: {
      ...state.form,
      width: Math.min(2048, Math.max(256, Math.round(width / multiple) * multiple)),
      height: Math.min(2048, Math.max(256, Math.round(height / multiple) * multiple))
    } })
  }

  reset() {
    const state = this.state()
    this.actions.clearAllInputs()
    this.actions.resetSequence()
    this.actions.closeSequence()
    const checkpoint = state.config?.image?.default_checkpoint || 'official'
    this.actions.patch({
      modules: emptyModules(),
      styleSelections: [{ name: 'retroanime', strength: 1 }],
      userLoraSelections: [],
      options: {
        checkpoint,
        identity_strength: 1, ref_boost: 4, source_ref_boost: 1, grounding_px: 768, steps: 8,
        identity_model: 'convrot', identity_encoder: 'heretic',
        sampling_preset: this.samplingPreset(checkpoint, 'default'),
        depth_strength: 0.8,
        vision_mode: 'descriptor', vision_megapixels: 1, style_reference_strength: 1,
        nk2e_mode: 'edit', nk2e_strength: 0.7, vae_mode: 'default', identity_fit_mode: 'fit',
        strict_mask_grow: 0, strict_mask_feather: 0,
        outpaint_left: 0, outpaint_top: 0, outpaint_right: 0, outpaint_bottom: 0,
        anypaint_strength: 1, anypaint_boundary_redraw_px: 32,
        filter_mode: checkpoint === 'official' ? 'balanced' : 'off', filter_strength: checkpoint === 'official' ? 1 : 0,
        prompt_enhancer: Boolean(state.config?.image?.default_prompt_enhancer), prompt_enhancer_strength: 1, prompt_text_scale: 1.75
      },
      form: { prompt: '', width: 1024, height: 1024, seed: -1, mode: 'create' },
      resolutionMode: 'smart', aspectRatio: '1:1', megapixels: 1,
      enhanceEnabled: state.config?.prompt_enhancement?.default_enabled ?? true,
      filterPromptPreset: '', parentJobID: '', identityPreset: '',
      identityPreserveItems: [...this.catalogs.defaultIdentityPreserveItems], identityPreserveCustom: '',
      depthPoseID: '', nk2ePoseID: '', nk2ePreprocessed: false,
      enhancedPrompt: '', enhancedSource: ''
    })
    this.actions.clearCloneMessage()
    this.applySmartResolution()
  }

  async enhance() {
    const state = this.state()
    const original = this.rawPrompt(state)
    if (!original || this.looksStructured(original)) return ''
    this.actions.patch({ enhancing: true, error: '' })
    try {
      const form = new FormData()
      form.append('prompt', original)
      form.append('mode', state.modules.identity && state.modules.depth ? 'edit_control' : state.modules.identity ? 'edit' : state.modules.anypaint ? 'paint' : (state.modules.depth || state.modules.nk2e) ? 'control' : 't2i')
      if (state.modules.identity) {
        form.append('identity_preset', state.identityPreset)
        form.append('identity_preserve_items', JSON.stringify(state.identityPreserveItems))
      }
      const result = await this.api.enhancePrompt(form)
      this.actions.patch({ enhancedPrompt: result.enhanced_prompt, enhancedSource: original })
      return result.enhanced_prompt
    } catch (error) {
      this.actions.patch({ error: error.message })
      return ''
    } finally {
      this.actions.patch({ enhancing: false })
    }
  }

  clonePrompt(job) {
    const state = this.state()
    this.actions.patch({ filterPromptPreset: '', form: { ...state.form, prompt: job.prompt || '' } })
    this.resetEnhancement()
  }

  cloneSettings(job) {
    const state = this.state()
    const params = job.params || {}
    const legacyKlein = params.mode === 'edit'
    const storedStyles = Array.isArray(params.styles) && params.styles.length
      ? params.styles.map((style) => ({ name: style.name, strength: Number(style.strength) }))
      : (params.style ? [{ name: params.style, strength: params.style_strength !== undefined ? Number(params.style_strength) : 1 }] : [])
    const storedUserLoras = Array.isArray(params.user_loras)
      ? params.user_loras.filter((selection) => selection.filename !== 'skc3vo.safetensors').map((selection) => ({ filename: selection.filename, strength: Number(selection.strength) }))
      : []
    const form = {
      ...state.form,
      mode: this.catalogs.imageModeChoices.includes(params.mode) ? params.mode : 'create',
      width: Number(params.width) || state.form.width,
      height: Number(params.height) || state.form.height,
      seed: Number.isFinite(Number(params.seed)) ? Number(params.seed) : -1
    }
    const patch = { form, resolutionMode: 'custom' }
    if (form.mode === 'create') {
      patch.modules = {
        identity: legacyKlein || Boolean(params.identity), depth: Boolean(params.depth), style: storedStyles.length > 0,
        userLora: storedUserLoras.length > 0, vision: Boolean(params.vision), styleReference: Boolean(params.style_reference),
        nk2e: Boolean(params.nk2e), anypaint: Boolean(params.anypaint)
      }
      patch.identityPreset = params.identity_preset || ''
      patch.identityPreserveItems = Array.isArray(params.identity_preserve_items)
        ? this.catalogs.identityPreserveCatalog.map((item) => item.id).filter((id) => params.identity_preserve_items.includes(id))
        : identityPreserveDefaults(patch.identityPreset, this.catalogs.defaultIdentityPreserveItems, patch.modules.depth)
      if (patch.modules.depth) patch.identityPreserveItems = patch.identityPreserveItems.filter((id) => id !== 'pose' && id !== 'composition')
      patch.identityPreserveCustom = params.identity_preserve_custom || ''
      patch.options = {
        ...state.options,
        checkpoint: params.checkpoint || 'official', identity_strength: params.identity_strength !== undefined ? Number(params.identity_strength) : 1,
        identity_model: params.identity_model || 'convrot', identity_encoder: params.identity_encoder || 'heretic',
        ref_boost: params.ref_boost !== undefined ? Number(params.ref_boost) : 4, source_ref_boost: params.source_ref_boost !== undefined ? Number(params.source_ref_boost) : 1,
        grounding_px: Number(params.grounding_px) || 768, steps: Number(params.steps) || (params.identity ? 10 : 8),
        depth_strength: params.depth_strength !== undefined ? Number(params.depth_strength) : 0.8,
        vision_mode: params.vision_mode || 'descriptor', vision_megapixels: params.vision_megapixels !== undefined ? Number(params.vision_megapixels) : 1,
        style_reference_strength: params.style_reference_strength !== undefined ? Number(params.style_reference_strength) : 1,
        nk2e_mode: params.nk2e_mode || 'edit', nk2e_strength: params.nk2e_strength !== undefined ? Number(params.nk2e_strength) : 0.7,
        vae_mode: params.vae_mode || 'default', identity_fit_mode: params.identity_fit_mode || 'fit',
        strict_mask_grow: Number(params.strict_mask_grow) || 0, strict_mask_feather: Number(params.strict_mask_feather) || 0,
        outpaint_left: Number(params.outpaint_left) || 0, outpaint_top: Number(params.outpaint_top) || 0,
        outpaint_right: Number(params.outpaint_right) || 0, outpaint_bottom: Number(params.outpaint_bottom) || 0,
        anypaint_strength: params.anypaint_strength !== undefined ? Number(params.anypaint_strength) : 1,
        anypaint_boundary_redraw_px: params.anypaint_boundary_redraw_px !== undefined ? Number(params.anypaint_boundary_redraw_px) : 32,
        filter_mode: params.filter_mode || 'balanced',
        filter_strength: params.filter_strength !== undefined ? Number(params.filter_strength) : this.actions.filterModeDefault(params.filter_mode || 'balanced'),
        prompt_enhancer: Boolean(params.prompt_enhancer), prompt_enhancer_strength: params.prompt_enhancer_strength !== undefined ? Number(params.prompt_enhancer_strength) : 1,
        prompt_text_scale: params.prompt_text_scale !== undefined ? Number(params.prompt_text_scale) : 1.75,
        sampling_preset: params.sampling_preset || (params.sampler === 'er_sde' ? 'detail' : params.sampler === 'euler_ancestral' ? 'moody' : 'default')
      }
      patch.styleSelections = storedStyles.length ? storedStyles : [{ name: 'retroanime', strength: 1 }]
      patch.userLoraSelections = storedUserLoras
    }
    this.actions.patch(patch)
  }

  async cloneReferences(job) {
    const inputs = await this.api.imageInputs(job.id)
    const stored = inputs.map((input) => ({ ...input, server: true }))
    this.actions.clearAllInputs()
    const legacyKlein = job.params?.mode === 'edit'
    let identitySeen = false
    for (const input of stored) {
      if (input.role === 'reference' && legacyKlein && !identitySeen) { this.actions.setKreaImage('identity', input); identitySeen = true }
      else if (input.role === 'reference' && legacyKlein) this.actions.setKreaImage('identityReference', input)
      else if (input.role === 'reference') this.actions.addReferences([input])
      else if (input.role === 'identity') this.actions.setKreaImage('identity', input)
      else if (input.role === 'identity_reference') this.actions.addIdentityReferences([input])
      else if (input.role === 'identity_mask') this.actions.setKreaImage('identityMask', input)
      else if (input.role === 'strict_mask') this.actions.setKreaImage('strictMask', input)
      else if (input.role === 'depth') this.actions.setKreaImage('depth', input)
      else if (input.role === 'vision') this.actions.addKreaReferences('vision', [input])
      else if (input.role === 'style_reference') this.actions.addKreaReferences('styleReference', [input])
      else if (input.role === 'nk2e') this.actions.setKreaImage('nk2e', input)
      else if (input.role === 'anypaint') this.actions.setKreaImage('anypaint', input)
      else if (input.role === 'anypaint_mask') this.actions.setKreaImage('anypaintMask', input)
    }
    const state = this.state()
    const mode = this.catalogs.imageModeChoices.includes(job.params?.mode) ? job.params.mode : 'create'
    const patch = { form: { ...state.form, mode } }
    if (mode === 'create') patch.modules = {
      ...state.modules,
      identity: legacyKlein || stored.some((input) => input.role === 'identity'), depth: stored.some((input) => input.role === 'depth'),
      vision: stored.some((input) => input.role === 'vision'), styleReference: stored.some((input) => input.role === 'style_reference'),
      nk2e: stored.some((input) => input.role === 'nk2e'), anypaint: stored.some((input) => input.role === 'anypaint')
    }
    this.actions.patch(patch)
    return stored.length
  }

  continueEditing(job) {
    const state = this.state()
    const source = { server: true, ref: `${job.id}:output:0`, url: job.output_url, name: `결과 ${job.id.slice(0, 8)}.png`, role: 'output' }
    this.actions.clearAllInputs()
    this.actions.setKreaImage('identity', source)
    this.actions.patch({
      modules: { ...emptyModules(), identity: true }, parentJobID: job.id,
      form: { ...state.form, prompt: '', width: Number(job.params?.width) || 1024, height: Number(job.params?.height) || 1024 },
      resolutionMode: 'custom', identityPreserveItems: [...this.catalogs.defaultIdentityPreserveItems], identityPreserveCustom: '',
      mobilePane: 'create'
    })
    this.resetEnhancement()
    this.actions.scrollTop()
  }

  async generate(sequencePrompts = null) {
    let state = this.state()
    this.actions.patch({ form: { ...state.form, width: snapDimension(state.form.width, 8, 256, 2048), height: snapDimension(state.form.height, 8, 256, 2048) } })
    state = this.state()
    const isSequence = Array.isArray(sequencePrompts)
    const requestedSequenceCount = isSequence ? sequencePrompts.length : 0
    if (isSequence) {
      try {
        await this.actions.planSequence()
        state = this.state()
        sequencePrompts = state.sequence.prompts.slice(0, requestedSequenceCount)
      } catch (error) {
        this.actions.patch({ error: error.message || String(error) })
        return
      }
    }
    if (!isSequence && this.enhancementActive(state) && !this.enhancementCurrent(state)) {
      await this.enhance()
      state = this.state()
      if (!this.enhancementCurrent(state)) return
    }
    this.actions.patch({ busy: true, error: '' })
    try {
      const sequence = state.sequence
      const firstPrompt = isSequence ? sequence.enhancedPrompts[0].trim() : this.rawPrompt(state)
      const userPrompt = isSequence ? sequencePrompts[0].trim() : (state.form.prompt.trim() || this.implicitPrompt(state))
      const form = buildImageGenerationForm({
        imageForm: state.form,
        prompt: isSequence ? firstPrompt : this.enhancementActive(state) ? state.enhancedPrompt : firstPrompt,
        originalPrompt: userPrompt,
        parentJobID: state.parentJobID,
        sequence: isSequence ? {
          prompts: sequencePrompts, enhancedPrompts: sequence.enhancedPrompts.slice(0, requestedSequenceCount),
          sharedPrompt: sequence.sharedPrompt, canonicalPrompt: sequence.canonicalPrompt,
          reidImage: (() => {
            const character = sequence.characters?.[0]
            if (!character?.references?.length) return null
            return character.references[Math.min(character.reidReferenceIndex || 0, character.references.length - 1)] || character.references[0]
          })()
        } : null,
        modules: state.modules, options: state.options,
        identity: { preset: state.identityPreset, hasUserPrompt: this.hasExtraIdentityPrompt(state), preserveItems: state.identityPreserveItems, preserveCustom: state.identityPreserveCustom, image: state.identityImage, references: state.identityReferences, mask: state.identityMask, strictMask: state.strictMask },
        depth: { image: state.depthImage }, styles: state.styleSelections, userLoras: state.userLoraSelections,
        visionImages: state.visionImages, styleReferenceImages: state.styleReferenceImages,
        nk2e: { image: state.nk2eImage, preprocessed: state.nk2ePreprocessed }, anypaint: { image: state.anypaintImage, mask: state.anypaintMask }, references: state.references
      })
      await this.api.image(form)
      this.actions.closeSequence()
      this.actions.clearGeneratedInputs()
      this.actions.patch({ form: { ...this.state().form, prompt: '' }, filterPromptPreset: '', parentJobID: '', identityPreset: '', mobilePane: 'results', enhancedPrompt: '', enhancedSource: '' })
      this.actions.showNewest()
      await this.actions.refresh()
    } catch (error) {
      this.actions.patch({ error: error.message })
    } finally {
      this.actions.patch({ busy: false })
    }
  }
}
