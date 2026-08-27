import { selectionLabel, toggleImageModuleState, toggleStyleSelection, toggleUserLoraSelection, updateSelectionStrength } from './imageModuleState.js'

export class ImageConfigurationController {
  constructor({ api, catalogs, actions }) {
    this.api = api
    this.catalogs = catalogs
    this.actions = actions
  }

  state() { return this.actions.getState() }

  toggleModule(module) {
    const state = this.state()
    const next = toggleImageModuleState({
      module, modules: state.modules, options: state.options, preserveItems: state.preserveItems,
      imagePixels: state.form.width * state.form.height
    })
    this.actions.patch({ modules: next.modules, options: next.options })
    this.actions.setPreserveItems(next.preserveItems)
    if (next.imageMegapixels != null) this.actions.patch({ megapixels: next.imageMegapixels })
    if (next.imageResolutionMode) this.actions.patch({ resolutionMode: next.imageResolutionMode })
    if (next.applySmartResolution) this.actions.applySmartResolution()
    if (next.message) this.actions.setMessage(next.message)
    next.clearTargets.forEach((target) => target === 'vision' || target === 'styleReference'
      ? this.actions.clearKreaRefs(target)
      : this.actions.setKreaImage(target, null))
  }

  async refreshUserLoras() {
    let catalog = []
    try { catalog = (await this.api.userLoras()).filter((item) => item.filename !== 'skc3vo.safetensors') } catch (_) {}
    const state = this.state()
    this.actions.patch({
      userLoraCatalog: catalog,
      userLoraSelections: state.userLoraSelections.filter((selection) => catalog.some((item) => item.filename === selection.filename))
    })
    return catalog
  }

  hasUserLora(filename) { return this.state().userLoraSelections.some((selection) => selection.filename === filename) }
  toggleUserLora(filename) {
    const state = this.state()
    this.actions.patch({ userLoraSelections: toggleUserLoraSelection(state.userLoraSelections, state.userLoraCatalog, filename) })
  }
  updateUserLoraStrength(filename, strength) {
    this.actions.patch({ userLoraSelections: updateSelectionStrength(this.state().userLoraSelections, 'filename', filename, strength) })
  }
  userLoraLabel(filename) { return selectionLabel(this.state().userLoraCatalog, 'filename', filename, 'name') }

  hasStyle(name) { return this.state().styleSelections.some((style) => style.name === name) }
  toggleStyle(name) { this.actions.patch({ styleSelections: toggleStyleSelection(this.state().styleSelections, name) }) }
  updateStyleStrength(name, strength) { this.actions.patch({ styleSelections: updateSelectionStrength(this.state().styleSelections, 'name', name, strength) }) }
  styleLabel(name) { return selectionLabel(this.catalogs.kreaStyleCatalog, 'name', name, 'label') }

  filterModeDefault(mode) {
    if (mode === 'adherence') return 0.05
    if (mode === 'balanced' || mode === 'strong') return 1
    return 0
  }
  filterModeMaximum(mode) { return mode === 'adherence' ? 0.2 : 2 }
  samplingPreset(checkpoint, current = 'default') {
    if (checkpoint?.startsWith('moody-')) return 'moody'
    return current === 'moody' ? 'default' : current
  }

  checkpointVisible(checkpoint) {
    if (checkpoint === 'official') return true
    const visible = this.state().settings?.image?.visible_checkpoints
    return !Array.isArray(visible) || visible.includes(checkpoint)
  }

  setCheckpointVisible(checkpoint, visible) {
    const state = this.state()
    if (!state.settings?.image || checkpoint === 'official') return
    const selected = new Set(state.settings.image.visible_checkpoints || this.catalogs.checkpointDisplayChoices.map(([id]) => id))
    if (visible) selected.add(checkpoint)
    else selected.delete(checkpoint)
    const settings = structuredClone(state.settings)
    settings.image.visible_checkpoints = ['official', ...this.catalogs.checkpointDisplayChoices.map(([id]) => id).filter((id) => selected.has(id))]
    if (!visible && settings.image.default_checkpoint === checkpoint) settings.image.default_checkpoint = 'official'
    this.actions.patch({ settings })
    if (!visible && state.options.checkpoint === checkpoint) this.selectCheckpoint('official')
  }

  checkpointReady(checkpoint) {
    const status = this.state().checkpointStatus
    if (checkpoint === 'ray-v2-nvfp4' || checkpoint === 'ray-v4-nvfp4') {
      const source = checkpoint.replace('-nvfp4', '')
      return Boolean(status?.nvfp4_conversion?.variants?.find((item) => item.id === source)?.validated)
    }
    return Boolean(status?.variants?.find((item) => item.id === checkpoint)?.ready)
  }

  selectCheckpoint(checkpoint) {
    const state = this.state()
    if (checkpoint === 'identity-convrot') {
      this.actions.patch({ options: { ...state.options, identity_model: 'convrot', sampling_preset: 'default', filter_mode: 'off', filter_strength: 0 } })
      return
    }
    this.actions.patch({ options: {
      ...state.options, checkpoint,
      identity_model: state.modules.identity ? 'selected' : state.options.identity_model,
      sampling_preset: this.samplingPreset(checkpoint, state.options.sampling_preset),
      ...(checkpoint === 'official' ? {} : { filter_mode: 'off', filter_strength: 0 })
    } })
  }

  selectedCheckpoint() {
    const state = this.state()
    return state.modules.identity && state.options.identity_model === 'convrot' ? 'identity-convrot' : state.options.checkpoint
  }

  selectedCheckpointSource() {
    const state = this.state()
    const checkpoint = this.selectedCheckpoint()
    if (checkpoint === 'identity-convrot') return state.checkpointStatus?.identity_runtime?.convrot_source || 'https://huggingface.co/Winnougan/Krea-2-Base-Turbo-NVFP4-FP8-INT8'
    if (checkpoint === 'official') return 'https://huggingface.co/krea/Krea-2-Turbo'
    const sourceID = checkpoint.replace('-nvfp4', '')
    return state.checkpointStatus?.variants?.find((item) => item.id === sourceID)?.source
      || state.checkpointStatus?.nvfp4_conversion?.variants?.find((item) => item.id === sourceID)?.source || ''
  }
}
