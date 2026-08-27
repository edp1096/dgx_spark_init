import { writable } from 'svelte/store'
import {
  appendMediaInputs,
  clearMediaInputs,
  normalizeImageFiles,
  releaseMediaInput,
  removeMediaInput,
  replaceMediaInput
} from './mediaInputs.js'

const initialState = {
  refs: [],
  identityImage: null,
  identityReferences: [],
  depthImage: null,
  nk2eImage: null,
  anypaintImage: null,
  anypaintMask: null,
  identityMask: null,
  strictMask: null,
  visionImages: [],
  styleReferenceImages: []
}

const singleKeys = {
  identity: 'identityImage',
  depth: 'depthImage',
  nk2e: 'nk2eImage',
  anypaint: 'anypaintImage',
  anypaintMask: 'anypaintMask',
  identityMask: 'identityMask',
  strictMask: 'strictMask'
}

export class ImageInputController {
  constructor() {
    this.current = { ...initialState }
    this.state = writable(this.current)
    this.unsubscribe = this.state.subscribe((value) => this.current = value)
  }

  setState(patch) {
    this.state.update((value) => ({ ...value, ...patch }))
  }

  addRefs(incoming, limit) {
    this.setState({ refs: appendMediaInputs(this.current.refs, incoming, limit) })
  }

  addRefFiles(files, limit) {
    this.addRefs(normalizeImageFiles(files), limit)
  }

  clearRefs() {
    this.setState({ refs: clearMediaInputs(this.current.refs) })
  }

  removeRef(index) {
    this.setState({ refs: removeMediaInput(this.current.refs, index) })
  }

  addKreaRefs(kind, incoming) {
    const key = kind === 'vision' ? 'visionImages' : 'styleReferenceImages'
    const limit = kind === 'vision' ? 4 : 2
    this.setState({ [key]: appendMediaInputs(this.current[key], incoming, limit) })
  }

  addKreaRefFiles(kind, files) {
    this.addKreaRefs(kind, normalizeImageFiles(files))
  }

  clearKreaRefs(kind) {
    const key = kind === 'vision' ? 'visionImages' : 'styleReferenceImages'
    this.setState({ [key]: clearMediaInputs(this.current[key]) })
  }

  removeKreaRef(kind, index) {
    const key = kind === 'vision' ? 'visionImages' : 'styleReferenceImages'
    this.setState({ [key]: removeMediaInput(this.current[key], index) })
  }

  addIdentityReferences(incoming) {
    this.setState({ identityReferences: appendMediaInputs(this.current.identityReferences, incoming, 3) })
  }

  addIdentityReferenceFiles(files) {
    this.addIdentityReferences(normalizeImageFiles(files))
  }

  clearIdentityReferences() {
    this.setState({ identityReferences: clearMediaInputs(this.current.identityReferences) })
  }

  removeIdentityReference(index) {
    this.setState({ identityReferences: removeMediaInput(this.current.identityReferences, index) })
  }

  setImage(kind, image) {
    if (kind === 'identityReference') {
      this.clearIdentityReferences()
      if (image) this.addIdentityReferences([image])
      return this.current.identityReferences[0] || null
    }
    const key = singleKeys[kind] || 'strictMask'
    const patch = {}
    if (kind === 'anypaint' && image && this.current.anypaintImage !== image && this.current.anypaintMask) {
      patch.anypaintMask = replaceMediaInput(this.current.anypaintMask, null)
    }
    patch[key] = replaceMediaInput(this.current[key], image)
    this.setState(patch)
    return this.current[key]
  }

  destroy() {
    ;[
      ...this.current.refs,
      ...this.current.identityReferences,
      ...this.current.visionImages,
      ...this.current.styleReferenceImages,
      this.current.identityImage,
      this.current.depthImage,
      this.current.nk2eImage,
      this.current.anypaintImage,
      this.current.anypaintMask,
      this.current.identityMask,
      this.current.strictMask
    ].filter(Boolean).forEach((input) => releaseMediaInput(input))
    this.unsubscribe?.()
  }
}
