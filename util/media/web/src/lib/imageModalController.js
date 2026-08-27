import { writable } from 'svelte/store'

const initialState = {
  image: null,
  garmentOpen: false,
  garmentInitialJob: null,
  sequenceOpen: false,
  sequenceMaskEditorIndex: -1,
  sequenceRegionPicker: -1,
  maskEditorMode: '',
  cannyEditorOpen: false,
  runtimeInfoOpen: false,
  featureModulesOpen: false,
  recentPickerTarget: '',
  presetPickerTarget: '',
  remoteTarget: ''
}

export class ImageModalController {
  constructor(actions) {
    this.actions = actions
    this.current = { ...initialState }
    this.state = writable(this.current)
    this.state.subscribe((value) => this.current = value)
  }

  setState(patch) {
    this.state.update((value) => ({ ...value, ...patch }))
  }

  showImage(event, src, title, detail = '', jobID = '') {
    event?.preventDefault()
    event?.stopPropagation()
    if (src) this.setState({ image: { src, title, detail, jobID } })
  }

  showImageOnKey(event, src, title, detail = '') {
    if (event.key === 'Enter' || event.key === ' ') this.showImage(event, src, title, detail)
  }

  closeImage() {
    this.setState({ image: null })
  }

  openGarment(job = null) {
    this.setState({
      garmentInitialJob: job?.output_url ? job : null,
      garmentOpen: true
    })
  }

  openGarmentFromImage(jobID) {
    const job = this.actions.getImageJobs().find((item) => item.id === jobID)
    this.closeImage()
    this.openGarment(job)
  }

  closeGarment() {
    this.setState({ garmentOpen: false, garmentInitialJob: null })
  }

  openSequence() {
    this.actions.resetSequence()
    this.setState({ sequenceOpen: true })
  }

  selectRecent(job, target = this.current.recentPickerTarget) {
    if (!job?.output_url || !target) return
    const image = {
      server: true,
      ref: `${job.id}:output:0`,
      url: job.output_url,
      name: `결과 ${job.id.slice(0, 8)}.png`,
      role: 'output'
    }
    if (target === 'sequenceBase') {
      this.actions.setSequenceBase({ ...image, jobID: job.id, prompt: job.prompt || '' })
      this.actions.setSequencePrompts(this.actions.getSequencePrompts().map((prompt, index) => index === 0 ? (job.prompt || prompt) : prompt))
    } else if (target === 'vision' || target === 'styleReference') {
      this.actions.addKreaRefObjects(target, [image])
    } else if (target === 'identityReference') {
      this.actions.addIdentityReferenceObjects([image])
    } else {
      this.actions.setKreaImage(target, image)
    }
    this.setState({ recentPickerTarget: '' })
  }

  async selectPreset(item, target = this.current.presetPickerTarget) {
    if (!item?.url || !target) return
    try {
      const response = await fetch(item.url)
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      const blob = await response.blob()
      const file = new File([blob], item.filename, { type: blob.type || 'image/webp' })
      file.poseID = item.library === 'pose' ? item.id : ''
      file.posePrompt = item.library === 'pose' ? (item.prompt || item.name || '') : ''
      if (target === 'vision' || target === 'styleReference') this.actions.addKreaRefs(target, [file])
      else if (target === 'identityReference') this.actions.addIdentityReferences([file])
      else this.actions.setKreaImage(target, file)
      this.setState({ presetPickerTarget: '' })
    } catch (cause) {
      this.actions.setError(`프리셋 이미지를 불러오지 못했습니다: ${cause.message}`)
    }
  }

  selectRemote(file, target = this.current.remoteTarget) {
    if (!file || !target) return
    if (target === 'vision' || target === 'styleReference') this.actions.addKreaRefs(target, [file])
    else if (target === 'identityReference') this.actions.addIdentityReferences([file])
    else this.actions.setKreaImage(target, file)
  }

  applyPaintedMask(file, mode = this.current.maskEditorMode) {
    if (mode === 'identity') this.actions.setKreaImage('identityMask', file)
    else if (mode === 'strict') this.actions.setKreaImage('strictMask', file)
    else this.actions.setKreaImage('anypaintMask', file)
    this.setState({ maskEditorMode: '' })
  }

  applyCannyMap(file) {
    this.actions.setKreaImage('nk2e', file)
    this.actions.setNK2EPreprocessed(true)
    this.setState({ cannyEditorOpen: false })
  }

  recentTitle(identityUI, target = this.current.recentPickerTarget) {
    if (target === 'sequenceBase') return '연속 생성 첫 장면 선택'
    if (target === 'identityReference') return `${identityUI.secondary} 선택`
    if (target === 'depth') return '자세·구도 이미지 선택'
    if (target === 'nk2e') return '편집·윤곽 이미지 선택'
    if (target === 'anypaint') return '부분 수정·확장 원본 선택'
    if (target === 'styleReference') return '스타일 참조 이미지 추가'
    if (target === 'vision') return '내용·구도 참조 이미지 추가'
    return `${identityUI.primary} 선택`
  }

  presetTitle(identityUI, target = this.current.presetPickerTarget) {
    if (target === 'identityReference') return `${identityUI.secondary} 프리셋 선택`
    if (target === 'depth') return '자세·구도 프리셋 선택'
    if (target === 'nk2e') return '편집·윤곽 프리셋 선택'
    if (target === 'anypaint') return '부분 수정·확장 원본 프리셋'
    if (target === 'styleReference') return '스타일 참조 프리셋 추가'
    if (target === 'vision') return '내용·구도 참조 프리셋 추가'
    return `${identityUI.primary} 프리셋 선택`
  }

  selectedRef(values, target = this.current.recentPickerTarget) {
    if (target === 'sequenceBase') return values.sequenceBase?.ref || ''
    if (target === 'identityReference') return values.identityReference?.ref || ''
    if (target === 'depth') return values.depth?.ref || ''
    if (target === 'nk2e') return values.nk2e?.ref || ''
    if (target === 'anypaint') return values.anypaint?.ref || ''
    if (target === 'identity') return values.identity?.ref || ''
    return ''
  }
}
