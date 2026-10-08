import { writable } from 'svelte/store'

export class CommonModalController {
  constructor(actions) {
    this.actions = actions
    this.current = {
      prompt: null,
      promptExamplesOpen: false,
      promptExamplesTarget: 'image'
    }
    this.state = writable(this.current)
    this.state.subscribe((value) => this.current = value)
  }

  setState(patch) {
    this.state.update((value) => ({ ...value, ...patch }))
  }

  showPrompt(title, detail, text) {
    this.setState({ prompt: { title, detail, text } })
  }

  closePrompt() {
    this.setState({ prompt: null })
  }

  openPromptExamples(target = 'image') {
    this.setState({ promptExamplesTarget: target, promptExamplesOpen: true })
  }

  closePromptExamples() {
    this.setState({ promptExamplesOpen: false })
  }

  applyPromptExample(preset, mode) {
    if (!preset) return
    if (this.current.promptExamplesTarget === 'video') this.actions.applyVideoPrompt(preset, mode)
    else this.actions.applyImagePrompt(preset, mode)
    this.closePromptExamples()
  }
}
