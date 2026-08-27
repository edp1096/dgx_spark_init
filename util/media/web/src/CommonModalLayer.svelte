<script>
  import PromptModal from './PromptModal.svelte'
  import PromptExamplesModal from './PromptExamplesModal.svelte'
  import { ltxPromptPresets, ltxPromptSources } from './ltxPromptPresets.js'
  import {
    filterPromptPresets,
    filterPromptSource,
    kreaPromptGuideSource,
    vibePromptGuideSource,
    wildcardPromptSource
  } from './lib/catalogs.js'

  export let controller
  export let prompt = null
  export let promptExamplesOpen = false
  export let promptExamplesTarget = 'image'
  export let imageSelectedID = ''
  export let videoSelectedID = ''
  export let videoHasConditionImage = false
  export let createRandomVideoPrompt
</script>

<PromptModal {prompt} onClose={() => controller.closePrompt()} />
<PromptExamplesModal
  open={promptExamplesOpen}
  examples={promptExamplesTarget === 'video' ? ltxPromptPresets : filterPromptPresets}
  selectedID={promptExamplesTarget === 'video' ? videoSelectedID : imageSelectedID}
  officialSource={kreaPromptGuideSource}
  communitySource={filterPromptSource}
  vibeSource={vibePromptGuideSource}
  wildcardSource={wildcardPromptSource}
  sourceLinks={promptExamplesTarget === 'video' ? ltxPromptSources : []}
  showWildcard
  wildcardMode={promptExamplesTarget}
  wildcardDisabled={promptExamplesTarget === 'video' && videoHasConditionImage}
  wildcardDisabledTitle="장면 이미지가 있을 때는 위의 프롬프트 만들기를 사용하세요."
  onRandomVideo={createRandomVideoPrompt}
  onApply={(preset, mode) => controller.applyPromptExample(preset, mode)}
  onClose={() => controller.closePromptExamples()}
/>
