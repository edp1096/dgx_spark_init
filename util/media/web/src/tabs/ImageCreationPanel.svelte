<script>
  import ImageCreationForm from './ImageCreationForm.svelte'
  export let model
  export let cannyEditorOpen
  export let featureModulesOpen
  export let filterPromptPreset
  export let identityPreserveCustom
  export let imageAspectRatio
  export let imageEnhanceEnabled
  export let imageEnhancedPrompt
  export let imageForm
  export let imageMegapixels
  export let imageResolutionMode
  export let kreaOptions
  export let maskEditorMode
  export let parentImageJobID
  export let presetImagePickerTarget
  export let recentImagePickerTarget
  export let remoteImageTarget
  export let runtimeInfoOpen
</script>

<ImageCreationForm
  {...model}
  bind:cannyEditorOpen
  bind:featureModulesOpen
  bind:filterPromptPreset
  bind:identityPreserveCustom
  bind:imageAspectRatio
  bind:imageEnhanceEnabled
  bind:imageEnhancedPrompt
  bind:imageForm
  bind:imageMegapixels
  bind:imageResolutionMode
  bind:kreaOptions
  bind:maskEditorMode
  bind:parentImageJobID
  bind:presetImagePickerTarget
  bind:recentImagePickerTarget
  bind:remoteImageTarget
  bind:runtimeInfoOpen
/>
