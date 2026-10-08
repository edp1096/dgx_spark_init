<script>
  import VideoCreationForm from './VideoCreationForm.svelte'
  export let model
  export let videoAdvancedOpen
  export let videoAudioPickerOpen
  export let videoDurationSeconds
  export let videoEndStrength
  export let videoEnhanceEnabled
  export let videoEnhancedPrompt
  export let videoForm
  export let videoImagePickerTarget
  export let videoPromptPreset
  export let videoRemoteImageTarget
  export let videoTimelineEditorOpen
</script>

<VideoCreationForm
  {...model}
  bind:videoAdvancedOpen
  bind:videoAudioPickerOpen
  bind:videoDurationSeconds
  bind:videoEndStrength
  bind:videoEnhanceEnabled
  bind:videoEnhancedPrompt
  bind:videoForm
  bind:videoImagePickerTarget
  bind:videoPromptPreset
  bind:videoRemoteImageTarget
  bind:videoTimelineEditorOpen
/>
