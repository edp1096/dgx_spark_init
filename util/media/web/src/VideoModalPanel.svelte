<script>
  import VideoModalLayer from './VideoModalLayer.svelte'
  export let controller
  export let modalState
  export let model
  export let videoForm
  export let videoEndStrength
</script>

<VideoModalLayer
  {...model}
  {controller}
  bind:video={$modalState.video}
  bind:subtitle={$modalState.subtitle}
  bind:subtitleRegenerateJob={$modalState.subtitleRegenerateJob}
  bind:regeneratingSubtitleJob={$modalState.regeneratingSubtitleJob}
  bind:audio={$modalState.audio}
  bind:recognitionVideoPickerOpen={$modalState.recognitionVideoPickerOpen}
  bind:audioPickerOpen={$modalState.audioPickerOpen}
  bind:timelineEditorOpen={$modalState.timelineEditorOpen}
  bind:imagePickerTarget={$modalState.imagePickerTarget}
  bind:remoteImageTarget={$modalState.remoteImageTarget}
  bind:framePickerSource={$modalState.framePickerSource}
  bind:upscaleSource={$modalState.upscaleSource}
  bind:upscaleBusy={$modalState.upscaleBusy}
  bind:videoForm
  bind:videoEndStrength
/>
