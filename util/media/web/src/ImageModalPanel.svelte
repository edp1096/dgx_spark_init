<script>
  import ImageModalLayer from './ImageModalLayer.svelte'
  export let controller
  export let modalState
  export let model
</script>

<ImageModalLayer
  {...model}
  {controller}
  bind:image={$modalState.image}
  bind:garmentOpen={$modalState.garmentOpen}
  bind:garmentInitialJob={$modalState.garmentInitialJob}
  bind:sequenceOpen={$modalState.sequenceOpen}
  bind:sequenceMaskEditorIndex={$modalState.sequenceMaskEditorIndex}
  bind:sequenceRegionPicker={$modalState.sequenceRegionPicker}
  bind:maskEditorMode={$modalState.maskEditorMode}
  bind:cannyEditorOpen={$modalState.cannyEditorOpen}
  bind:runtimeInfoOpen={$modalState.runtimeInfoOpen}
  bind:recentPickerTarget={$modalState.recentPickerTarget}
  bind:presetPickerTarget={$modalState.presetPickerTarget}
  bind:remoteTarget={$modalState.remoteTarget}
/>
