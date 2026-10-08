<script>
  import ImageResults from './ImageResults.svelte'
  import VideoResults from './VideoResults.svelte'
  import RecognitionResults from './RecognitionResults.svelte'

  export let kind
  export let model
</script>

{#if kind === 'image'}
  <ImageResults {...model} />
{:else if kind === 'video'}
  <VideoResults {...model} />
{:else if kind === 'recognition'}
  <RecognitionResults {...model} />
{/if}
