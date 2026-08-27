<script>
  import VideoModal from './VideoModal.svelte'
  import SubtitleModal from './SubtitleModal.svelte'
  import SubtitleRegenerateModal from './SubtitleRegenerateModal.svelte'
  import VideoFramePicker from './VideoFramePicker.svelte'
  import VideoUpscaleModal from './VideoUpscaleModal.svelte'
  import AudioModal from './AudioModal.svelte'
  import RecentVideoPicker from './RecentVideoPicker.svelte'
  import RecentAudioPicker from './RecentAudioPicker.svelte'
  import VideoTimelineEditor from './VideoTimelineEditor.svelte'
  import RecentImagePicker from './RecentImagePicker.svelte'
  import RemoteImageModal from './RemoteImageModal.svelte'
  import { framesForDuration } from './lib/videoTiming.js'

  export let controller
  export let video = null
  export let subtitle = null
  export let subtitleRegenerateJob = null
  export let regeneratingSubtitleJob = ''
  export let audio = null
  export let recognitionVideoPickerOpen = false
  export let audioPickerOpen = false
  export let timelineEditorOpen = false
  export let imagePickerTarget = ''
  export let remoteImageTarget = ''
  export let framePickerSource = null
  export let upscaleSource = null
  export let upscaleBusy = false
  export let videoJobs
  export let recognitionJobs
  export let speechJobs
  export let imageJobs
  export let recognitionSourceVideoJob
  export let videoAudioClips
  export let videoDurationSeconds
  export let videoForm
  export let videoImage
  export let videoEndImage
  export let videoEndStrength
  export let videoKeyframes
  export let usePickedVideoFrame
  export let togglePickedVideoAudio
  export let moveVideoKeyframe
  export let moveVideoAudio
  export let updateVideoKeyframe
  export let removeVideoKeyframe
  export let addVideoKeyframe
  export let setVideoConditionImage
  export let videoImagePreview
</script>

<VideoModal {video} onClose={() => controller.setState({ video: null })} onSelectFrames={(id) => controller.openFramePicker(videoJobs.find((job) => job.id === id))} onUpscale={(id) => controller.openUpscale(videoJobs.find((job) => job.id === id))} onTranscribe={(id) => controller.sendVideoToRecognition(videoJobs.find((job) => job.id === id))} onLoadSettings={(id) => controller.loadVideoSettings(videoJobs.find((job) => job.id === id))} />
<SubtitleModal result={subtitle} onClose={() => controller.setState({ subtitle: null })} onSelectFrames={(id) => controller.openFramePicker(recognitionJobs.find((job) => job.id === id))} onUpscale={(id) => controller.openUpscale(recognitionJobs.find((job) => job.id === id))} onRegenerate={(id) => controller.openSubtitleRegenerate(recognitionJobs.find((job) => job.id === id))} />
<SubtitleRegenerateModal job={subtitleRegenerateJob} busy={Boolean(regeneratingSubtitleJob)} onSubmit={(options) => controller.regenerateSubtitle(options)} onClose={() => controller.setState({ subtitleRegenerateJob: null })} />
<VideoFramePicker source={framePickerSource} onUse={usePickedVideoFrame} onClose={() => controller.setState({ framePickerSource: null })} />
<VideoUpscaleModal source={upscaleSource} busy={upscaleBusy} onSubmit={(options) => controller.submitUpscale(options)} onClose={() => controller.setState({ upscaleSource: null })} />
<AudioModal {audio} onClose={() => controller.setState({ audio: null })} onA2V={(id) => controller.sendAudioToVideo(speechJobs.find((job) => job.id === id))} />
<RecentVideoPicker open={recognitionVideoPickerOpen} jobs={videoJobs} selectedID={recognitionSourceVideoJob?.id || ''} onSelect={(job) => controller.sendVideoToRecognition(job)} onClose={() => controller.setState({ recognitionVideoPickerOpen: false })} />
<RecentAudioPicker open={audioPickerOpen} jobs={speechJobs} selectedIDs={videoAudioClips.map((clip) => clip.job.id)} multiple onSelect={togglePickedVideoAudio} onClose={() => controller.setState({ audioPickerOpen: false })} />
<VideoTimelineEditor open={timelineEditorOpen} overlayOpen={Boolean(imagePickerTarget || remoteImageTarget)} duration={(framesForDuration(videoDurationSeconds, videoForm.fps) - 1) / Number(videoForm.fps || 1)} fps={videoForm.fps} startImage={videoImage} endImage={videoEndImage} startStrength={videoForm.image_strength} endStrength={videoEndStrength} keyframes={videoKeyframes} audioClips={videoAudioClips} imageURL={videoImagePreview} onMove={moveVideoKeyframe} onMoveAudio={moveVideoAudio} onUpdate={updateVideoKeyframe} onRemove={removeVideoKeyframe} onAdd={addVideoKeyframe} onSetStrength={(target, value) => { if (target === 'start') videoForm.image_strength = Number(value); else videoEndStrength = Number(value) }} onFile={setVideoConditionImage} onRecent={(target) => controller.setState({ imagePickerTarget: target })} onRemote={(target) => controller.setState({ remoteImageTarget: target })} onClear={(target) => setVideoConditionImage(target, null)} onClose={() => controller.setState({ timelineEditorOpen: false })} />
<RecentImagePicker open={Boolean(imagePickerTarget)} title={controller.conditionTitle(imagePickerTarget)} jobs={imageJobs} selectedRef="" onSelect={(job) => controller.selectRecentImage(job, imagePickerTarget)} onClose={() => controller.setState({ imagePickerTarget: '' })} />
<RemoteImageModal open={Boolean(remoteImageTarget)} title={controller.conditionTitle(remoteImageTarget, ' URL 가져오기')} onImport={(file) => controller.selectRemoteImage(file, remoteImageTarget)} onClose={() => controller.setState({ remoteImageTarget: '' })} />
