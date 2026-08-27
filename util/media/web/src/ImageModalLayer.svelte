<script>
  import ImageSequenceModal from './ImageSequenceModal.svelte'
  import ImageSequenceRegionModal from './ImageSequenceRegionModal.svelte'
  import RuntimeInfoModal from './RuntimeInfoModal.svelte'
  import MaskEditor from './MaskEditor.svelte'
  import CannyEditor from './CannyEditor.svelte'
  import ImageModal from './ImageModal.svelte'
  import GarmentExtractorModal from './GarmentExtractorModal.svelte'
  import RecentImagePicker from './RecentImagePicker.svelte'
  import PresetImagePicker from './PresetImagePicker.svelte'
  import RemoteImageModal from './RemoteImageModal.svelte'
  import { filterPromptPresets, remoteImageTitles } from './lib/catalogs.js'

  export let controller
  export let image = null
  export let garmentOpen = false
  export let garmentInitialJob = null
  export let sequenceOpen = false
  export let sequenceMaskEditorIndex = -1
  export let sequenceRegionPicker = -1
  export let maskEditorMode = ''
  export let cannyEditorOpen = false
  export let runtimeInfoOpen = false
  export let recentPickerTarget = ''
  export let presetPickerTarget = ''
  export let remoteTarget = ''
  export let busy
  export let imageForm
  export let kreaOptions
  export let kreaModules
  export let imageSequenceBase
  export let imageSequenceStrength
  export let setImageSequenceBase
  export let setImageSequenceStrength
  export let imageSequencePrompts
  export let imageSequenceMaskPreviews
  export let imageSequenceRegions
  export let applyRobotSequenceExample
  export let clearImageSequenceMasks
  export let imageSequenceBlockedMessage
  export let removeImageSequenceScene
  export let updateImageSequencePrompt
  export let updateImageSequenceRegion
  export let imageSequenceRegionOption
  export let addImageSequenceScene
  export let generateImage
  export let useImageSequenceMask
  export let kreaAnyPaintPreview
  export let kreaIdentityPreview
  export let kreaAnyPaintMaskPreview
  export let kreaIdentityMaskPreview
  export let kreaStrictMaskPreview
  export let kreaNK2EPreview
  export let kreaNK2EPreprocessed
  export let imageJobs
  export let identityUI
  export let kreaIdentityReference
  export let kreaDepthImage
  export let kreaNK2EImage
  export let kreaAnyPaintImage
  export let kreaIdentityImage
  export let submitGarmentExtraction
</script>

<ImageSequenceModal
  bind:imageSequenceOpen={sequenceOpen}
  {busy}
  {imageForm}
  {kreaOptions}
  {kreaModules}
  {imageSequenceBase}
  {imageSequenceStrength}
  {setImageSequenceBase}
  {setImageSequenceStrength}
  {imageSequencePrompts}
  {imageSequenceMaskPreviews}
  {imageSequenceRegions}
  bind:imageSequenceMaskEditorIndex={sequenceMaskEditorIndex}
  bind:imageSequenceRegionPicker={sequenceRegionPicker}
  bind:recentImagePickerTarget={recentPickerTarget}
  {applyRobotSequenceExample}
  {clearImageSequenceMasks}
  {imageSequenceBlockedMessage}
  {removeImageSequenceScene}
  {updateImageSequencePrompt}
  {imageSequenceRegionOption}
  {addImageSequenceScene}
  {generateImage}
/>

<ImageSequenceRegionModal bind:openIndex={sequenceRegionPicker} regions={imageSequenceRegions} onSelect={updateImageSequenceRegion} />
<RuntimeInfoModal bind:open={runtimeInfoOpen} />
<MaskEditor open={Boolean(maskEditorMode)} source={maskEditorMode === 'anypaint' ? kreaAnyPaintPreview : kreaIdentityPreview} existingMask={maskEditorMode === 'anypaint' ? kreaAnyPaintMaskPreview : maskEditorMode === 'identity' ? kreaIdentityMaskPreview : kreaStrictMaskPreview} title={maskEditorMode === 'identity' ? '닮음 집중 영역' : maskEditorMode === 'strict' ? '변경 허용 영역' : '수정 영역 칠하기'} description={maskEditorMode === 'identity' ? '빨간 영역의 Identity 주의를 더 높입니다.' : maskEditorMode === 'strict' ? '빨간 영역만 생성 결과를 쓰고 바깥 픽셀은 원본 그대로 둡니다.' : '빨간 영역을 Krea가 새로 생성합니다.'} outputName={`${maskEditorMode || 'krea'}-mask.png`} onApply={(file) => controller.applyPaintedMask(file, maskEditorMode)} onClose={() => controller.setState({ maskEditorMode: '' })} />
<MaskEditor open={sequenceMaskEditorIndex >= 1} source={imageSequenceBase?.url || ''} existingMask={imageSequenceMaskPreviews[sequenceMaskEditorIndex] || ''} title={`장면 ${sequenceMaskEditorIndex + 1} 변경 허용 영역`} description="빨간 영역만 다음 장면에서 새로 그립니다. 움직이기 전 위치와 이동할 위치를 함께 넉넉히 칠하세요." outputName={`sequence-scene-${sequenceMaskEditorIndex + 1}-mask.png`} onApply={useImageSequenceMask} onClose={() => controller.setState({ sequenceMaskEditorIndex: -1 })} />
<CannyEditor open={cannyEditorOpen} source={kreaNK2EPreview} preprocessed={kreaNK2EPreprocessed} onApply={(file) => controller.applyCannyMap(file)} onClose={() => controller.setState({ cannyEditorOpen: false })} />
<ImageModal {image} onGarmentExtract={(jobID) => controller.openGarmentFromImage(jobID)} onClose={() => controller.closeImage()} />
<GarmentExtractorModal open={garmentOpen} jobs={imageJobs} initialJob={garmentInitialJob} onSubmit={submitGarmentExtraction} onClose={() => controller.closeGarment()} />
<RecentImagePicker open={Boolean(recentPickerTarget)} title={controller.recentTitle(identityUI, recentPickerTarget)} jobs={imageJobs} selectedRef={controller.selectedRef({ sequenceBase: imageSequenceBase, identityReference: kreaIdentityReference, depth: kreaDepthImage, nk2e: kreaNK2EImage, anypaint: kreaAnyPaintImage, identity: kreaIdentityImage }, recentPickerTarget)} onSelect={(job) => controller.selectRecent(job, recentPickerTarget)} onClose={() => controller.setState({ recentPickerTarget: '' })} />
<PresetImagePicker open={Boolean(presetPickerTarget)} title={controller.presetTitle(identityUI, presetPickerTarget)} examples={filterPromptPresets} initialTab={presetPickerTarget === 'depth' || presetPickerTarget === 'nk2e' ? 'pose' : 'example'} onSelect={(item) => controller.selectPreset(item, presetPickerTarget)} onClose={() => controller.setState({ presetPickerTarget: '' })} />
<RemoteImageModal open={Boolean(remoteTarget)} title={remoteImageTitles[remoteTarget] || 'URL 이미지 가져오기'} append={remoteTarget === 'vision' || remoteTarget === 'styleReference' || remoteTarget === 'identityReference'} onImport={(file) => controller.selectRemote(file, remoteTarget)} onClose={() => controller.setState({ remoteTarget: '' })} />
