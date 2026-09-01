<script>
  import ImageSequenceModal from './ImageSequenceModal.svelte'
  import RuntimeInfoModal from './RuntimeInfoModal.svelte'
  import MaskEditor from './MaskEditor.svelte'
  import CannyEditor from './CannyEditor.svelte'
  import ImageModal from './ImageModal.svelte'
  import GarmentExtractorModal from './GarmentExtractorModal.svelte'
  import FaceSwapModal from './FaceSwapModal.svelte'
  import RecentImagePicker from './RecentImagePicker.svelte'
  import PresetImagePicker from './PresetImagePicker.svelte'
  import RemoteImageModal from './RemoteImageModal.svelte'
  import { filterPromptPresets, remoteImageTitles } from './lib/catalogs.js'

  export let controller
  export let image = null
  export let garmentOpen = false
  export let garmentInitialJob = null
  export let faceSwapOpen = false
  export let faceSwapInitialJob = null
  export let sequenceOpen = false
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
  export let imageSequenceEntryMode
  export let imageSequenceStoryIdea
  export let imageSequenceSceneCount
  export let imageSequencePrompts
  export let imageSequenceEnhancedPrompts
  export let imageSequenceSceneTitles
  export let imageSequenceSharedPrompt
  export let imageSequenceSharedPromptEdited
  export let imageSequencePlanning
  export let imageSequencePlanError
  export let imageSequenceCharacters
  export let setImageSequenceEntryMode
  export let setImageSequenceStoryIdea
  export let setImageSequenceSceneCount
  export let setImageSequenceSharedPrompt
  export let applyStorySequenceExample
  export let applySceneSequenceExample
  export let applyCharacterSequenceExample
  export let planImageSequence
  export let imageSequenceBlockedMessage
  export let removeImageSequenceScene
  export let moveImageSequenceScene
  export let updateImageSequencePrompt
  export let addImageSequenceScene
  export let addImageSequenceCharacter
  export let removeImageSequenceCharacter
  export let setImageSequenceCharacterName
  export let addImageSequenceCharacterFiles
  export let addImageSequenceCharacterResult
  export let removeImageSequenceCharacterReference
  export let setImageSequenceCharacterReIDReference
  export let toggleImageSequenceCharacterTrait
  export let generateImageSequenceCharacterSheet
  export let approveImageSequenceCharacterSheet
  export let discardImageSequenceCharacterSheet
  export let toggleImageSequenceCharacterTurntableFrame
  export let analyzeImageSequenceCharacter
  export let setImageSequenceCharacterDescription
  export let setImageSequenceCharacterPrompt
  export let imageSequenceCharacterPreview
  export let imageSequenceCharacterReadinessMessage
  export let generateImage
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
  export let submitFaceSwap
</script>

<ImageSequenceModal
  bind:imageSequenceOpen={sequenceOpen}
  {busy}
  {imageForm}
  {kreaOptions}
  {kreaModules}
  {imageSequenceEntryMode}
  {imageSequenceStoryIdea}
  {imageSequenceSceneCount}
  {imageSequencePrompts}
  {imageSequenceEnhancedPrompts}
  {imageSequenceSceneTitles}
  {imageSequenceSharedPrompt}
  {imageSequenceSharedPromptEdited}
  {imageSequencePlanning}
  {imageSequencePlanError}
  {imageSequenceCharacters}
  {imageJobs}
  {setImageSequenceEntryMode}
  {setImageSequenceStoryIdea}
  {setImageSequenceSceneCount}
  {setImageSequenceSharedPrompt}
  {applyStorySequenceExample}
  {applySceneSequenceExample}
  {applyCharacterSequenceExample}
  {planImageSequence}
  {imageSequenceBlockedMessage}
  {removeImageSequenceScene}
  {moveImageSequenceScene}
  {updateImageSequencePrompt}
  {addImageSequenceScene}
  {addImageSequenceCharacter}
  {removeImageSequenceCharacter}
  {setImageSequenceCharacterName}
  {addImageSequenceCharacterFiles}
  {addImageSequenceCharacterResult}
  {removeImageSequenceCharacterReference}
  {setImageSequenceCharacterReIDReference}
  {toggleImageSequenceCharacterTrait}
  {generateImageSequenceCharacterSheet}
  {approveImageSequenceCharacterSheet}
  {discardImageSequenceCharacterSheet}
  {toggleImageSequenceCharacterTurntableFrame}
  {analyzeImageSequenceCharacter}
  {setImageSequenceCharacterDescription}
  {setImageSequenceCharacterPrompt}
  {imageSequenceCharacterPreview}
  {imageSequenceCharacterReadinessMessage}
  {generateImage}
/>

<RuntimeInfoModal bind:open={runtimeInfoOpen} />
<MaskEditor open={Boolean(maskEditorMode)} source={maskEditorMode === 'anypaint' ? kreaAnyPaintPreview : kreaIdentityPreview} existingMask={maskEditorMode === 'anypaint' ? kreaAnyPaintMaskPreview : maskEditorMode === 'identity' ? kreaIdentityMaskPreview : kreaStrictMaskPreview} title={maskEditorMode === 'identity' ? '닮음 집중 영역' : maskEditorMode === 'strict' ? '변경 허용 영역' : '수정 영역 칠하기'} description={maskEditorMode === 'identity' ? '빨간 영역의 Identity 주의를 더 높입니다.' : maskEditorMode === 'strict' ? '빨간 영역만 생성 결과를 쓰고 바깥 픽셀은 원본 그대로 둡니다.' : '빨간 영역을 Krea가 새로 생성합니다.'} outputName={`${maskEditorMode || 'krea'}-mask.png`} onApply={(file) => controller.applyPaintedMask(file, maskEditorMode)} onClose={() => controller.setState({ maskEditorMode: '' })} />
<CannyEditor open={cannyEditorOpen} source={kreaNK2EPreview} preprocessed={kreaNK2EPreprocessed} onApply={(file) => controller.applyCannyMap(file)} onClose={() => controller.setState({ cannyEditorOpen: false })} />
<ImageModal {image} onGarmentExtract={(jobID) => controller.openGarmentFromImage(jobID)} onFaceSwap={(jobID) => controller.openFaceSwapFromImage(jobID)} onClose={() => controller.closeImage()} />
<GarmentExtractorModal open={garmentOpen} jobs={imageJobs} initialJob={garmentInitialJob} onSubmit={submitGarmentExtraction} onClose={() => controller.closeGarment()} />
<FaceSwapModal open={faceSwapOpen} jobs={imageJobs} initialJob={faceSwapInitialJob} onSubmit={submitFaceSwap} onClose={() => controller.closeFaceSwap()} />
<RecentImagePicker open={Boolean(recentPickerTarget)} title={controller.recentTitle(identityUI, recentPickerTarget)} jobs={imageJobs} selectedRef={controller.selectedRef({ identityReference: kreaIdentityReference, depth: kreaDepthImage, nk2e: kreaNK2EImage, anypaint: kreaAnyPaintImage, identity: kreaIdentityImage }, recentPickerTarget)} onSelect={(job) => controller.selectRecent(job, recentPickerTarget)} onClose={() => controller.setState({ recentPickerTarget: '' })} />
<PresetImagePicker open={Boolean(presetPickerTarget)} title={controller.presetTitle(identityUI, presetPickerTarget)} examples={filterPromptPresets} initialTab={presetPickerTarget === 'depth' || presetPickerTarget === 'nk2e' ? 'pose' : 'example'} onSelect={(item) => controller.selectPreset(item, presetPickerTarget)} onClose={() => controller.setState({ presetPickerTarget: '' })} />
<RemoteImageModal open={Boolean(remoteTarget)} title={remoteImageTitles[remoteTarget] || 'URL 이미지 가져오기'} append={remoteTarget === 'vision' || remoteTarget === 'styleReference' || remoteTarget === 'identityReference'} onImport={(file) => controller.selectRemote(file, remoteTarget)} onClose={() => controller.setState({ remoteTarget: '' })} />
