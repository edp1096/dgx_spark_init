<script>
  import { onMount, onDestroy } from 'svelte'
  import { api } from './api.js'
  import LoraStudio from './LoraStudio.svelte'
  import AssistantChat from './AssistantChat.svelte'
  import AppHeader from './AppHeader.svelte'
  import AppNavigation from './AppNavigation.svelte'
  import ImageModalPanel from './ImageModalPanel.svelte'
  import VideoModalPanel from './VideoModalPanel.svelte'
  import CommonModalLayer from './CommonModalLayer.svelte'
  import HistoryTab from './tabs/HistoryTab.svelte'
  import SpeechTab from './tabs/SpeechTab.svelte'
  import MediaResultsLayer from './tabs/MediaResultsLayer.svelte'
  import RecognitionForm from './tabs/RecognitionForm.svelte'
  import SettingsTab from './tabs/SettingsTab.svelte'
  import ImageCreationPanel from './tabs/ImageCreationPanel.svelte'
  import VideoCreationPanel from './tabs/VideoCreationPanel.svelte'
  import { identityPreserveCatalog, defaultIdentityPreserveItems, checkpointDisplayChoices, pageSizeOptions, imagePageSizeOptions, engineStatusCatalog, imageModeChoices, kreaModuleLabels, identityPresetUI, imageAspectRatios, kindLabels, statusLabels, translationLanguages, kreaStyleCatalog, videoResolutionPresets } from './lib/catalogs.js'
  import { durationFromFrames, framesForDuration } from './lib/videoTiming.js'
  import { formatBytes, subtitleTranslationWarningText, imagePromptModalText, videoPromptModalText } from './lib/mediaPresentation.js'
  import { createVideoVisualContext } from './lib/assistantVisualContext.js'
  import { ImageModalController } from './lib/imageModalController.js'
  import { VideoModalController } from './lib/videoModalController.js'
  import { CommonModalController } from './lib/commonModalController.js'
  import { JobController } from './lib/jobController.js'
  import { imageEnhancementActive as canEnhanceImagePrompt, imageEnhancementCurrent as isCurrentImageEnhancement, isPureOutpaint as checkPureOutpaint } from './lib/imageWorkflow.js'
  import { videoAccelerationPreview as describeVideoAcceleration, videoConditioningDisabledReason as disabledVideoConditioningReason, videoEnhancementActive as canEnhanceVideoPrompt, videoEnhancementCurrent as isCurrentVideoEnhancement, videoResolutionPresetID, videoStage2TokenCount as calculateVideoStage2TokenCount } from './lib/videoWorkflow.js'
  import { mediaInputPreview } from './lib/mediaInputs.js'
  import { ImageInputController } from './lib/imageInputController.js'
  import { MediaTransferController } from './lib/mediaTransferController.js'
  import { SettingsController } from './lib/settingsController.js'
  import { AssistantController } from './lib/assistantController.js'
  import { ImageSequenceController, imageSequenceBlockedMessage as sequenceBlockedMessage } from './lib/imageSequenceController.js'
  import { VideoTimelineController } from './lib/videoTimelineController.js'
  import { AppLifecycleController, runtimeDefaults } from './lib/appLifecycleController.js'
  import { ImageResultController } from './lib/imageResultController.js'
  import { ImageCreationController } from './lib/imageCreationController.js'
  import { VideoGenerationController } from './lib/videoGenerationController.js'
  import { SpeechRecognitionController } from './lib/speechRecognitionController.js'
  import { MediaListController } from './lib/mediaListController.js'
  import { ImageConfigurationController } from './lib/imageConfigurationController.js'
  import { AppPresentationController } from './lib/appPresentationController.js'
  function imageGenerationProgress(job) { return appPresentationController.imageProgress(job) }
  function videoGenerationProgress(job) { return appPresentationController.videoProgress(job) }
  function speechGenerationProgress(job) { return appPresentationController.speechProgress(job) }
  function recognitionProgressText(job) { return appPresentationController.recognitionText(job) }
  function recognitionProgressTiming(job) { return appPresentationController.recognitionTiming(job) }
  function recognitionProgressPercent(job) { return appPresentationController.recognitionPercent(job) }


  let tab = 'image'
  let colorTheme = document.documentElement.dataset.theme === 'light' ? 'light' : 'dark'
  let config = null
  let settings = null
  let savedMessage = ''
  let settingsSection = 'connection'
  let jobs = []
  let engineStates = { image: 'offline', speech: 'offline', recognition: 'offline', video: 'offline', prompt: 'offline', media: 'offline', trainer: 'offline', upscale: 'offline', garment: 'offline', faceswap: 'offline' }
  let busy = false
  let error = ''
  const imageInputController = new ImageInputController()
  const imageInputState = imageInputController.state
  const imageSequenceController = new ImageSequenceController()
  const imageSequenceState = imageSequenceController.state
  $: refs = $imageInputState.refs
  $: kreaIdentityImage = $imageInputState.identityImage
  $: kreaIdentityReferences = $imageInputState.identityReferences
  $: kreaDepthImage = $imageInputState.depthImage
  $: kreaNK2EImage = $imageInputState.nk2eImage
  $: kreaAnyPaintImage = $imageInputState.anypaintImage
  $: kreaAnyPaintMask = $imageInputState.anypaintMask
  $: kreaIdentityMask = $imageInputState.identityMask
  $: kreaStrictMask = $imageInputState.strictMask
  $: kreaVisionImages = $imageInputState.visionImages
  $: kreaStyleReferenceImages = $imageInputState.styleReferenceImages
  let imageForm = { prompt: '', width: 1024, height: 1024, seed: -1, mode: 'create' }
  let imageEnhanceEnabled = true
  let imageEnhancedPrompt = ''
  let imageEnhancedSource = ''
  let imageDisabledMessage = ''
  let imageEnhancementIsActive = false
  let imageEnhancementIsCurrent = false
  let imageResolutionMode = 'smart'
  let imageAspectRatio = '1:1'
  let imageMegapixels = 1
  let filterPromptPreset = ''
  $: imageSequencePrompts = $imageSequenceState.prompts
  $: imageSequenceEntryMode = $imageSequenceState.entryMode
  $: imageSequenceStoryIdea = $imageSequenceState.storyIdea
  $: imageSequenceSceneCount = $imageSequenceState.sceneCount
  $: imageSequenceEnhancedPrompts = $imageSequenceState.enhancedPrompts
  $: imageSequenceSceneTitles = $imageSequenceState.sceneTitles
  $: imageSequenceSharedPrompt = $imageSequenceState.sharedPrompt
  $: imageSequenceSharedPromptEdited = $imageSequenceState.sharedPromptEdited
  $: imageSequencePlanning = $imageSequenceState.planning
  $: imageSequencePlanError = $imageSequenceState.planError
  $: imageSequenceCharacters = $imageSequenceState.characters
  let kreaModules = { identity: false, depth: false, style: false, userLora: false, vision: false, styleReference: false, nk2e: false, anypaint: false }
  let kreaNK2EPreprocessed = false
  let parentImageJobID = ''
  let identityPreserveItems = [...defaultIdentityPreserveItems]
  let identityPreserveCustom = ''
  let identityPreset = ''
  let depthPoseID = ''
  let nk2ePoseID = ''
  let activeKreaModuleLabels = []
  let kreaModuleMessage = ''

  onDestroy(() => {
    videoTimelineController.destroy()
    imageInputController.destroy()
    imageSequenceController.destroy()
    imageResultController.destroy()
  })
  $: kreaIdentityReference = kreaIdentityReferences[0] || null
  $: kreaIdentityPreview = mediaInputPreview(kreaIdentityImage)
  $: kreaIdentityReferencePreview = mediaInputPreview(kreaIdentityReference)
  $: kreaDepthPreview = mediaInputPreview(kreaDepthImage)
  $: kreaNK2EPreview = mediaInputPreview(kreaNK2EImage)
  $: kreaAnyPaintPreview = mediaInputPreview(kreaAnyPaintImage)
  $: kreaAnyPaintMaskPreview = mediaInputPreview(kreaAnyPaintMask)
  $: kreaIdentityMaskPreview = mediaInputPreview(kreaIdentityMask)
  $: kreaStrictMaskPreview = mediaInputPreview(kreaStrictMask)
  let kreaStyleSelections = [{ name: 'retroanime', strength: 1 }]
  let userLoraCatalog = []
  let userLoraSelections = []
  let kreaOptions = {
    checkpoint: 'official',
    identity_strength: 1, ref_boost: 4, source_ref_boost: 1, grounding_px: 768, steps: 8,
    identity_model: 'convrot', identity_encoder: 'heretic',
    sampling_preset: 'default',
    depth_strength: 0.8,
    vision_mode: 'descriptor', vision_megapixels: 1, style_reference_strength: 1,
    nk2e_mode: 'edit', nk2e_strength: 0.7, vae_mode: 'default', identity_fit_mode: 'fit',
    strict_mask_grow: 0, strict_mask_feather: 0,
    outpaint_left: 0, outpaint_top: 0, outpaint_right: 0, outpaint_bottom: 0,
    anypaint_strength: 1, anypaint_boundary_redraw_px: 32,
    filter_mode: 'balanced', filter_strength: 1,
    prompt_enhancer: false, prompt_enhancer_strength: 1, prompt_text_scale: 1.75
  }
  let speechForm = { text: '', instructions: '', language: 'Korean', speaker: 'Sohee', seed: -1 }
  let recognitionForm = {
    source: 'url', url: '', language: 'Auto', context: '',
    output_formats: ['srt', 'txt'], translation_mode: 'none', target_language: 'Korean',
    media_part: '', media_source: ''
  }
  let recognitionFile = null
  let recognitionSourceVideoJob = null
  let recognitionFileInput
  let recognitionOptions = null
  let loadingRecognitionOptions = false
  let videoForm = { prompt: '', width: 768, height: 512, fps: 24, seed: -1, image_strength: 1 }
  let videoDurationSeconds = 5
  let settingsVideoDurationSeconds = 5
  let videoImage = null
  let videoEndImage = null
  let videoEndStrength = 1
  let videoKeyframes = []
  let videoAudioClips = []
  let videoAudioJob = null
  let nextVideoKeyframeID = 1
  let nextVideoAudioClipID = 1

  $: videoAudioJob = videoAudioClips[0]?.job || null
  let videoEnhanceEnabled = true
  let videoEnhancedPrompt = ''
  let videoEnhancedSource = ''
  let videoEnhancedImageKey = ''
  let videoEnhancementIsActive = false
  let videoEnhancementIsCurrent = false
  let creatingVideoPrompt = false
  let videoPromptCreationMessage = ''
  let videoPromptPreset = ''
  let videoAdvancedOpen = false
  let enhancingPrompt = false
  let deletingJob = ''
  let cancellingJob = ''
  let retryingJob = ''
  let hfToken = ''
  let civitaiToken = ''
  let checkpointSelection = {
    'ray-v1': true, 'ray-v2': true, 'ray-v3': true, 'ray-v4': true,
    'moody-v7': true, 'moody-cutie-v4': true, 'moody-amateur-v1': true,
    'chriscole-edit-v1.1': true
  }
  let nvfp4Selection = { 'ray-v2': true, 'ray-v4': true }
  let removeBF16Sources = false
  let subtitleView = 'gallery'
  let imageView = 'gallery'
  let speechView = 'gallery'
  let mobileImagePane = 'create'
  let mobileVideoPane = 'create'
  let mobileSpeechPane = 'create'
  let mobileRecognitionPane = 'create'
  let videoView = 'gallery'
  let progressClock = Date.now()
  let listPageSizes = { image: 8, video: 8, speech: 10, recognition: 10, history: 20 }
  let listPages = { image: 1, video: 1, speech: 1, recognition: 1, history: 1 }
  let listSortOrders = { image: 'desc', video: 'desc', speech: 'desc', recognition: 'desc', history: 'desc' }
  let listTagFilters = { image: [], video: [], speech: [], recognition: [] }
  let listTagExclusions = { image: [], video: [], speech: [], recognition: [] }
  let listTagUntaggedOnly = { image: false, video: false, speech: false, recognition: false }
  let listTagMatchModes = { image: 'or', video: 'or', speech: 'or', recognition: 'or' }
  let mobileEngineOpen = false

  let monitoredEngineStatuses = []
  let engineAggregate = 'down'
  let engineAggregateLabel = 'API 확인 중'
  $: enginePresentation = (engineStates, appPresentationController.enginePresentation())
  $: monitoredEngineStatuses = enginePresentation.statuses
  $: {
    engineAggregate = enginePresentation.aggregate
    engineAggregateLabel = enginePresentation.label
  }
  $: identityUI = identityPresetUI[identityPreset] || identityPresetUI['']
  function snapVideoDuration(seconds, fps = videoForm.fps) {
    return durationFromFrames(framesForDuration(seconds, fps), fps)
  }

  function currentVideoResolutionPreset() {
    return videoResolutionPresetID(videoForm, videoResolutionPresets)
  }

  function applyVideoResolutionPreset(preset) {
    if (!preset) return
    videoForm = { ...videoForm, width: preset.width, height: preset.height }
  }

  function videoStage2TokenCount(width = videoForm.width, height = videoForm.height, seconds = videoDurationSeconds, fps = videoForm.fps) {
    return calculateVideoStage2TokenCount(width, height, seconds, fps)
  }

  function videoAccelerationPreview() {
    return describeVideoAcceleration({ acceleration: config?.video?.acceleration, tokens: videoStage2TokenCount() })
  }

  const activeJobs = () => appPresentationController.activeJobs()

  let imageJobs = []
  let videoJobs = []
  let speechJobs = []
  let recognitionJobs = []
  let pagedImageJobs = []
  let pagedVideoJobs = []
  let pagedSpeechJobs = []
  let pagedRecognitionJobs = []
  let pagedHistoryJobs = []
  const appPresentationController = new AppPresentationController({ engineCatalog: engineStatusCatalog, actions: { getState: appPresentationSnapshot } })
  const mediaListController = new MediaListController({
    storage: localStorage,
    pageSizeOptions,
    imagePageSizeOptions,
    actions: { getState: mediaListSnapshot, patch: patchMediaListState }
  })
  const jobController = new JobController({
    api,
    getError: () => error,
    setError: (value) => error = value
  })
  const jobState = jobController.state
  const imageCreationController = new ImageCreationController({
    api,
    catalogs: { identityPreserveCatalog, defaultIdentityPreserveItems, identityPresetUI, imageAspectRatios, imageModeChoices },
    actions: {
      getState: imageCreationSnapshot,
      patch: patchImageCreationState,
      setKreaImage,
      clearAllInputs: clearAllImageInputs,
      clearGeneratedInputs,
      resetSequence: () => imageSequenceController.reset(),
      planSequence: () => imageSequenceController.plan(api),
      closeSequence: () => $imageModalState.sequenceOpen = false,
      clearCloneMessage: () => imageResultController.setState({ cloneMessage: '' }),
      addReferences: addRefObjects,
      addIdentityReferences: addIdentityReferenceObjects,
      addKreaReferences: addKreaRefObjects,
      filterModeDefault,
      showNewest: () => showNewestListPage('image'),
      refresh,
      scrollTop: () => window.scrollTo({ top: 0, behavior: 'smooth' })
    }
  })
  const imageConfigurationController = new ImageConfigurationController({
    api,
    catalogs: { kreaStyleCatalog, checkpointDisplayChoices },
    actions: {
      getState: imageConfigurationSnapshot,
      patch: patchImageConfigurationState,
      setPreserveItems: (items) => imageCreationController.setPreserveItems(items),
      applySmartResolution: () => imageCreationController.applySmartResolution(),
      setMessage: (value) => imageResultController.setState({ cloneMessage: value }),
      clearKreaRefs: (kind) => imageInputController.clearKreaRefs(kind),
      setKreaImage
    }
  })
  const imageResultController = new ImageResultController({
    api,
    actions: {
      setError: (value) => error = value,
      showNewest: () => showNewestListPage('image'),
      refresh,
      showResults: () => mobileImagePane = 'results',
      showCreate: () => mobileImagePane = 'create',
      clearParentJob: () => parentImageJobID = '',
      clonePrompt: (job) => cloneImagePrompt(job),
      cloneSettings: (job) => cloneImageSettings(job),
      cloneReferences: (job) => cloneImageReferences(job),
      scrollTop: () => window.scrollTo({ top: 0, behavior: 'smooth' })
    }
  })
  const imageResultState = imageResultController.state
  const settingsController = new SettingsController({
    api,
    setError: (value) => error = value,
    setMessage: (value) => savedMessage = value,
    setBusy: (value) => busy = value
  })
  const settingsState = settingsController.state
  const imageModalController = new ImageModalController({
    getImageJobs: () => imageJobs,
    resetSequence: () => imageSequenceController.reset([rawImagePrompt(), '']),
    addKreaRefObjects,
    addIdentityReferenceObjects,
    addKreaRefs,
    addIdentityReferences,
    setKreaImage,
    setNK2EPreprocessed: (value) => kreaNK2EPreprocessed = value,
    setError: (value) => error = value
  })
  const imageModalState = imageModalController.state
  const videoModalController = new VideoModalController({
    getJobs: () => jobs,
    getRecognitionJobs: () => recognitionJobs,
    getVideoKeyframes: () => videoKeyframes,
    setVideoConditionImage,
    setError: (value) => error = value,
    regenerateSubtitle: async (jobID, options) => {
      await api.regenerateSubtitle(jobID, options)
      await refresh()
    },
    submitUpscale: async (jobID, options) => {
      await api.upscaleVideo(jobID, options)
      tab = 'video'
      mobileVideoPane = 'results'
      await refresh()
    },
    sendVideoToRecognition,
    loadVideoSettings: loadVideoJobSettings,
    sendAudioToVideo
  })
  const videoModalState = videoModalController.state
  const commonModalController = new CommonModalController({
    applyImagePrompt: (preset, mode) => {
      const currentPrompt = imageForm.prompt.trimEnd()
      filterPromptPreset = preset.wildcard ? '' : preset.id
      imageForm.prompt = mode === 'append' && currentPrompt ? `${currentPrompt}\n${preset.prompt}` : preset.prompt
      resetImageEnhancement()
    },
    applyVideoPrompt: (preset, mode) => {
      const currentPrompt = videoForm.prompt.trimEnd()
      videoPromptPreset = preset.wildcard ? '' : preset.id
      videoForm.prompt = mode === 'append' && currentPrompt ? `${currentPrompt}\n${preset.prompt}` : preset.prompt
      resetVideoEnhancement()
    }
  })
  const commonModalState = commonModalController.state
  const videoTimelineController = new VideoTimelineController({
    getDuration: () => videoDurationSeconds,
    setDuration: (value) => videoDurationSeconds = value,
    getFPS: () => videoForm.fps,
    getStartImage: () => videoImage,
    setStartImage: (value) => videoImage = value,
    getEndImage: () => videoEndImage,
    setEndImage: (value) => videoEndImage = value,
    setEndStrength: (value) => videoEndStrength = value,
    getKeyframes: () => videoKeyframes,
    setKeyframes: (value) => videoKeyframes = value,
    getAudioClips: () => videoAudioClips,
    setAudioClips: (value) => videoAudioClips = value,
    setPromptMessage: (value) => videoPromptCreationMessage = value,
    resetEnhancement: resetVideoEnhancement,
    allocateKeyframeID: () => nextVideoKeyframeID++,
    sendAudioToVideo: (job) => sendAudioToVideo(job)
  })
  const videoGenerationController = new VideoGenerationController({
    api,
    actions: {
      getState: videoGenerationSnapshot,
      patch: patchVideoGenerationState,
      clearConditioning: () => videoTimelineController.clearConditioning(),
      clearAudio: () => videoTimelineController.clearAudio(),
      normalizeTiming: () => videoTimelineController.normalizeTiming(),
      visualContext: videoAssistantVisualContext,
      showNewest: () => showNewestListPage('video'),
      refresh
    }
  })
  const speechRecognitionController = new SpeechRecognitionController({
    api,
    actions: {
      getState: speechRecognitionSnapshot,
      patch: patchSpeechRecognitionState,
      clearRecognitionFileInput: () => { if (recognitionFileInput) recognitionFileInput.value = '' },
      closeRecognitionPicker: () => $videoModalState.recognitionVideoPickerOpen = false,
      showNewest: showNewestListPage,
      refresh
    }
  })
  const mediaTransferController = new MediaTransferController({
    switchTab: switchAssistantTab,
    setVideoConditionImage,
    getVideoKeyframes: () => videoKeyframes,
    setVideoKeyframes: (value) => videoKeyframes = value,
    getVideoForm: () => videoForm,
    setVideoForm: (value) => videoForm = value,
    getVideoDuration: () => videoDurationSeconds,
    setVideoDuration: (value) => videoDurationSeconds = value,
    nearestAvailableVideoKeyframeFrame,
    allocateVideoKeyframeID: () => nextVideoKeyframeID++,
    normalizeVideoImage: normalizedVideoImage,
    setRecognitionSourceVideoJob: (value) => recognitionSourceVideoJob = value,
    setRecognitionFile: (value) => recognitionFile = value,
    clearRecognitionFileInput: () => { if (recognitionFileInput) recognitionFileInput.value = '' },
    getRecognitionForm: () => recognitionForm,
    setRecognitionForm: (value) => recognitionForm = value,
    resetRecognitionOptions,
    clearVideoConditioning,
    getConfig: () => config,
    getSpeechJobs: () => speechJobs,
    allocateVideoAudioClipID: () => nextVideoAudioClipID++,
    getVideoAudioClips: () => videoAudioClips,
    setVideoAudioClips: (value) => videoAudioClips = value,
    resetVideoEnhancement,
    setError: (value) => error = value,
    snapVideoDuration,
    normalizeVideoTiming
  })
  const assistantController = new AssistantController({
    openSettings,
    setTab: (value) => tab = value,
    setMobilePane: (target, pane) => {
      if (target === 'image') mobileImagePane = pane
      if (target === 'video') mobileVideoPane = pane
      if (target === 'speech') mobileSpeechPane = pane
      if (target === 'recognition') mobileRecognitionPane = pane
    },
    actionContext: () => ({
      setFeatureModulesOpen: (value) => imageModalController.setState({ featureModulesOpen: value }),
      getImageForm: () => imageForm,
      setImageForm: (value) => imageForm = value,
      setImageEnhanceEnabled: (value) => imageEnhanceEnabled = value,
      setImageResolutionMode: (value) => imageResolutionMode = value,
      resetImageEnhancement,
      getVideoForm: () => videoForm,
      setVideoForm: (value) => videoForm = value,
      setVideoDurationSeconds: (value) => videoDurationSeconds = value,
      snapVideoDuration,
      normalizeVideoTiming,
      setVideoEnhanceEnabled: (value) => videoEnhanceEnabled = value,
      resetVideoEnhancement,
      getSpeechForm: () => speechForm,
      setSpeechForm: (value) => speechForm = value,
      getRecognitionForm: () => recognitionForm,
      setRecognitionForm: (value) => recognitionForm = value,
      getKreaModules: () => kreaModules,
      setKreaModules: (value) => kreaModules = value,
      toggleKreaModule,
      applyIdentityPreset,
      getImageJobs: () => imageJobs,
      addKreaRefObjects,
      setKreaImage,
      getKreaOptions: () => kreaOptions,
      setKreaOptions: (value) => kreaOptions = value
    }),
    setError: (value) => error = value,
    getError: () => error,
    imageDisabledReason,
    imageEnhancementActive,
    imageEnhancementCurrent,
    enhanceImagePrompt,
    generateImage,
    getVideoForm: () => videoForm,
    getVideoAudioJob: () => videoAudioJob,
    getVideoImage: () => videoImage,
    getVideoEndImage: () => videoEndImage,
    getVideoKeyframes: () => videoKeyframes,
    createVideoPromptFromScenes,
    videoEnhancementActive,
    videoEnhancementCurrent,
    enhanceVideoPrompt,
    generateVideo,
    getSpeechForm: () => speechForm,
    generateSpeech,
    getRecognitionForm: () => recognitionForm,
    getRecognitionFile: () => recognitionFile,
    getRecognitionSourceVideoJob: () => recognitionSourceVideoJob,
    recognizeSpeech
  })
  const appLifecycleController = new AppLifecycleController({
    api,
    storage: localStorage,
    actions: {
      pageSizeOptionsFor,
      preferenceDefaults: () => ({ pageSizes: listPageSizes, sortOrders: listSortOrders }),
      applyPreferences: ({ views, pageSizes, sortOrders }) => {
        subtitleView = views.subtitle
        imageView = views.image
        videoView = views.video
        speechView = views.speech
        listPageSizes = pageSizes
        listSortOrders = sortOrders
      },
      applyConfig: applyRuntimeConfig,
      setError: (value) => error = value,
      refreshUserLoras,
      refreshJobs: refresh,
      refreshSystemUsage,
      refreshVideoModels: refreshVideoModelStatus,
      refreshImageModels: refreshImageCheckpointStatus,
      shouldRefreshModels: () => tab === 'settings' || Boolean(videoModelStatus?.preparing),
      setProgressClock: (value) => progressClock = value
    }
  })
  $: jobs = $jobState.jobs
  $: engineStates = $jobState.engineStates
  $: deletingJob = $jobState.deletingJob
  $: cancellingJob = $jobState.cancellingJob
  $: retryingJob = $jobState.retryingJob
  $: updatingTagsJob = $jobState.updatingTagsJob
  $: upscalingImageJob = $imageResultState.upscalingJob
  $: detailEnhancingImageJob = $imageResultState.detailEnhancingJob
  $: cloningImageJob = $imageResultState.cloningJob
  $: imageCloneMessage = $imageResultState.cloneMessage
  $: systemUsage = $settingsState.systemUsage
  $: videoModelStatus = $settingsState.videoModelStatus
  $: imageCheckpointStatus = $settingsState.imageCheckpointStatus
  $: preparingVideoModels = $settingsState.preparingVideoModels
  $: preparingImageCheckpoints = $settingsState.preparingImageCheckpoints
  $: convertingImageCheckpoints = $settingsState.convertingImageCheckpoints
  $: savingDownloadCredentials = $settingsState.savingDownloadCredentials
  $: storage = $settingsState.storage
  $: cleaningStorage = $settingsState.cleaningStorage
  $: imageJobs = (jobs, listSortOrders, listTagFilters, listTagExclusions, listTagUntaggedOnly, listTagMatchModes, mediaListController.ordered('image'))
  $: videoJobs = (jobs, listSortOrders, listTagFilters, listTagExclusions, listTagUntaggedOnly, listTagMatchModes, mediaListController.ordered('video'))
  $: speechJobs = (jobs, listSortOrders, listTagFilters, listTagExclusions, listTagUntaggedOnly, listTagMatchModes, mediaListController.ordered('speech'))
  $: recognitionJobs = (jobs, listSortOrders, listTagFilters, listTagExclusions, listTagUntaggedOnly, listTagMatchModes, mediaListController.ordered('recognition'))
  $: pagedImageJobs = (jobs, listPages, listPageSizes, listSortOrders, listTagFilters, listTagExclusions, listTagUntaggedOnly, listTagMatchModes, mediaListController.pageItems('image'))
  $: pagedVideoJobs = (jobs, listPages, listPageSizes, listSortOrders, listTagFilters, listTagExclusions, listTagUntaggedOnly, listTagMatchModes, mediaListController.pageItems('video'))
  $: pagedSpeechJobs = (jobs, listPages, listPageSizes, listSortOrders, listTagFilters, listTagExclusions, listTagUntaggedOnly, listTagMatchModes, mediaListController.pageItems('speech'))
  $: pagedRecognitionJobs = (jobs, listPages, listPageSizes, listSortOrders, listTagFilters, listTagExclusions, listTagUntaggedOnly, listTagMatchModes, mediaListController.pageItems('recognition'))
  $: pagedHistoryJobs = (jobs, listPages, listPageSizes, listSortOrders, mediaListController.pageItems('history'))

  function appPresentationSnapshot() {
    return {
      jobs, progressClock, engineStates, tab, busy,
      imageForm, imageEnhanceEnabled, activeKreaModuleLabels,
      videoForm, videoDurationSeconds, videoEnhanceEnabled, videoImage, videoEndImage, videoAudioJob, videoAudioClips, videoKeyframes,
      speechForm, recognitionForm, recognitionFile, recognitionSourceVideoJob,
      pagedImageJobs, listPages, listPageSizes
    }
  }

  function mediaListSnapshot() {
    return {
      jobs, pages: listPages, pageSizes: listPageSizes, sortOrders: listSortOrders,
      tagFilters: listTagFilters, tagExclusions: listTagExclusions, tagUntaggedOnly: listTagUntaggedOnly, tagMatchModes: listTagMatchModes,
      views: { image: imageView, video: videoView, speech: speechView, subtitle: subtitleView },
      mobilePanes: { image: mobileImagePane, video: mobileVideoPane, speech: mobileSpeechPane, recognition: mobileRecognitionPane }
    }
  }

  function patchMediaListState(patch) {
    if ('pages' in patch) listPages = patch.pages
    if ('pageSizes' in patch) listPageSizes = patch.pageSizes
    if ('sortOrders' in patch) listSortOrders = patch.sortOrders
    if ('tagFilters' in patch) listTagFilters = patch.tagFilters
    if ('tagExclusions' in patch) listTagExclusions = patch.tagExclusions
    if ('tagUntaggedOnly' in patch) listTagUntaggedOnly = patch.tagUntaggedOnly
    if ('tagMatchModes' in patch) listTagMatchModes = patch.tagMatchModes
    if ('views' in patch) {
      imageView = patch.views.image; videoView = patch.views.video; speechView = patch.views.speech; subtitleView = patch.views.subtitle
    }
    if ('mobilePanes' in patch) {
      mobileImagePane = patch.mobilePanes.image; mobileVideoPane = patch.mobilePanes.video
      mobileSpeechPane = patch.mobilePanes.speech; mobileRecognitionPane = patch.mobilePanes.recognition
    }
  }

  function clampListPages() {
    mediaListController.clampPages()
  }

  function setListPage(key, page) {
    mediaListController.setPage(key, page)
  }

  function pageSizeOptionsFor(key) {
    return mediaListController.sizeOptions(key)
  }

  function setListPageSize(key, pageSize) {
    mediaListController.setPageSize(key, pageSize)
  }

  function setListSortOrder(key, order) {
    mediaListController.setSortOrder(key, order)
  }

  function setListTagFilter(key, tags) {
    mediaListController.setTagFilter(key, tags)
  }

  function setListTagExclusions(key, tags) {
    mediaListController.setTagExclusions(key, tags)
  }

  function setListTagUntaggedOnly(key, enabled) {
    mediaListController.setTagUntaggedOnly(key, enabled)
  }

  function setListTagMatchMode(key, mode) {
    mediaListController.setTagMatchMode(key, mode)
  }

  async function updateJobTags(job, tags) {
    const updated = await jobController.updateTags(job, tags)
    if (updated) clampListPages()
    return updated
  }

  function showNewestListPage(key) {
    mediaListController.showNewest(key)
  }

  async function refresh() {
    const refreshed = await jobController.refresh()
    if (refreshed) clampListPages()
  }

  async function refreshSystemUsage() {
    return settingsController.refreshSystemUsage()
  }

  async function refreshVideoModelStatus() {
    return settingsController.refreshVideoModels()
  }

  async function refreshImageCheckpointStatus() {
    return settingsController.refreshImageCheckpoints()
  }

  async function prepareImageCheckpoints() {
    const variants = Object.entries(checkpointSelection).filter(([, selected]) => selected).map(([id]) => id)
    const result = await settingsController.prepareImageCheckpoints({ civitaiToken, hfToken, variants })
    if (result.clearTokens) { civitaiToken = ''; hfToken = '' }
  }

  async function convertImageCheckpointsNVFP4() {
    const variants = Object.entries(nvfp4Selection).filter(([, selected]) => selected).map(([id]) => id)
    const result = await settingsController.convertImageCheckpoints({ civitaiToken, variants, removeBF16Sources })
    if (result.clearCivitaiToken) civitaiToken = ''
  }

  async function prepareVideoModels() {
    const result = await settingsController.prepareVideoModels(hfToken)
    if (result.clearHFToken) hfToken = ''
  }

  async function saveDownloadCredentials() {
    const result = await settingsController.saveDownloadCredentials(civitaiToken, hfToken)
    if (result.clearTokens) { civitaiToken = ''; hfToken = '' }
  }

  function applyRuntimeConfig(value, smartResolution = true) {
    const defaults = runtimeDefaults(value, kreaOptions, imageModeChoices)
    config = value
    settings = structuredClone(value)
    imageForm = { ...imageForm, ...defaults.imageForm }
    if (smartResolution) applySmartResolution()
    speechForm = { ...speechForm, ...defaults.speechForm }
    recognitionForm = { ...recognitionForm, ...defaults.recognitionForm }
    videoForm = { ...videoForm, ...defaults.videoForm }
    videoDurationSeconds = defaults.videoDuration
    videoEnhanceEnabled = defaults.enhanceEnabled
    imageEnhanceEnabled = defaults.enhanceEnabled
    kreaOptions = defaults.options
  }

  onMount(() => appLifecycleController.start())

  function setSubtitleView(view) {
    mediaListController.setView('subtitle', view)
  }

  function setImageView(view) {
    mediaListController.setView('image', view)
  }

  function setVideoView(view) {
    mediaListController.setView('video', view)
  }

  function setSpeechView(view) {
    mediaListController.setView('speech', view)
  }

  function setMobilePane(key, pane) {
    mediaListController.setMobilePane(key, pane)
  }

  function addRefs(files) {
    const limit = imageForm.mode === 'control' ? 1 : (config?.image.max_reference_images || 4)
    imageInputController.addRefFiles(files, limit)
  }

  function addRefObjects(incoming) {
    const limit = imageForm.mode === 'control' ? 1 : (config?.image.max_reference_images || 4)
    imageInputController.addRefs(incoming, limit)
  }

  function clearRefs() {
    imageInputController.clearRefs()
  }

  function removeRef(index) {
    imageInputController.removeRef(index)
  }

  function imageConfigurationSnapshot() {
    return {
      form: imageForm, modules: kreaModules, options: kreaOptions, preserveItems: identityPreserveItems,
      megapixels: imageMegapixels, resolutionMode: imageResolutionMode,
      userLoraCatalog, userLoraSelections, styleSelections: kreaStyleSelections,
      settings, checkpointStatus: imageCheckpointStatus
    }
  }

  function patchImageConfigurationState(patch) {
    if ('modules' in patch) kreaModules = patch.modules
    if ('options' in patch) kreaOptions = patch.options
    if ('megapixels' in patch) imageMegapixels = patch.megapixels
    if ('resolutionMode' in patch) imageResolutionMode = patch.resolutionMode
    if ('userLoraCatalog' in patch) userLoraCatalog = patch.userLoraCatalog
    if ('userLoraSelections' in patch) userLoraSelections = patch.userLoraSelections
    if ('styleSelections' in patch) kreaStyleSelections = patch.styleSelections
    if ('settings' in patch) settings = patch.settings
  }

  function toggleKreaModule(module) {
    imageConfigurationController.toggleModule(module)
  }

  async function refreshUserLoras() {
    return imageConfigurationController.refreshUserLoras()
  }

  function hasUserLora(filename) {
    return imageConfigurationController.hasUserLora(filename)
  }

  function toggleUserLora(filename) {
    imageConfigurationController.toggleUserLora(filename)
  }

  function updateUserLoraStrength(filename, strength) {
    imageConfigurationController.updateUserLoraStrength(filename, strength)
  }

  function userLoraLabel(filename) {
    return imageConfigurationController.userLoraLabel(filename)
  }

  function hasKreaStyle(name) {
    return imageConfigurationController.hasStyle(name)
  }

  function toggleKreaStyle(name) {
    imageConfigurationController.toggleStyle(name)
  }

  function updateKreaStyleStrength(name, strength) {
    imageConfigurationController.updateStyleStrength(name, strength)
  }

  function kreaStyleLabel(name) {
    return imageConfigurationController.styleLabel(name)
  }

  function addKreaRefs(kind, files) {
    imageInputController.addKreaRefFiles(kind, files)
  }

  function addKreaRefObjects(kind, incoming) {
    imageInputController.addKreaRefs(kind, incoming)
  }

  function clearKreaRefs(kind) {
    imageInputController.clearKreaRefs(kind)
  }

  function removeKreaRef(kind, index) {
    imageInputController.removeKreaRef(kind, index)
  }

  function addIdentityReferenceObjects(incoming) {
    imageInputController.addIdentityReferences(incoming)
  }

  function addIdentityReferences(files) {
    imageInputController.addIdentityReferenceFiles(files)
  }

  function clearIdentityReferences() {
    imageInputController.clearIdentityReferences()
  }

  function removeIdentityReference(index) {
    imageInputController.removeIdentityReference(index)
  }

  function setKreaImage(kind, image) {
    const normalized = imageInputController.setImage(kind, image)
    if (kind === 'identity') {
      parentImageJobID = normalized?.server && normalized.role === 'output' ? String(normalized.ref || '').split(':')[0] : ''
    } else if (kind === 'depth') {
      depthPoseID = normalized?.poseID || ''
    } else if (kind === 'nk2e') {
      nk2ePoseID = normalized?.poseID || ''
      kreaNK2EPreprocessed = Boolean(normalized?.preprocessed)
    }
  }

  async function usePickedVideoFrame(file, target, sourceTime, sourceDuration) {
    return mediaTransferController.usePickedVideoFrame(file, target, sourceTime, sourceDuration)
  }

  function sendVideoToRecognition(job) {
    mediaTransferController.sendVideoToRecognition(job)
  }

  function clearRecognitionSourceVideo() {
    mediaTransferController.clearRecognitionSourceVideo()
  }

  function loadVideoJobSettings(job) {
    mediaTransferController.loadVideoJobSettings(job)
  }

  async function sendAudioToVideo(job) {
    return mediaTransferController.sendAudioToVideo(job)
  }

  function removeVideoAudio(id) {
    videoTimelineController.removeAudio(id)
  }

  function moveVideoAudio(id, rawStart) {
    videoTimelineController.moveAudio(id, rawStart)
  }

  function togglePickedVideoAudio(job) {
    videoTimelineController.toggleAudio(job)
  }

  function imageCreationSnapshot() {
    return {
      config, busy, form: imageForm, enhanceEnabled: imageEnhanceEnabled, enhancedPrompt: imageEnhancedPrompt, enhancedSource: imageEnhancedSource,
      resolutionMode: imageResolutionMode, aspectRatio: imageAspectRatio, megapixels: imageMegapixels, filterPromptPreset,
      modules: kreaModules, options: kreaOptions, identityPreset, identityPreserveItems, identityPreserveCustom,
      styleSelections: kreaStyleSelections, userLoraSelections, parentJobID: parentImageJobID, depthPoseID, nk2ePoseID, nk2ePreprocessed: kreaNK2EPreprocessed,
      references: refs, identityImage: kreaIdentityImage, identityReferences: kreaIdentityReferences, identityMask: kreaIdentityMask, strictMask: kreaStrictMask,
      depthImage: kreaDepthImage, nk2eImage: kreaNK2EImage, anypaintImage: kreaAnyPaintImage, anypaintMask: kreaAnyPaintMask,
      visionImages: kreaVisionImages, styleReferenceImages: kreaStyleReferenceImages,
      sequence: $imageSequenceState
    }
  }

  function patchImageCreationState(patch) {
    if ('config' in patch) config = patch.config
    if ('busy' in patch) busy = patch.busy
    if ('error' in patch) error = patch.error
    if ('enhancing' in patch) enhancingPrompt = patch.enhancing
    if ('form' in patch) imageForm = patch.form
    if ('enhanceEnabled' in patch) imageEnhanceEnabled = patch.enhanceEnabled
    if ('enhancedPrompt' in patch) imageEnhancedPrompt = patch.enhancedPrompt
    if ('enhancedSource' in patch) imageEnhancedSource = patch.enhancedSource
    if ('resolutionMode' in patch) imageResolutionMode = patch.resolutionMode
    if ('aspectRatio' in patch) imageAspectRatio = patch.aspectRatio
    if ('megapixels' in patch) imageMegapixels = patch.megapixels
    if ('filterPromptPreset' in patch) filterPromptPreset = patch.filterPromptPreset
    if ('modules' in patch) kreaModules = patch.modules
    if ('options' in patch) kreaOptions = patch.options
    if ('identityPreset' in patch) identityPreset = patch.identityPreset
    if ('identityPreserveItems' in patch) identityPreserveItems = patch.identityPreserveItems
    if ('identityPreserveCustom' in patch) identityPreserveCustom = patch.identityPreserveCustom
    if ('styleSelections' in patch) kreaStyleSelections = patch.styleSelections
    if ('userLoraSelections' in patch) userLoraSelections = patch.userLoraSelections
    if ('parentJobID' in patch) parentImageJobID = patch.parentJobID
    if ('depthPoseID' in patch) depthPoseID = patch.depthPoseID
    if ('nk2ePoseID' in patch) nk2ePoseID = patch.nk2ePoseID
    if ('nk2ePreprocessed' in patch) kreaNK2EPreprocessed = patch.nk2ePreprocessed
    if ('mobilePane' in patch) mobileImagePane = patch.mobilePane
  }

  function clearAllImageInputs() {
    clearRefs()
    for (const kind of ['identity', 'identityReference', 'identityMask', 'strictMask', 'depth', 'nk2e', 'anypaint', 'anypaintMask']) setKreaImage(kind, null)
    clearKreaRefs('vision')
    clearKreaRefs('styleReference')
  }

  function clearGeneratedInputs() {
    clearAllImageInputs()
  }

  function identityHasExtraUserPrompt() {
    return imageCreationController.hasExtraIdentityPrompt()
  }

  function rawImagePrompt() {
    return imageCreationController.rawPrompt()
  }

  function toggleIdentityPreserveItem(id) {
    imageCreationController.togglePreserveItem(id)
  }

  function applyIdentityPreset(value) {
    imageCreationController.applyIdentityPreset(value)
  }

  function isPureOutpaint() {
    return checkPureOutpaint({ modules: kreaModules, anypaintImage: kreaAnyPaintImage, anypaintMask: kreaAnyPaintMask, options: kreaOptions })
  }

  function imageDisabledReason() {
    return imageCreationController.disabledReason()
  }

  function kreaModuleDisabledReason() {
    return imageCreationController.moduleDisabledReason()
  }

  function disableAllKreaModules() {
    for (const name of Object.keys(kreaModules)) {
      if (kreaModules[name]) toggleKreaModule(name)
    }
  }

  function handleFeatureModulesKeydown(event) {
    if (event.key !== 'Escape' || !$imageModalState.featureModulesOpen) return
    if ($imageModalState.maskEditorMode || $imageModalState.cannyEditorOpen || $imageModalState.image || $imageModalState.runtimeInfoOpen || $imageModalState.recentPickerTarget || $imageModalState.presetPickerTarget || $imageModalState.remoteTarget) return
    $imageModalState.featureModulesOpen = false
  }

  function looksLikeStructuredPrompt(value = imageForm.prompt) {
    return imageCreationController.looksStructured(value)
  }

  function imageEnhancementActive(enabled = imageEnhanceEnabled, prompt = rawImagePrompt()) {
    if (enabled === imageEnhanceEnabled && prompt === rawImagePrompt()) return imageCreationController.enhancementActive()
    return canEnhanceImagePrompt({ enabled, prompt, structured: looksLikeStructuredPrompt(prompt), identityTryonWithoutUserPrompt: kreaModules.identity && identityPreset === 'tryon' && !identityHasExtraUserPrompt() })
  }

  function imageEnhancementCurrent(enhanced = imageEnhancedPrompt, source = imageEnhancedSource, current = rawImagePrompt()) {
    if (enhanced === imageEnhancedPrompt && source === imageEnhancedSource && current === rawImagePrompt()) return imageCreationController.enhancementCurrent()
    return isCurrentImageEnhancement({ enhanced, source, current })
  }

  // These values are rendered in the submit controls. Keep their dependencies
  // explicit so nested form bindings immediately update the button state.
  $: imageEnhancementIsActive = imageEnhancementActive(imageEnhanceEnabled, rawImagePrompt())
  $: activeKreaModuleLabels = Object.entries(kreaModules).filter(([, enabled]) => enabled).map(([name]) => kreaModuleLabels[name])
  $: kreaModuleMessage = (
    kreaModules, identityPreset, kreaIdentityImage, kreaIdentityReference, kreaDepthImage, kreaVisionImages, kreaStyleReferenceImages,
    kreaStyleSelections, userLoraSelections, kreaNK2EImage, kreaAnyPaintImage, kreaAnyPaintMask, kreaOptions,
    kreaModuleDisabledReason()
  )
  $: imageEnhancementIsCurrent = (
    imageForm, identityPreserveItems, identityPreserveCustom, kreaModules,
    imageEnhancementCurrent(imageEnhancedPrompt, imageEnhancedSource, rawImagePrompt())
  )
  $: if (imageCheckpointStatus?.identity_runtime && !imageCheckpointStatus.identity_runtime.heretic_ready && kreaOptions.identity_encoder === 'heretic') {
    kreaOptions = { ...kreaOptions, identity_encoder: 'default' }
  }
  $: imageDisabledMessage = (
    busy, jobs, imageForm, refs, kreaModules, identityPreset, kreaIdentityImage, kreaIdentityReference, kreaDepthImage,
    kreaVisionImages, kreaStyleReferenceImages, kreaStyleSelections, userLoraSelections,
    kreaNK2EImage, kreaAnyPaintImage, kreaAnyPaintMask, kreaOptions,
    imageDisabledReason()
  )

  function resetImageEnhancement() {
    imageCreationController.resetEnhancement()
  }

  function resetImageCreation() {
    imageCreationController.reset()
  }

  async function enhanceImagePrompt() {
    return imageCreationController.enhance()
  }

  function applySmartResolution() {
    imageCreationController.applySmartResolution()
  }

  function useCustomImageResolution() {
    imageResolutionMode = 'custom'
  }

  function cloneImagePrompt(job) {
    imageCreationController.clonePrompt(job)
  }

  async function createRandomVideoPrompt(variant = 'no_camera') {
    if (videoImage || videoEndImage || videoKeyframes.some((item) => item.image)) {
      throw new Error('시작·마지막·키프레임 이미지가 있을 때는 장면 이미지의 프롬프트 만들기를 사용하세요.')
    }
    const wildcard = await api.randomPromptWildcard(variant)
    const duration = Math.max(0.1, Number(videoDurationSeconds) || 5)
    const raw = `Duration: ${duration.toFixed(1)} seconds\nMuse scene seed: ${wildcard.muse}\nStyle seed: ${wildcard.style}`
    const form = new FormData()
    form.append('prompt', raw)
    form.append('mode', 't2v_wildcard')
    const result = await api.enhancePrompt(form)
    const variantLabel = wildcard.muse_variant === 'muse.txt' ? 'Muse' : 'Muse (No Camera)'
    return {
      id: `ltx-wildcard-${wildcard.muse_index}-${wildcard.style_index}-${Date.now()}`,
      label: `${variantLabel} 영상 · 장면 ${wildcard.muse_index} + 스타일 ${wildcard.style_index}`,
      prompt: result.enhanced_prompt,
      source: wildcard.source,
      sourceKey: 'ltx-wildcard',
      sourceLabel: 'Crocody Muse × Style · LTX 변환',
      previewIcon: '🎬',
      previewTone: 'green',
      wildcard: { ...wildcard, duration }
    }
  }

  function filterModeDefault(mode) { return imageConfigurationController.filterModeDefault(mode) }
  function checkpointVisible(checkpoint) { return imageConfigurationController.checkpointVisible(checkpoint) }
  function setCheckpointVisible(checkpoint, visible) { imageConfigurationController.setCheckpointVisible(checkpoint, visible) }
  function displayCheckpointReady(checkpoint) { return imageConfigurationController.checkpointReady(checkpoint) }
  function selectKreaCheckpoint(checkpoint) { imageConfigurationController.selectCheckpoint(checkpoint) }
  function selectedKreaCheckpoint() { return imageConfigurationController.selectedCheckpoint() }
  function selectedKreaCheckpointSource() { return imageConfigurationController.selectedCheckpointSource() }
  function filterModeMaximum(mode) { return imageConfigurationController.filterModeMaximum(mode) }
  function cloneImageSettings(job) {
    imageCreationController.cloneSettings(job)
  }
  async function cloneImageReferences(job) {
    return imageCreationController.cloneReferences(job)
  }
  function continueEditing(job) {
    imageCreationController.continueEditing(job)
  }
  async function cloneImageJob(job, part) {
    return imageResultController.clone(job, part)
  }

  function imageSequenceBlockedMessage() {
    return sequenceBlockedMessage({
      mode: imageForm.mode, modules: kreaModules, moduleReason: kreaModuleDisabledReason(),
      checkpoint: kreaOptions.checkpoint,
      hasReIDReference: Boolean(imageSequenceCharacters[0]?.references?.length)
    })
  }

  function setImageSequenceEntryMode(value) { imageSequenceController.setEntryMode(value) }
  function setImageSequenceStoryIdea(value) { imageSequenceController.setStoryIdea(value) }
  function setImageSequenceSceneCount(value) { imageSequenceController.setSceneCount(value) }
  function setImageSequenceSharedPrompt(value) { imageSequenceController.setSharedPrompt(value) }
  function applyStorySequenceExample() { imageSequenceController.applyStoryExample() }
  function applySceneSequenceExample() { imageSequenceController.applySceneExample() }

  async function planImageSequence() {
    try {
      await imageSequenceController.plan(api)
    } catch (planError) {
      error = planError.message || String(planError)
    }
  }

  function addImageSequenceScene() {
    imageSequenceController.addScene()
  }

  function removeImageSequenceScene(index) {
    imageSequenceController.removeScene(index)
  }

  function moveImageSequenceScene(index, direction) {
    imageSequenceController.moveScene(index, direction)
  }

  function updateImageSequencePrompt(index, value) {
    imageSequenceController.updatePrompt(index, value)
  }

  function addImageSequenceCharacter() { imageSequenceController.addCharacter() }
  function removeImageSequenceCharacter(index) { imageSequenceController.removeCharacter(index) }
  function setImageSequenceCharacterName(index, value) { imageSequenceController.setCharacterName(index, value) }
  function addImageSequenceCharacterFiles(index, files) { imageSequenceController.addCharacterFiles(index, files) }
  function addImageSequenceCharacterResult(index, job) { imageSequenceController.addCharacterResult(index, job) }
  function removeImageSequenceCharacterReference(index, referenceIndex) { imageSequenceController.removeCharacterReference(index, referenceIndex) }
  function setImageSequenceCharacterReIDReference(index, referenceIndex) { imageSequenceController.setCharacterReIDReference(index, referenceIndex) }
  function toggleImageSequenceCharacterTrait(index, trait) { imageSequenceController.toggleCharacterTrait(index, trait) }
  async function generateImageSequenceCharacterSheet(index) {
    try { await imageSequenceController.generateCharacterSheet(index, api) }
    catch (sheetError) { error = sheetError.message || String(sheetError) }
  }
  function approveImageSequenceCharacterSheet(index) { imageSequenceController.approveCharacterSheet(index) }
  function discardImageSequenceCharacterSheet(index) { imageSequenceController.discardCharacterSheet(index) }
  async function analyzeImageSequenceCharacter(index) {
    try {
      await imageSequenceController.analyzeCharacter(index, api)
    } catch (analysisError) {
      error = analysisError.message || String(analysisError)
    }
  }
  function setImageSequenceCharacterDescription(index, value) { imageSequenceController.setCharacterDescription(index, value) }
  function setImageSequenceCharacterPrompt(index, value) { imageSequenceController.setCharacterCanonicalPrompt(index, value) }
  function imageSequenceCharacterPreview(reference) { return imageSequenceController.characterPreview(reference) }
  function imageSequenceCharacterReadinessMessage() { return imageSequenceController.characterReadinessMessage() }

  async function generateImage(sequencePrompts = null) {
    return imageCreationController.generate(sequencePrompts)
  }
  async function upscaleImage(job) {
    return imageResultController.upscale(job)
  }

  async function detailEnhanceImage(job) {
    return imageResultController.detailEnhance(job)
  }

  async function submitGarmentExtraction(form) {
    return imageResultController.submitGarment(form)
  }

  async function submitFaceSwap(form) {
    return imageResultController.submitFaceSwap(form)
  }

  function speechRecognitionSnapshot() {
    return { config, busy, speechForm, recognitionForm, recognitionFile, recognitionSourceVideoJob, recognitionOptions, loadingRecognitionOptions }
  }

  function patchSpeechRecognitionState(patch) {
    if ('busy' in patch) busy = patch.busy
    if ('error' in patch) error = patch.error
    if ('speechForm' in patch) speechForm = patch.speechForm
    if ('recognitionForm' in patch) recognitionForm = patch.recognitionForm
    if ('recognitionFile' in patch) recognitionFile = patch.recognitionFile
    if ('recognitionSourceVideoJob' in patch) recognitionSourceVideoJob = patch.recognitionSourceVideoJob
    if ('recognitionOptions' in patch) recognitionOptions = patch.recognitionOptions
    if ('loadingRecognitionOptions' in patch) loadingRecognitionOptions = patch.loadingRecognitionOptions
    if ('mobileSpeechPane' in patch) mobileSpeechPane = patch.mobileSpeechPane
    if ('mobileRecognitionPane' in patch) mobileRecognitionPane = patch.mobileRecognitionPane
  }

  async function generateSpeech() {
    return speechRecognitionController.generateSpeech()
  }

  async function recognizeSpeech() {
    return speechRecognitionController.recognize()
  }

  function resetRecognitionOptions() {
    speechRecognitionController.resetOptions()
  }

  function updateRecognitionURL(event) {
    speechRecognitionController.updateURL(event.currentTarget.value)
  }

  function updateRecognitionFile(event) {
    speechRecognitionController.updateFile(event.currentTarget.files?.[0] || null)
  }

  function clearRecognitionFile() {
    speechRecognitionController.clearFile()
  }

  function selectedRecognitionPart() {
    return speechRecognitionController.selectedPart()
  }

  function selectRecognitionPart(partID) {
    speechRecognitionController.selectPart(partID)
  }

  async function loadRecognitionOptions() {
    return speechRecognitionController.loadOptions()
  }
  function videoGenerationSnapshot() {
    return {
      config, busy, form: videoForm, duration: videoDurationSeconds,
      startImage: videoImage, endImage: videoEndImage, endStrength: videoEndStrength, keyframes: videoKeyframes, audioClips: videoAudioClips,
      enhanceEnabled: videoEnhanceEnabled, enhancedPrompt: videoEnhancedPrompt, enhancedSource: videoEnhancedSource, enhancedImageKey: videoEnhancedImageKey,
      creatingPrompt: creatingVideoPrompt, promptCreationMessage: videoPromptCreationMessage, promptPreset: videoPromptPreset,
      advancedOpen: videoAdvancedOpen, assistantState
    }
  }

  function patchVideoGenerationState(patch) {
    if ('busy' in patch) busy = patch.busy
    if ('error' in patch) error = patch.error
    if ('form' in patch) videoForm = patch.form
    if ('duration' in patch) videoDurationSeconds = patch.duration
    if ('enhanceEnabled' in patch) videoEnhanceEnabled = patch.enhanceEnabled
    if ('enhancedPrompt' in patch) videoEnhancedPrompt = patch.enhancedPrompt
    if ('enhancedSource' in patch) videoEnhancedSource = patch.enhancedSource
    if ('enhancedImageKey' in patch) videoEnhancedImageKey = patch.enhancedImageKey
    if ('enhancing' in patch) enhancingPrompt = patch.enhancing
    if ('creatingPrompt' in patch) creatingVideoPrompt = patch.creatingPrompt
    if ('promptCreationMessage' in patch) videoPromptCreationMessage = patch.promptCreationMessage
    if ('promptPreset' in patch) videoPromptPreset = patch.promptPreset
    if ('nextKeyframeID' in patch) nextVideoKeyframeID = patch.nextKeyframeID
    if ('audioPickerOpen' in patch) $videoModalState.audioPickerOpen = patch.audioPickerOpen
    if ('advancedOpen' in patch) videoAdvancedOpen = patch.advancedOpen
    if ('mobilePane' in patch) mobileVideoPane = patch.mobilePane
  }

  async function generateVideo() {
    return videoGenerationController.generate()
  }

  function videoInputKey(image) {
    return videoGenerationController.imageKey(image)
  }

  function videoImageKey() {
    return videoInputKey(videoImage)
  }

  function videoEnhancementCurrent(enhanced = videoEnhancedPrompt, source = videoEnhancedSource, prompt = videoForm.prompt, imageKey = videoEnhancedImageKey, currentImageKey = videoImageKey()) {
    if (enhanced === videoEnhancedPrompt && source === videoEnhancedSource && prompt === videoForm.prompt && imageKey === videoEnhancedImageKey && currentImageKey === videoImageKey()) return videoGenerationController.enhancementCurrent()
    return isCurrentVideoEnhancement({ enhanced, source, prompt, imageKey, currentImageKey })
  }

  function videoEnhancementActive(enabled = videoEnhanceEnabled, image = videoImage, currentConfig = config) {
    if (enabled === videoEnhanceEnabled && image === videoImage && currentConfig === config) return videoGenerationController.enhancementActive()
    return canEnhanceVideoPrompt({ enabled, image, visionEnabled: currentConfig?.prompt_enhancement.vision_enabled })
  }

  $: videoEnhancementIsActive = videoEnhancementActive(videoEnhanceEnabled, videoImage, config)
  $: videoEnhancementIsCurrent = (
    videoImage,
    videoEnhancementCurrent(videoEnhancedPrompt, videoEnhancedSource, videoForm.prompt, videoEnhancedImageKey, videoImageKey())
  )

  function resetVideoEnhancement() {
    videoGenerationController.resetEnhancement()
  }

  function normalizedVideoImage(image) {
    return videoTimelineController.normalizeImage(image)
  }

  function videoImagePreview(image) {
    return videoTimelineController.imagePreview(image)
  }

  function setVideoConditionImage(target, image) {
    return videoTimelineController.setConditionImage(target, image)
  }

  function addVideoKeyframe() {
    return videoTimelineController.addKeyframe()
  }

  function videoKeyframeCapacity() {
    return videoTimelineController.capacity()
  }

  function normalizeVideoTiming() {
    return videoTimelineController.normalizeTiming()
  }

  function nearestAvailableVideoKeyframeFrame(rawFrame, excludeID = null) {
    return videoTimelineController.nearestFrame(rawFrame, excludeID)
  }

  function removeVideoKeyframe(id) {
    videoTimelineController.removeKeyframe(id)
  }

  function updateVideoKeyframe(id, field, value) {
    videoTimelineController.updateKeyframe(id, field, value)
  }

  function moveVideoKeyframe(id, rawTime) {
    videoTimelineController.moveKeyframe(id, rawTime)
  }

  function clearVideoConditioning() {
    videoTimelineController.clearConditioning()
  }

  function resetVideoCreation() {
    videoGenerationController.reset()
  }
  function resetSpeechCreation() {
    speechRecognitionController.resetSpeech()
  }

  function resetRecognitionCreation() {
    speechRecognitionController.resetRecognition()
  }
  function videoConditioningDisabledReason() {
    return disabledVideoConditioningReason({ audioSelected: Boolean(videoAudioJob), a2vReady: videoModelStatus?.a2v_ready, seconds: videoDurationSeconds, fps: videoForm.fps, keyframes: videoKeyframes })
  }

  async function enhanceVideoPrompt() {
    return videoGenerationController.enhance()
  }
  async function deleteJob(job) {
    if (await jobController.deleteJob(job, confirm)) clampListPages()
  }

  async function cancelJob(job) {
    if (await jobController.cancelJob(job)) clampListPages()
  }

  async function retryJob(job) {
    if (await jobController.retryJob(job)) clampListPages()
  }

  async function clearFinishedJobs() {
    if (await jobController.clearFinishedJobs(confirm)) clampListPages()
  }

  function openSettings() {
    settings = structuredClone(config)
    settingsVideoDurationSeconds = durationFromFrames(config.video.default_frames, config.video.default_fps)
    savedMessage = ''
    error = ''
    tab = 'settings'
    settingsController.setState({ storage: null })
    settingsController.loadStorage()
    refreshVideoModelStatus()
    refreshImageCheckpointStatus()
  }

  async function videoAssistantVisualContext(message) {
    const conditions = [
      ...(videoImage ? [{ label: 'START', detail: '0초', image: videoImage }] : []),
      ...videoKeyframes.filter((item) => item.image).map((item, index) => ({ label: `KEYFRAME ${index + 1}`, detail: `${Number(item.time).toFixed(1)}초`, image: item.image })),
      ...(videoEndImage ? [{ label: 'END', detail: `${((framesForDuration(videoDurationSeconds, videoForm.fps) - 1) / videoForm.fps).toFixed(1)}초`, image: videoEndImage }] : [])
    ]
    return createVideoVisualContext({ message, conditions, imageURL: videoImagePreview })
  }

  async function createVideoPromptFromScenes(automatic = false) {
    return videoGenerationController.createPromptFromScenes(automatic)
  }
  $: assistantState = (
    tab, busy, imageForm, imageEnhanceEnabled, activeKreaModuleLabels,
    videoForm, videoDurationSeconds, videoEnhanceEnabled, videoImage, videoEndImage, videoAudioJob, videoAudioClips, videoKeyframes,
    speechForm, recognitionForm, recognitionFile, recognitionSourceVideoJob, pagedImageJobs, listPages, listPageSizes,
    appPresentationController.assistantState()
  )
  $: imageResultsModel = {
    jobs: imageJobs, pagedJobs: pagedImageJobs, view: imageView, page: listPages.image, pageSize: listPageSizes.image, sortOrder: listSortOrders.image,
    garmentOnline: engineStates.garment === 'online', faceSwapOnline: engineStates.faceswap === 'online', imageOnline: engineStates.image_create === 'online', upscaleOnline: engineStates.upscale === 'online',
    cloningJob: cloningImageJob, detailEnhancingJob: detailEnhancingImageJob, upscalingJob: upscalingImageJob,
    cancellingJob, retryingJob, deletingJob, updatingTagsJob, progressFor: imageGenerationProgress, promptText: imagePromptModalText,
    tagOptions: mediaListController.tagOptions('image'), filterTags: listTagFilters.image, excludedTags: listTagExclusions.image,
    untaggedOnly: listTagUntaggedOnly.image, tagMatchMode: listTagMatchModes.image,
    onFilterTags: (tags) => setListTagFilter('image', tags), onExcludeTags: (tags) => setListTagExclusions('image', tags),
    onUntaggedOnly: (enabled) => setListTagUntaggedOnly('image', enabled), onTagMatchMode: (mode) => setListTagMatchMode('image', mode), onEditTags: updateJobTags,
    onView: setImageView, onPage: (page) => setListPage('image', page), onPageSize: (size) => setListPageSize('image', size), onSort: (order) => setListSortOrder('image', order),
    onShow: (...args) => imageModalController.showImage(...args), onPrompt: (job, detail, text) => commonModalController.showPrompt('전체 프롬프트', detail, text),
    onClone: cloneImageJob, onContinueEditing: continueEditing, onGarment: (job) => imageModalController.openGarment(job), onFaceSwap: (job) => imageModalController.openFaceSwap(job),
    onDetail: detailEnhanceImage, onUpscale: upscaleImage, onCancel: cancelJob, onRetry: retryJob, onDelete: deleteJob
  }
  $: videoResultsModel = {
    jobs: videoJobs, pagedJobs: pagedVideoJobs, view: videoView, page: listPages.video, pageSize: listPageSizes.video, sortOrder: listSortOrders.video,
    upscaleOnline: engineStates.upscale === 'online', cancellingJob, retryingJob, deletingJob, updatingTagsJob,
    tagOptions: mediaListController.tagOptions('video'), filterTags: listTagFilters.video, excludedTags: listTagExclusions.video,
    untaggedOnly: listTagUntaggedOnly.video, tagMatchMode: listTagMatchModes.video,
    onFilterTags: (tags) => setListTagFilter('video', tags), onExcludeTags: (tags) => setListTagExclusions('video', tags),
    onUntaggedOnly: (enabled) => setListTagUntaggedOnly('video', enabled), onTagMatchMode: (mode) => setListTagMatchMode('video', mode), onEditTags: updateJobTags,
    progressFor: videoGenerationProgress, promptText: videoPromptModalText, onView: setVideoView,
    onPage: (page) => setListPage('video', page), onPageSize: (size) => setListPageSize('video', size), onSort: (order) => setListSortOrder('video', order),
    onShow: (job) => videoModalController.showVideo(job), onPrompt: (job, detail, text) => commonModalController.showPrompt('전체 프롬프트', detail, text),
    onShowUpscaleSource: (job) => videoModalController.showUpscaleSource(job), onLoadSettings: loadVideoJobSettings,
    onFrame: (job) => videoModalController.openFramePicker(job), onUpscale: (job) => videoModalController.openUpscale(job),
    onSendToRecognition: (job) => videoModalController.sendVideoToRecognition(job), onCancel: cancelJob, onRetry: retryJob, onDelete: deleteJob
  }
  $: recognitionResultsModel = {
    jobs: recognitionJobs, pagedJobs: pagedRecognitionJobs, view: subtitleView, page: listPages.recognition, pageSize: listPageSizes.recognition, sortOrder: listSortOrders.recognition,
    upscaleOnline: engineStates.upscale === 'online', cancellingJob, retryingJob, deletingJob, updatingTagsJob,
    tagOptions: mediaListController.tagOptions('recognition'), filterTags: listTagFilters.recognition, excludedTags: listTagExclusions.recognition,
    untaggedOnly: listTagUntaggedOnly.recognition, tagMatchMode: listTagMatchModes.recognition,
    onFilterTags: (tags) => setListTagFilter('recognition', tags), onExcludeTags: (tags) => setListTagExclusions('recognition', tags),
    onUntaggedOnly: (enabled) => setListTagUntaggedOnly('recognition', enabled), onTagMatchMode: (mode) => setListTagMatchMode('recognition', mode), onEditTags: updateJobTags,
    progressText: recognitionProgressText, progressTiming: recognitionProgressTiming, progressPercent: recognitionProgressPercent,
    warningText: subtitleTranslationWarningText, onView: setSubtitleView,
    onPage: (page) => setListPage('recognition', page), onPageSize: (size) => setListPageSize('recognition', size), onSort: (order) => setListSortOrder('recognition', order),
    onShow: (job) => videoModalController.showSubtitle(job), onWarning: (job, warnings, text) => commonModalController.showPrompt('번역 경고', `${warnings.length}개 자막은 원문을 유지했습니다.`, text),
    onRegenerate: (job) => videoModalController.openSubtitleRegenerate(job), onFrame: (job) => videoModalController.openFramePicker(job),
    onUpscale: (job) => videoModalController.openUpscale(job), onCancel: cancelJob, onRetry: retryJob, onDelete: deleteJob
  }
  $: imageCreationModel = {
    activeJobs, activeKreaModuleLabels, addIdentityReferences, addKreaRefs, addRefs, applyIdentityPreset, applySmartResolution, busy,
    checkpointVisible, config, disableAllKreaModules, enhanceImagePrompt, enhancingPrompt, filterModeDefault, filterModeMaximum,
    generateImage, hasKreaStyle, hasUserLora, identityPreserveItems, identityPreset, identityUI, imageCheckpointStatus,
    imageCloneMessage, clearImageCloneMessage: () => imageResultController.setState({ cloneMessage: '' }), imageDisabledMessage,
    imageEnhancementIsActive, imageEnhancementIsCurrent, isPureOutpaint, kreaAnyPaintImage, kreaAnyPaintMask, kreaAnyPaintMaskPreview,
    kreaAnyPaintPreview, kreaDepthImage, kreaDepthPreview, kreaIdentityImage, kreaIdentityMask, kreaIdentityMaskPreview,
    kreaIdentityPreview, kreaIdentityReferences, kreaModuleMessage, kreaModules, kreaNK2EImage, kreaNK2EPreprocessed,
    kreaNK2EPreview, kreaStrictMask, kreaStrictMaskPreview, kreaStyleLabel, kreaStyleReferenceImages, kreaStyleSelections,
    kreaVisionImages, looksLikeStructuredPrompt, openGarmentExtractor: (job) => imageModalController.openGarment(job), openFaceSwap: (job) => imageModalController.openFaceSwap(job),
    openImageSequence: () => imageModalController.openSequence(),
    openPromptExamples: (target) => commonModalController.openPromptExamples(target),
    rawImagePrompt, refreshUserLoras, refs, removeIdentityReference, removeKreaRef, removeRef, resetImageCreation, resetImageEnhancement,
    selectKreaCheckpoint, selectedKreaCheckpoint, selectedKreaCheckpointSource, setKreaImage,
    showImage: (...args) => imageModalController.showImage(...args), showImageOnKey: (...args) => imageModalController.showImageOnKey(...args),
    toggleIdentityPreserveItem, toggleKreaModule, toggleKreaStyle, toggleUserLora, updateKreaStyleStrength, updateUserLoraStrength,
    useCustomImageResolution, userLoraCatalog, userLoraLabel, userLoraSelections
  }
  $: videoCreationModel = {
    activeJobs, addVideoKeyframe, applyVideoResolutionPreset, busy, config, createVideoPromptFromScenes, creatingVideoPrompt,
    currentVideoResolutionPreset, enhanceVideoPrompt, enhancingPrompt, generateVideo, moveVideoAudio, moveVideoKeyframe,
    normalizeVideoTiming, openPromptExamples: (target) => commonModalController.openPromptExamples(target), removeVideoAudio,
    removeVideoKeyframe, resetVideoCreation, resetVideoEnhancement, setVideoConditionImage,
    showImage: (...args) => imageModalController.showImage(...args), updateVideoKeyframe, videoAccelerationPreview, videoAudioClips,
    videoAudioJob, videoConditioningDisabledReason, videoEndImage, videoEnhancementIsActive, videoEnhancementIsCurrent,
    videoImage, videoImagePreview, videoKeyframeCapacity, videoKeyframes, videoPromptCreationMessage
  }
  $: speechTabModel = {
    mobilePane: mobileSpeechPane, form: speechForm, busy, activeJobs: activeJobs(), jobs: speechJobs, pagedJobs: pagedSpeechJobs,
    view: speechView, page: listPages.speech, pageSize: listPageSizes.speech, sortOrder: listSortOrders.speech,
    cancellingJob, retryingJob, deletingJob, updatingTagsJob, progressFor: speechGenerationProgress,
    tagOptions: mediaListController.tagOptions('speech'), filterTags: listTagFilters.speech, excludedTags: listTagExclusions.speech,
    untaggedOnly: listTagUntaggedOnly.speech, tagMatchMode: listTagMatchModes.speech,
    onFilterTags: (tags) => setListTagFilter('speech', tags), onExcludeTags: (tags) => setListTagExclusions('speech', tags),
    onUntaggedOnly: (enabled) => setListTagUntaggedOnly('speech', enabled), onTagMatchMode: (mode) => setListTagMatchMode('speech', mode), onEditTags: updateJobTags,
    onMobilePane: (pane) => setMobilePane('speech', pane), onReset: resetSpeechCreation, onGenerate: generateSpeech, onView: setSpeechView,
    onPage: (page) => setListPage('speech', page), onPageSize: (size) => setListPageSize('speech', size), onSort: (order) => setListSortOrder('speech', order),
    onPrompt: (job) => commonModalController.showPrompt('음성 원문', `${job.params?.speaker || 'VOICE'}${job.params?.seed >= 0 ? ` · seed ${job.params.seed}` : ''}`, job.prompt || ''),
    onShow: (job) => videoModalController.showAudio(job), onSendToVideo: (job) => videoModalController.sendAudioToVideo(job),
    onCancel: cancelJob, onRetry: retryJob, onDelete: deleteJob
  }
  $: recognitionFormModel = {
    form: recognitionForm, file: recognitionFile, sourceVideoJob: recognitionSourceVideoJob, options: recognitionOptions,
    selectedPart: selectedRecognitionPart(), loadingOptions: loadingRecognitionOptions, config, busy, activeJobs: activeJobs(),
    onReset: resetRecognitionCreation, onSubmit: recognizeSpeech, onClearSourceVideo: clearRecognitionSourceVideo,
    onURL: updateRecognitionURL, onLoadOptions: loadRecognitionOptions, onFile: updateRecognitionFile,
    onOpenVideoPicker: () => $videoModalState.recognitionVideoPickerOpen = true, onClearFile: clearRecognitionFile, onSelectPart: selectRecognitionPart
  }
  $: settingsTabModel = {
    settings, savedMessage, engineStates, imageCheckpointStatus, videoModelStatus, savingDownloadCredentials,
    preparingImageCheckpoints, convertingImageCheckpoints, preparingVideoModels, storage, cleaningStorage, busy,
    saveSettings, saveDownloadCredentials, displayCheckpointReady, checkpointVisible, setCheckpointVisible,
    prepareImageCheckpoints, convertImageCheckpointsNVFP4, prepareVideoModels, cleanupTemporaryStorage, snapVideoDuration
  }
  $: historyTabModel = {
    jobs, pagedJobs: pagedHistoryJobs, page: listPages.history, pageSize: listPageSizes.history, sortOrder: listSortOrders.history,
    deletingJob, retryingJob, activeJobs: activeJobs(), onClear: clearFinishedJobs,
    onPage: (page) => setListPage('history', page), onPageSize: (size) => setListPageSize('history', size), onSort: (order) => setListSortOrder('history', order),
    onPrompt: (job) => commonModalController.showPrompt(`${kindLabels[job.kind] || job.kind} 작업`, `${new Date(job.created_at).toLocaleString()} · ${statusLabels[job.status] || job.status}`, job.prompt),
    onRetry: retryJob, onDelete: deleteJob
  }
  $: imageModalModel = {
    busy, imageForm, kreaOptions, kreaModules, imageSequenceEntryMode, imageSequenceStoryIdea, imageSequenceSceneCount,
    imageSequencePrompts, imageSequenceEnhancedPrompts, imageSequenceSceneTitles, imageSequenceSharedPrompt, imageSequenceSharedPromptEdited,
    imageSequencePlanning, imageSequencePlanError, imageSequenceCharacters, setImageSequenceEntryMode, setImageSequenceStoryIdea, setImageSequenceSceneCount,
    setImageSequenceSharedPrompt,
    applyStorySequenceExample, applySceneSequenceExample, planImageSequence,
    imageSequenceBlockedMessage, removeImageSequenceScene, moveImageSequenceScene, updateImageSequencePrompt,
    addImageSequenceScene, addImageSequenceCharacter, removeImageSequenceCharacter, setImageSequenceCharacterName,
    addImageSequenceCharacterFiles, addImageSequenceCharacterResult, removeImageSequenceCharacterReference,
    setImageSequenceCharacterReIDReference, toggleImageSequenceCharacterTrait,
    generateImageSequenceCharacterSheet, approveImageSequenceCharacterSheet, discardImageSequenceCharacterSheet,
    analyzeImageSequenceCharacter, setImageSequenceCharacterDescription, setImageSequenceCharacterPrompt,
    imageSequenceCharacterPreview, imageSequenceCharacterReadinessMessage, generateImage, kreaAnyPaintPreview,
    kreaIdentityPreview, kreaAnyPaintMaskPreview, kreaIdentityMaskPreview, kreaStrictMaskPreview, kreaNK2EPreview,
    kreaNK2EPreprocessed, imageJobs, identityUI, kreaIdentityReference, kreaDepthImage, kreaNK2EImage,
    kreaAnyPaintImage, kreaIdentityImage, submitGarmentExtraction, submitFaceSwap
  }
  $: videoModalModel = {
    videoJobs, recognitionJobs, speechJobs, imageJobs, recognitionSourceVideoJob, videoAudioClips, videoDurationSeconds,
    videoImage, videoEndImage, videoKeyframes, usePickedVideoFrame, togglePickedVideoAudio, moveVideoKeyframe, moveVideoAudio,
    updateVideoKeyframe, removeVideoKeyframe, addVideoKeyframe, setVideoConditionImage, videoImagePreview
  }

  function switchAssistantTab(nextTab, results = false) {
    assistantController.switchTab(nextTab, results)
  }

  function applyAssistantActions(actions = []) {
    return assistantController.applyActions(actions)
  }

  async function executeAssistantOperation(kind) {
    return assistantController.execute(kind)
  }

  async function cleanupTemporaryStorage() {
    return settingsController.cleanupStorage(confirm, formatBytes)
  }

  function toggleColorTheme() {
    colorTheme = colorTheme === 'light' ? 'dark' : 'light'
    document.documentElement.dataset.theme = colorTheme
    document.documentElement.style.colorScheme = colorTheme
    localStorage.setItem('spark-media-theme', colorTheme)
  }

  async function saveSettings() {
    const result = await settingsController.saveConfig(settings, settingsVideoDurationSeconds)
    if (!result) return
    applyRuntimeConfig(result.config, false)
    settingsVideoDurationSeconds = durationFromFrames(result.config.video.default_frames, result.config.video.default_fps)
    await refresh()
  }

</script>

<svelte:window onkeydown={handleFeatureModulesKeydown} />

<datalist id="translation-languages">
  {#each translationLanguages as language}<option value={language}></option>{/each}
</datalist>

<svelte:head><meta name="theme-color" content={colorTheme === 'light' ? '#f2f5f1' : '#101318'}></svelte:head>

<AppHeader {colorTheme} {engineAggregate} {engineAggregateLabel} {engineStates} {imageForm} bind:mobileEngineOpen {monitoredEngineStatuses} {recognitionForm} {systemUsage} {tab} {toggleColorTheme} />

<main>
  <AppNavigation {jobs} {openSettings} {refreshUserLoras} bind:tab />

  {#if error}<div class="error"><span>{error}</span><button onclick={() => error = ''}>×</button></div>{/if}

  {#if tab === 'image'}
    <div class="mobile-image-nav" role="tablist" aria-label="모바일 이미지 화면">
      <button type="button" role="tab" aria-selected={mobileImagePane === 'create'} class:active={mobileImagePane === 'create'} onclick={() => setMobilePane('image', 'create')}><span>만들기</span><small>설정·기능 모듈</small></button>
      <button type="button" role="tab" aria-selected={mobileImagePane === 'results'} class:active={mobileImagePane === 'results'} onclick={() => setMobilePane('image', 'results')}><span>생성 이미지 목록</span><small>{imageJobs.length}개{#if activeJobs().some((job) => job.kind === 'image')} · 생성 중{/if}</small></button>
    </div>
    <section class="workspace image-workspace" class:mobile-results={mobileImagePane === 'results'}>
      <ImageCreationPanel
        model={imageCreationModel}
        bind:cannyEditorOpen={$imageModalState.cannyEditorOpen}
        bind:featureModulesOpen={$imageModalState.featureModulesOpen}
        bind:filterPromptPreset
        bind:identityPreserveCustom
        bind:imageAspectRatio
        bind:imageEnhanceEnabled
        bind:imageEnhancedPrompt
        bind:imageForm
        bind:imageMegapixels
        bind:imageResolutionMode
        bind:kreaOptions
        bind:maskEditorMode={$imageModalState.maskEditorMode}
        bind:parentImageJobID
        bind:presetImagePickerTarget={$imageModalState.presetPickerTarget}
        bind:recentImagePickerTarget={$imageModalState.recentPickerTarget}
        bind:remoteImageTarget={$imageModalState.remoteTarget}
        bind:runtimeInfoOpen={$imageModalState.runtimeInfoOpen}
      />
      <MediaResultsLayer kind="image" model={imageResultsModel} />
    </section>
  {:else if tab === 'video'}
    <div class="mobile-image-nav" role="tablist" aria-label="모바일 영상 화면">
      <button type="button" role="tab" aria-selected={mobileVideoPane === 'create'} class:active={mobileVideoPane === 'create'} onclick={() => setMobilePane('video', 'create')}><span>만들기</span><small>영상 생성 설정</small></button>
      <button type="button" role="tab" aria-selected={mobileVideoPane === 'results'} class:active={mobileVideoPane === 'results'} onclick={() => setMobilePane('video', 'results')}><span>생성 영상 목록</span><small>{videoJobs.length}개{#if activeJobs().some((job) => job.kind === 'video')} · 생성 중{/if}</small></button>
    </div>
    <section class="workspace mobile-media-workspace" class:mobile-results={mobileVideoPane === 'results'}>
      <VideoCreationPanel
        model={videoCreationModel}
        bind:videoAdvancedOpen
        bind:videoAudioPickerOpen={$videoModalState.audioPickerOpen}
        bind:videoDurationSeconds
        bind:videoEndStrength
        bind:videoEnhanceEnabled
        bind:videoEnhancedPrompt
        bind:videoForm
        bind:videoImagePickerTarget={$videoModalState.imagePickerTarget}
        bind:videoPromptPreset
        bind:videoRemoteImageTarget={$videoModalState.remoteImageTarget}
        bind:videoTimelineEditorOpen={$videoModalState.timelineEditorOpen}
      />
      <MediaResultsLayer kind="video" model={videoResultsModel} />
    </section>
  {:else if tab === 'speech'}
    <SpeechTab {...speechTabModel} />
  {:else if tab === 'recognition'}
    <div class="mobile-image-nav" role="tablist" aria-label="모바일 받아쓰기 화면">
      <button type="button" role="tab" aria-selected={mobileRecognitionPane === 'create'} class:active={mobileRecognitionPane === 'create'} onclick={() => setMobilePane('recognition', 'create')}><span>만들기</span><small>받아쓰기 설정</small></button>
      <button type="button" role="tab" aria-selected={mobileRecognitionPane === 'results'} class:active={mobileRecognitionPane === 'results'} onclick={() => setMobilePane('recognition', 'results')}><span>생성 자막 목록</span><small>{recognitionJobs.length}개{#if activeJobs().some((job) => job.kind === 'recognition')} · 처리 중{/if}</small></button>
    </div>
    <section class="workspace mobile-media-workspace" class:mobile-results={mobileRecognitionPane === 'results'}>
      <RecognitionForm {...recognitionFormModel} bind:fileInput={recognitionFileInput} />
      <MediaResultsLayer kind="recognition" model={recognitionResultsModel} />
    </section>
  {:else if tab === 'lora'}
    <LoraStudio {imageJobs} onChanged={refreshUserLoras} onOpenSettings={openSettings} />
  {:else if tab === 'settings' && settings}
    <SettingsTab
      {...settingsTabModel}
      bind:settingsSection
      bind:civitaiToken
      bind:hfToken
      bind:checkpointSelection
      bind:nvfp4Selection
      bind:removeBF16Sources
      bind:settingsVideoDurationSeconds
    />
  {:else}
    <HistoryTab {...historyTabModel} />
  {/if}
</main>

<ImageModalPanel controller={imageModalController} modalState={imageModalState} model={imageModalModel} />
<VideoModalPanel
  controller={videoModalController}
  modalState={videoModalState}
  model={videoModalModel}
  bind:videoForm
  bind:videoEndStrength
/>
<AssistantChat state={assistantState} onActions={applyAssistantActions} onExecute={executeAssistantOperation} getVisualContext={videoAssistantVisualContext} />
<CommonModalLayer
  controller={commonModalController}
  prompt={$commonModalState.prompt}
  promptExamplesOpen={$commonModalState.promptExamplesOpen}
  promptExamplesTarget={$commonModalState.promptExamplesTarget}
  imageSelectedID={filterPromptPreset}
  videoSelectedID={videoPromptPreset}
  videoHasConditionImage={Boolean(videoImage) || Boolean(videoEndImage) || videoKeyframes.some((item) => item.image)}
  {createRandomVideoPrompt}
/>
