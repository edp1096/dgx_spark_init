export function appendMediaInput(form, uploadField, reuseField, image) {
  if (!image) return
  if (image.server) form.append(reuseField, image.ref)
  else form.append(uploadField, image.file || image)
}

export function buildImageGenerationForm(input) {
  const {
    imageForm, prompt, originalPrompt, sequence, parentJobID,
    modules, options, identity, depth, styles, userLoras,
    visionImages, styleReferenceImages, nk2e, anypaint, references
  } = input
  const form = new FormData()
  Object.entries(imageForm).forEach(([key, value]) => form.append(key, key === 'prompt' ? prompt : value))
  form.append('original_prompt', originalPrompt)

  if (sequence) {
    form.append('sequence_prompts', JSON.stringify(sequence.prompts.map((value) => value.trim())))
    form.append('sequence_enhanced_prompts', JSON.stringify(sequence.enhancedPrompts.map((value) => value.trim())))
    form.append('sequence_shared_prompt', sequence.sharedPrompt || '')
    form.append('sequence_canonical_prompt', sequence.canonicalPrompt || '')
    appendMediaInput(form, 'sequence_character_image', 'reuse_sequence_character_image', sequence.reidImage)
  }
  if (parentJobID) form.append('parent_job_id', parentJobID)

  if (imageForm.mode === 'create') {
    for (const key of ['steps', 'checkpoint', 'filter_mode', 'filter_strength', 'prompt_enhancer', 'prompt_enhancer_strength', 'prompt_text_scale', 'sampling_preset']) {
      form.append(key, options[key])
    }
    if (modules.identity) {
      form.append('identity_preset', identity.preset)
      if (identity.preset === 'tryon') {
        form.append('identity_auto_prompt', 'true')
        form.append('identity_user_prompt', identity.hasUserPrompt ? 'true' : 'false')
      }
      form.append('identity_preserve_items', JSON.stringify(identity.preserveItems))
      form.append('identity_preserve_custom', identity.preserveCustom)
      appendMediaInput(form, 'identity_image', 'reuse_identity_image', identity.image)
      identity.references.forEach((image) => appendMediaInput(form, 'identity_reference', 'reuse_identity_reference', image))
      appendMediaInput(form, 'identity_mask', 'reuse_identity_mask', identity.mask)
      appendMediaInput(form, 'strict_mask', 'reuse_strict_mask', identity.strictMask)
      for (const key of ['identity_strength', 'ref_boost', 'source_ref_boost', 'grounding_px', 'strict_mask_grow', 'strict_mask_feather', 'vae_mode', 'identity_fit_mode', 'identity_model', 'identity_encoder']) {
        form.append(key, options[key])
      }
    }
    if (modules.depth) {
      appendMediaInput(form, 'depth_image', 'reuse_depth_image', depth.image)
      form.append('depth_strength', options.depth_strength)
      if (depth.image?.posePrompt) {
        form.append('depth_pose_prompt', depth.image.posePrompt)
        form.append('prepare_pose_reference', 'true')
      }
    }
    if (modules.style) form.append('styles', JSON.stringify(styles))
    if (modules.userLora) form.append('user_loras', JSON.stringify(userLoras))
    if (modules.vision) {
      visionImages.forEach((image) => appendMediaInput(form, 'vision_images', 'reuse_vision_images', image))
      form.append('vision_mode', options.vision_mode)
      form.append('vision_megapixels', options.vision_megapixels)
    }
    if (modules.styleReference) {
      styleReferenceImages.forEach((image) => appendMediaInput(form, 'style_reference_images', 'reuse_style_reference_images', image))
      form.append('style_reference_strength', options.style_reference_strength)
    }
    if (modules.nk2e) {
      appendMediaInput(form, 'nk2e_image', 'reuse_nk2e_image', nk2e.image)
      form.append('nk2e_mode', options.nk2e_mode)
      form.append('nk2e_strength', options.nk2e_strength)
      form.append('nk2e_preprocessed', nk2e.preprocessed)
    }
    if (modules.anypaint) {
      appendMediaInput(form, 'anypaint_image', 'reuse_anypaint_image', anypaint.image)
      appendMediaInput(form, 'anypaint_mask', 'reuse_anypaint_mask', anypaint.mask)
      for (const key of ['outpaint_left', 'outpaint_top', 'outpaint_right', 'outpaint_bottom', 'anypaint_strength', 'anypaint_boundary_redraw_px']) {
        form.append(key, options[key])
      }
    }
  }
  references.forEach((image) => appendMediaInput(form, 'references', 'reuse_references', image))
  return form
}

export function buildVideoGenerationForm(input) {
  const { videoForm, prompt, originalPrompt, numFrames, startImage, endImage, endStrength, audioClips, keyframes } = input
  const form = new FormData()
  Object.entries(videoForm).forEach(([key, value]) => form.append(key, key === 'prompt' ? prompt : value))
  form.append('num_frames', numFrames)
  form.append('original_prompt', originalPrompt)
  appendMediaInput(form, 'start_image', 'reuse_start_image', startImage)
  appendMediaInput(form, 'end_image', 'reuse_end_image', endImage)
  form.append('end_image_strength', endStrength)
  form.append('audio_count', audioClips.length)
  audioClips.forEach((clip, index) => {
    form.append(`reuse_audio_job_${index}`, clip.job.id)
    form.append(`audio_start_${index}`, clip.start)
    form.append(`audio_duration_${index}`, clip.duration || 0)
  })
  const selectedKeyframes = keyframes.filter((keyframe) => keyframe.image)
  form.append('keyframe_count', selectedKeyframes.length)
  selectedKeyframes.forEach((keyframe, index) => {
    appendMediaInput(form, `keyframe_image_${index}`, `reuse_keyframe_image_${index}`, keyframe.image)
    form.append(`keyframe_time_${index}`, keyframe.time)
    form.append(`keyframe_strength_${index}`, keyframe.strength)
  })
  return form
}
