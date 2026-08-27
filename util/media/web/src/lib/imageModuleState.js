const clearTargetsByModule = {
  identity: ['identity', 'identityReference'],
  depth: ['depth'],
  nk2e: ['nk2e'],
  anypaint: ['anypaint', 'anypaintMask'],
  vision: ['vision'],
  styleReference: ['styleReference']
}

export function toggleImageModuleState({ module, modules, options, preserveItems, imagePixels }) {
  const enabled = !modules[module]
  const result = {
    enabled,
    modules: { ...modules, [module]: enabled },
    options,
    preserveItems,
    imageMegapixels: null,
    imageResolutionMode: null,
    applySmartResolution: false,
    message: '',
    clearTargets: enabled ? [] : (clearTargetsByModule[module] || [])
  }
  if (!enabled) return result
  if (module === 'depth') result.preserveItems = preserveItems.filter((id) => id !== 'pose' && id !== 'composition')
  if (module !== 'identity') return result
  result.options = {
    ...options,
    steps: Math.max(10, Number(options.steps) || 0),
    filter_mode: 'off',
    filter_strength: 0
  }
  if (imagePixels > 2 * 1024 * 1024) {
    result.imageMegapixels = 2
    result.imageResolutionMode = 'smart'
    result.applySmartResolution = true
    result.message = 'Identity 편집은 최대 2MP이므로 이미지 크기를 고해상도 2MP로 조정했습니다.'
  }
  return result
}

export function toggleUserLoraSelection(selections, catalog, filename, limit = 5) {
  if (selections.some((selection) => selection.filename === filename)) {
    return selections.filter((selection) => selection.filename !== filename)
  }
  if (selections.length >= limit) return selections
  const lora = catalog.find((item) => item.filename === filename)
  const recommended = Number(lora?.recommended_strength)
  return [...selections, {
    filename,
    strength: Number.isFinite(recommended) ? recommended : 1
  }]
}

export function toggleStyleSelection(selections, name) {
  return selections.some((style) => style.name === name)
    ? selections.filter((style) => style.name !== name)
    : [...selections, { name, strength: 1 }]
}

export function updateSelectionStrength(selections, key, value, strength) {
  return selections.map((selection) => selection[key] === value ? { ...selection, strength: Number(strength) } : selection)
}

export function selectionLabel(catalog, key, value, labelKey) {
  return catalog.find((item) => item[key] === value)?.[labelKey] || value
}
