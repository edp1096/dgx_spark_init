// Labels come from the runtime catalog; selected values remain API model IDs.
export function modelDisplayName(modelID, catalog, preferredBundle = '') {
 if (!modelID) return '';
 const bundles = catalog?.bundles || [];
 const components = [...(catalog?.components || []), ...(catalog?.support_services || [])];
 const ordered = [...bundles.filter(b => b.id === preferredBundle), ...bundles.filter(b => b.id !== preferredBundle)];
 for (const bundle of ordered) {
  const boundLLM = Object.entries(bundle.bindings || {}).some(([id, binding]) =>
   binding?.model === modelID && components.some(c => c.id === id && c.role === 'llm'));
  if (bundle.model_id === modelID || boundLLM) return bundle.name;
 }
 return components.find(c => c.model === modelID)?.name || modelID;
}

export function modelWeightOptions(component) { return component?.model_presentation?.weights || []; }

export function selectedModelWeight(component) {
 const weights = modelWeightOptions(component);
 return weights.find(w => w.id === component?.runtime_options?.MODEL_VARIANT)
  || weights.find(w => w.model_id && w.model_id === component?.model)
  || weights.find(w => w.id === component?.model_presentation?.selected_variant)
  || weights[0];
}

export function modelWeightSummary(component) {
 const weight = selectedModelWeight(component);
 return weight ? [weight.label, weight.format].filter(Boolean).join(' · ') : '';
}
