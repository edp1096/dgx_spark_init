// Services are shared recipes. A set stores only its own deployment overrides.
export function resolveSetMembers(catalog, bundle) {
  if (!catalog || !bundle) return [];
  const definitions = new Map(catalog.components.map(component => [component.id, component]));
  return bundle.components.filter(id => definitions.has(id)).map(id => ({
    ...definitions.get(id), ...Object.fromEntries(Object.entries(bundle.bindings?.[id] || {}).filter(([, value]) => value != null)),
  }));
}

export function setDeploymentValue(bundle, componentID, field, value) {
  bundle.bindings ||= {};
  bundle.bindings[componentID] = { ...bundle.bindings[componentID], [field]: value };
}

export function resetDeployment(bundle, componentID) {
  if (bundle.bindings) delete bundle.bindings[componentID];
}

export function hostIsUsed(catalog, hostID) {
  return catalog.components.some(c => c.host === hostID || c.worker_host === hostID)
    || catalog.bundles.some(b => resolveSetMembers(catalog, b).some(c => c.host === hostID || c.worker_host === hostID));
}
