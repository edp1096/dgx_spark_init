export function hydrateMessages(items) {
  const hydrated = items.map((item) => ({
    ...item,
    variant_index: Math.max(0, (item.variants?.length || 1) - 1),
    activity: '',
  }));
  for (let index = 1; index < hydrated.length; index += 1) {
    const message = hydrated[index];
    const parent = hydrated[index - 1];
    if (message.role !== 'assistant' || parent?.role !== 'user' || !message.variants?.length) continue;
    const matching = message.variants
      .map((variant, variantIndex) => ({ variant, variantIndex }))
      .filter(({ variant }) => (variant.parent_variant ?? 0) === (parent.variant_index ?? 0));
    if (!matching.length) continue;
    const selected = matching[matching.length - 1];
    applyVariant(message, selected.variantIndex);
  }
  return hydrated;
}

export function variantIndices(message, messageIndex, messageList) {
  const indices = (message.variants || []).map((_, index) => index);
  if (message.role !== 'assistant') return indices;
  const parent = messageList[messageIndex - 1];
  if (parent?.role !== 'user') return indices;
  return indices.filter((index) => (message.variants[index].parent_variant ?? 0) === (parent.variant_index ?? 0));
}

export function applyVariant(message, variantIndex) {
  const variant = message.variants?.[variantIndex];
  if (!variant) return false;
  message.content = variant.content || '';
  message.reasoning_content = variant.reasoning_content || '';
  message.tool_trace = variant.tool_trace || [];
  message.attachments = variant.attachments || [];
  message.variant_index = variantIndex;
  return true;
}
