const GENERIC_REASONING_LEVELS = ['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'];
const QWEN_REASONING_LEVELS = ['none', 'low', 'medium', 'xhigh'];
const GLM53_REASONING_LEVELS = ['off', 'low', 'high', 'max'];
const THINKING_OFF_VALUES = new Set(['', '0', '0.0', 'none', 'off', 'false', 'no_think', 'disabled']);

export function modelCapabilities(modelType) {
  const type = String(modelType || '').trim().toLowerCase();
  if (type === 'gemma4') {
    return { family: 'gemma4', reasoning: 'toggle', reasoningLevels: ['on', 'none'] };
  }
  if (type === 'qwen3.8-exl3') {
    return { family: 'qwen3.8-exl3', reasoning: 'toggle', reasoningLevels: ['on', 'none'] };
  }
  if (type === 'qwen3.8') {
    return { family: 'qwen3.8', reasoning: 'effort', reasoningLevels: QWEN_REASONING_LEVELS };
  }
  if (type === 'glm5.3') {
    return { family: 'glm5.3', reasoning: 'effort', reasoningLevels: GLM53_REASONING_LEVELS };
  }
  return { family: 'generic', reasoning: 'effort', reasoningLevels: GENERIC_REASONING_LEVELS };
}

export function thinkingToggleValue(value) {
  return THINKING_OFF_VALUES.has(String(value || '').trim().toLowerCase()) ? 'none' : 'on';
}

export function normalizeReasoningEffort(modelType, value) {
  const profile = modelCapabilities(modelType);
  const normalized = String(value || '').trim().toLowerCase();
  if (profile.family === 'qwen3.8') {
    return profile.reasoningLevels.includes(normalized) ? normalized : 'medium';
  }
  if (profile.family === 'glm5.3') {
    if (THINKING_OFF_VALUES.has(normalized)) return 'off';
    return profile.reasoningLevels.includes(normalized) ? normalized : 'max';
  }
  if (profile.reasoning === 'toggle') return thinkingToggleValue(normalized);
  return String(value || '').trim();
}

export function reasoningEffortLabel(value) {
  switch (value) {
    case 'off':
    case 'none': return '꺼짐';
    case 'low': return 'Low';
    case 'high': return 'High';
    case 'xhigh': return 'XHigh';
    case 'max': return 'Max';
    case 'medium': return 'Medium';
    default: return String(value || '');
  }
}
