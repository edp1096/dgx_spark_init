import defaults from '../../../internal/config/assets/prompt_composer.defaults.json' with { type: 'json' };

export const promptCategories = defaults.categories;
export const selectionGroups = [ ['', '여러 개 함께 사용'], ['response_language', '답변 언어 중 하나'], ['speech', '존댓말·반말 중 하나'], ['length', '답변 길이 중 하나'], ['level', '설명 수준 중 하나'], ['clarification', '질문 방식 중 하나'] ];
export function ensureComposer(model) {
  model.prompt_composer ??= { enabled: false, blocks: structuredClone(defaults.blocks), combinations: [], persona_id: '', condition_ids: [], extra: model.system_prompt || '' };
  const c = model.prompt_composer;
  c.blocks ||= []; c.combinations ||= []; c.condition_ids ||= []; c.persona_id ||= ''; c.extra ||= '';
  return c;
}
export function renderPrompt(c) {
  const parts = [], selected = new Set(c.condition_ids || []);
  const identity = [];
  if (c.character_name?.trim()) identity.push(`대화에서 사용하는 너의 이름은 ${c.character_name.trim()}이다.`);
  if (c.character_description?.trim()) identity.push(c.character_description.trim());
  if (identity.length) parts.push(`[AI 캐릭터]\n${identity.join('\n')}`);
  const persona = c.blocks.find(b => b.kind === 'persona' && b.id === c.persona_id);
  if (persona) {
    parts.push(`[페르소나]\n${persona.prompt.trim()}`);
    if (selected.size) parts.push('말투·형식이 충돌하면 페르소나의 기본 표현보다 아래 추가 조건을 따른다.');
  }
  for (const category of promptCategories) {
    const lines = c.blocks.filter(b => b.kind === 'condition' && b.category === category.id && selected.has(b.id)).map(b => b.prompt.trim());
    if (lines.length) parts.push(`[${category.label}]\n${lines.join('\n')}`);
  }
  if (c.extra?.trim()) parts.push(`[직접 추가]\n${c.extra.trim()}`);
  return parts.join('\n\n');
}
export function toggleCondition(c, id, checked) {
  const block = c.blocks.find(b => b.id === id);
  if (!block || block.kind !== 'condition') return;
  c.condition_ids = c.condition_ids.filter(other => other !== id && (!checked || !block.group || c.blocks.find(b => b.id === other)?.group !== block.group));
  if (checked) c.condition_ids.push(id);
}
export function selectionOf(c) {
  return { persona_id: c.persona_id || '', condition_ids: [...c.condition_ids], extra: c.extra || '' };
}
export function selectionError(c, s = c) {
  if (s.persona_id && !c.blocks.some(b => b.kind === 'persona' && b.id === s.persona_id)) return '페르소나를 찾을 수 없습니다.';
  const groups = new Set(), ids = new Set();
  for (const id of s.condition_ids || []) {
    const b = c.blocks.find(b => b.id === id && b.kind === 'condition');
    if (!b) return '선택한 조건을 찾을 수 없습니다.';
    if (ids.has(id) || (b.group && groups.has(b.group))) return '함께 선택할 수 없는 조건이 있습니다.';
    ids.add(id); if (b.group) groups.add(b.group);
  }
  return '';
}
export function switchPromptMode(model, enabled) {
  const c = ensureComposer(model);
  if (enabled && !c.enabled) {
    const text = (model.system_prompt || '').trim();
    if (text !== renderPrompt(c)) {
      const exact = c.blocks.find(b => b.prompt.trim() === text && text);
      c.persona_id = exact?.kind === 'persona' ? exact.id : '';
      c.condition_ids = exact?.kind === 'condition' ? [exact.id] : [];
      c.extra = exact ? '' : text;
    }
  }
  if (!enabled && c.enabled) model.system_prompt = renderPrompt(c);
  c.enabled = enabled;
  if (enabled) model.system_prompt_preset = '';
}
export function removeBlock(c, id) {
  c.blocks = c.blocks.filter(b => b.id !== id);
  for (const s of [c, ...c.combinations]) {
    if (s.persona_id === id) s.persona_id = '';
    s.condition_ids = s.condition_ids.filter(other => other !== id);
  }
}
