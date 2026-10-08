export const outputPresets = [
  { output: 8192, label: '32K 문맥 · 출력 8K' },
  { output: 16384, label: '64K 문맥 · 출력 16K' },
  { output: 32768, label: '128K 문맥 · 출력 32K' },
  { output: 65536, label: '256K–1M 이상 문맥 · 출력 64K' },
];

export function selectedOutputPreset(context) {
  if (context?.output_auto) return 'auto';
  return outputPresets.some(p => p.output === Number(context?.output_reserve)) ? String(context.output_reserve) : 'custom';
}

export function applyOutputPreset(context, preset) {
  context.output_auto = preset === 'auto';
  const selected = outputPresets.find(p => String(p.output) === preset);
  if (selected) context.output_reserve = selected.output;
}

// Preview only: the server computes the same budget after context detection.
export function effectiveOutputReserve(context) {
  const window = Number(context?.window_tokens);
  if (!context?.output_auto || !(window > 0)) return context?.output_reserve;
  for (const size of [262144, 131072, 65536, 32768]) {
    if (window >= size) return Math.min(size / 4, Math.max(1, Math.floor((window - context.safety_margin) / 2)));
  }
  return Math.max(1, Math.floor((window - context.safety_margin) / 4));
}
