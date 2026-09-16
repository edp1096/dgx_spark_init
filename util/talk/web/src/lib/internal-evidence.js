// App reference records are not answer text. Keep literal code examples intact.
export function stripInternalEvidence(value) {
  let fence = '';
  const marker = '[Historical tool evidence:';
  return String(value || '').split('\n').filter(line => {
    const trimmed = line.trim();
    if (/^(?:```|~~~)/.test(trimmed)) {
      const next = trimmed.slice(0, 3);
      if (!fence) fence = next; else if (fence === next) fence = '';
      return true;
    }
    if (fence) return true;
    if (/^\[Historical tool evidence: [\w.-]+; archive_id=\d+; use context_read for original\]/.test(trimmed)) return false;
    // Suppress a marker while its streamed line is still incomplete.
    if (trimmed.length >= 11 && marker.startsWith(trimmed)) return false;
    if (trimmed.startsWith(marker) && !trimmed.includes(']')) return false;
    return true;
  }).join('\n');
}
