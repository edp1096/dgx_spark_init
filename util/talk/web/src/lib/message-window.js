export const INITIAL_MESSAGE_WINDOW = 32;
export const MAX_MESSAGE_WINDOW = 48;
export const MESSAGE_WINDOW_CHUNK = 24;

export function initialMessageStart(length, size = INITIAL_MESSAGE_WINDOW) {
  return Math.max(0, Number(length || 0) - size);
}

export function shiftedMessageWindow(length, start, end, direction, size = MAX_MESSAGE_WINDOW, step = MESSAGE_WINDOW_CHUNK) {
  if (direction === 'previous') {
    const nextStart = Math.max(0, start - step);
    return { start: nextStart, end: Math.min(length, nextStart + size) };
  }
  const nextEnd = Math.min(length, end + step);
  return { start: Math.max(0, nextEnd - size), end: nextEnd };
}

export function messageWindowAround(length, targetIndex, size = MAX_MESSAGE_WINDOW) {
  if (targetIndex < 0 || targetIndex >= length) return null;
  const start = Math.max(0, Math.min(targetIndex - Math.floor(size / 4), length - size));
  return { start, end: Math.min(length, start + size) };
}
