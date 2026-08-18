export const attachmentAccept = [
  'image/png', 'image/jpeg', 'image/webp',
  'audio/mpeg', 'audio/wav', 'audio/ogg',
  'video/x-msvideo', 'video/quicktime', 'video/mp4', 'video/ogg', 'video/x-ms-wmv', 'video/webm',
  '.png', '.jpg', '.jpeg', '.webp', '.mp3', '.wav', '.ogg', '.avi', '.mov', '.mp4', '.wmv', '.webm',
].join(',');

export const maxImageBytes = 15 * 1024 * 1024;
export const maxAttachmentBytes = 64 * 1024 * 1024;
export const maxMessageBytes = 96 * 1024 * 1024;

const supportedMIMEs = new Set([
  'image/png', 'image/jpeg', 'image/webp',
  'audio/mpeg', 'audio/mp3', 'audio/wav', 'audio/x-wav', 'audio/ogg', 'application/ogg',
  'video/x-msvideo', 'video/avi', 'video/quicktime', 'video/mp4', 'video/ogg',
  'video/x-ms-wmv', 'video/x-ms-asf', 'video/webm',
]);

const supportedExtension = /\.(png|jpe?g|webp|mp3|wav|ogg|avi|mov|mp4|wmv|webm)$/i;

export function isSupportedAttachmentFile(file) {
  return supportedMIMEs.has((file?.type || '').toLowerCase()) || supportedExtension.test(file?.name || '');
}

export function hasFileDrag(dataTransfer) {
  if (!dataTransfer) return false;
  if (Array.from(dataTransfer.files || []).length) return true;
  if (Array.from(dataTransfer.items || []).some((item) => item.kind === 'file')) return true;
  return Array.from(dataTransfer.types || []).includes('Files');
}

export function attachmentKind(attachment) {
  const mime = (attachment?.mime || attachment?.type || '').toLowerCase();
  if (mime.startsWith('image/')) return 'image';
  if (mime.startsWith('audio/')) return 'audio';
  if (mime.startsWith('video/')) return 'video';
  const name = attachment?.name || '';
  if (/\.(png|jpe?g|webp)$/i.test(name)) return 'image';
  if (/\.(mp3|wav)$/i.test(name) || (/\.ogg$/i.test(name) && !mime.startsWith('video/'))) return 'audio';
  return 'video';
}

export function canPreviewVideo(attachment) {
  return ['video/mp4', 'video/webm', 'video/ogg'].includes((attachment?.mime || '').toLowerCase());
}

export function formatAttachmentSize(bytes) {
  if (!bytes) return '';
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
