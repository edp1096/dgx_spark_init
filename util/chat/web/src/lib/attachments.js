export const attachmentAccept = [
  'image/png', 'image/jpeg', 'image/webp',
  'audio/mpeg', 'audio/wav', 'audio/ogg',
  'video/x-msvideo', 'video/quicktime', 'video/mp4', 'video/ogg', 'video/x-ms-wmv', 'video/webm',
	'application/pdf', 'text/plain', 'text/markdown', 'text/html', 'text/csv', 'text/tab-separated-values',
	'application/json', 'application/xml', 'text/xml', 'text/yaml', 'application/yaml',
	'application/javascript', 'text/javascript', 'application/sql',
	'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
	'application/vnd.openxmlformats-officedocument.presentationml.presentation',
	'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
	'application/vnd.oasis.opendocument.text', 'application/vnd.oasis.opendocument.presentation', 'application/vnd.oasis.opendocument.spreadsheet',
	'application/epub+zip', 'application/vnd.hancom.hwpx',
  '.png', '.jpg', '.jpeg', '.webp', '.mp3', '.wav', '.ogg', '.avi', '.mov', '.mp4', '.wmv', '.webm',
	'.pdf', '.txt', '.md', '.markdown', '.html', '.htm', '.csv', '.tsv', '.json', '.xml', '.yaml', '.yml', '.toml',
	'.js', '.jsx', '.ts', '.tsx', '.css', '.scss', '.py', '.go', '.rs', '.java', '.c', '.h', '.cpp', '.hpp', '.cs', '.sh', '.ps1', '.sql', '.ini', '.conf', '.log',
	'.docx', '.pptx', '.xlsx', '.odt', '.odp', '.ods', '.epub', '.hwpx',
].join(',');

export const maxImageBytes = 15 * 1024 * 1024;
export const maxAttachmentBytes = 64 * 1024 * 1024;
export const maxMessageBytes = 96 * 1024 * 1024;

const supportedMIMEs = new Set([
  'image/png', 'image/jpeg', 'image/webp',
  'audio/mpeg', 'audio/mp3', 'audio/wav', 'audio/x-wav', 'audio/ogg', 'application/ogg',
  'video/x-msvideo', 'video/avi', 'video/quicktime', 'video/mp4', 'video/ogg',
  'video/x-ms-wmv', 'video/x-ms-asf', 'video/webm',
	'application/pdf', 'text/plain', 'text/markdown', 'text/html', 'text/csv', 'text/tab-separated-values',
	'application/json', 'application/xml', 'text/xml', 'text/yaml', 'application/yaml',
	'application/javascript', 'text/javascript', 'application/sql',
	'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
	'application/vnd.openxmlformats-officedocument.presentationml.presentation',
	'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
	'application/vnd.oasis.opendocument.text', 'application/vnd.oasis.opendocument.presentation', 'application/vnd.oasis.opendocument.spreadsheet',
	'application/epub+zip', 'application/vnd.hancom.hwpx',
]);

const supportedExtension = /\.(png|jpe?g|webp|mp3|wav|ogg|avi|mov|mp4|wmv|webm|pdf|txt|md|markdown|html?|csv|tsv|json|xml|ya?ml|toml|js|jsx|ts|tsx|css|scss|py|go|rs|java|c|h|cpp|hpp|cs|sh|ps1|sql|ini|conf|log|docx|pptx|xlsx|odt|odp|ods|epub|hwpx)$/i;

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
	if (mime === 'application/pdf' || mime.startsWith('text/') || mime.includes('document') || mime.includes('presentation') || mime.includes('spreadsheet') || mime.includes('opendocument') || mime.includes('epub') || mime.includes('hwpx') || ['application/json', 'application/xml', 'application/yaml'].includes(mime)) return 'document';
  const name = attachment?.name || '';
  if (/\.(png|jpe?g|webp)$/i.test(name)) return 'image';
  if (/\.(mp3|wav)$/i.test(name) || (/\.ogg$/i.test(name) && !mime.startsWith('video/'))) return 'audio';
	if (/\.(pdf|txt|md|markdown|html?|csv|tsv|json|xml|ya?ml|toml|js|jsx|ts|tsx|css|scss|py|go|rs|java|c|h|cpp|hpp|cs|sh|ps1|sql|ini|conf|log|docx|pptx|xlsx|odt|odp|ods|epub|hwpx)$/i.test(name)) return 'document';
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
