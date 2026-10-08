export const SESSION_PAGE_SIZE = 15;

export function sessionPage(items, requested = 0) {
  const pages = Math.max(1, Math.ceil(items.length / SESSION_PAGE_SIZE));
  const page = Math.max(0, Math.min(pages - 1, requested));
  const start = page * SESSION_PAGE_SIZE;
  return { page, pages, start, end: Math.min(items.length, start + SESSION_PAGE_SIZE), total: items.length, items: items.slice(start, start + SESSION_PAGE_SIZE) };
}
