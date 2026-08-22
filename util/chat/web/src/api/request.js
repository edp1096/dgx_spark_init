export async function request(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const text = await response.text();
    let message = text;
    try { message = JSON.parse(text)?.error || text; } catch { /* plain text error */ }
    throw new Error(message || `HTTP ${response.status}`);
  }
  if (response.status === 204) return null;
  return response.json();
}
