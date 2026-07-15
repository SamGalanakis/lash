// Thin client for the workflow-graph-roundtrip backend contract.
// URLs are relative so the same code works behind the Vite dev proxy and when
// served single-origin from `frontend/dist/`.

export async function fetchWorkflow() {
  const res = await fetch('/workflow', { headers: { accept: 'application/json' } });
  if (!res.ok) throw new Error(`GET /workflow failed: ${res.status}`);
  return res.json();
}

// Returns { ok: true, document } on success, or { ok: false, status, error }
// carrying the typed render error so the UI can show it without losing the draft.
export async function saveWorkflow(document) {
  const res = await fetch('/workflow', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(document),
  });
  if (res.ok) {
    return { ok: true, document: await res.json() };
  }
  let body = null;
  try {
    body = await res.json();
  } catch {
    body = null;
  }
  return { ok: false, status: res.status, error: body?.error ?? null };
}

// Opens POST /run and yields parsed run-event payloads as they stream in.
// Each call is a brand-new run/invocation. `signal` aborts it (a new Play).
export async function* runWorkflow(signal) {
  const res = await fetch('/run', { method: 'POST', signal });
  if (!res.ok) {
    let detail = '';
    try {
      detail = JSON.stringify(await res.json());
    } catch {
      detail = await res.text().catch(() => '');
    }
    throw new Error(`POST /run failed: ${res.status} ${detail}`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    // SSE frames are separated by a blank line.
    let sep;
    while ((sep = buffer.indexOf('\n\n')) !== -1) {
      const frame = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      const event = parseFrame(frame);
      if (event) yield event;
    }
  }
}

function parseFrame(frame) {
  const lines = frame.split('\n');
  let dataLine = null;
  let eventName = null;
  for (const line of lines) {
    if (line.startsWith('data:')) dataLine = line.slice(5).trim();
    else if (line.startsWith('event:')) eventName = line.slice(6).trim();
  }
  if (eventName === 'keep-alive' || !dataLine) return null;
  try {
    return JSON.parse(dataLine);
  } catch {
    return null;
  }
}
