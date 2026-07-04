/** Centralized API client module.
 *
 *  Extracted from app.js apiGet/apiPost/apiDelete helpers.
 *  During gradual migration, both global functions and module exports exist.
 */

/** GET request — returns parsed JSON. */
export async function apiGet(url) {
  const r = await fetch(url);
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.error || `${r.status}`);
  }
  return r.json();
}

/** POST request — auto-detects JSON vs FormData body. */
export async function apiPost(url, body) {
  const opts = { method: 'POST' };
  if (body instanceof FormData) {
    opts.body = body;
  } else {
    opts.headers = { 'Content-Type': 'application/json' };
    opts.body = JSON.stringify(body);
  }
  const r = await fetch(url, opts);
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.error || `${r.status}`);
  }
  return r.json();
}

/** DELETE request — returns parsed JSON. */
export async function apiDelete(url) {
  const r = await fetch(url, { method: 'DELETE' });
  const data = await r.json();
  if (!r.ok) throw new Error((data && data.detail) || r.statusText);
  return data;
}

// ── Convenience functions for common API calls ──────────────────────────

/** Upload files to a task. */
export function uploadFiles(files, taskId) {
  const fd = new FormData();
  for (const f of files) fd.append('files', f);
  fd.append('task_id', taskId || '');
  return apiPost('/api/upload', fd);
}

/** Append files to existing task. */
export function appendFiles(files, taskId, mode) {
  const fd = new FormData();
  for (const f of files) fd.append('files', f);
  fd.append('task_id', taskId || '');
  if (mode) fd.append('mode', mode);
  return apiPost('/api/append', fd);
}

/** Get task status. */
export function getTaskStatus(taskId) {
  return apiGet(`/api/status?task_id=${encodeURIComponent(taskId)}`);
}

/** Send a chat message. */
export function sendChat(taskId, message, history, action) {
  return apiPost('/api/chat', {
    task_id: taskId,
    message: message,
    history: history || [],
    action: action,
  });
}

/** Start document generation. */
export function startGeneration(taskId, prompt) {
  return apiPost('/api/gen', { task_id: taskId, prompt: prompt });
}

/** Stream chat response — returns the fetch Response for SSE reading. */
export function streamChat(taskId, message, history, action) {
  return fetch('/api/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task_id: taskId, message, history, action }),
  });
}

/** Submit satisfaction feedback. */
export function submitSatisfaction(taskId, satisfied, feedback) {
  return apiPost('/api/satisfaction', {
    task_id: taskId,
    satisfied: satisfied,
    feedback: feedback || '',
  });
}

/** Save chat history. */
export function saveChatHistory(taskId, history) {
  return apiPost('/api/chat/history/save', {
    task_id: taskId,
    history: history,
  });
}

/** Load chat history. */
export function loadChatHistory(taskId) {
  return fetch(`/api/chat/history?task_id=${encodeURIComponent(taskId)}`).then(r => r.json());
}

/** List all tasks. */
export function listTasks() {
  return apiGet('/api/tasks');
}

/** List versions for a task. */
export function listVersions(taskId, type) {
  return apiGet(`/api/versions?task_id=${encodeURIComponent(taskId)}&type=${type}`);
}

/** Select a specific version. */
export function selectVersion(taskId, type, versionNum) {
  return apiPost('/api/versions/select', {
    task_id: taskId, type: type, version: versionNum,
  });
}

/** Delete a version. */
export function deleteVersion(taskId, type, versionNum) {
  return apiPost('/api/versions/delete', {
    task_id: taskId, type: type, version: versionNum,
  });
}

/** Redo generation from a base version. */
export function redoVersion(taskId, type, baseVersion, feedback) {
  return apiPost('/api/versions/redo', {
    task_id: taskId, type: type, base_version: baseVersion, feedback: feedback,
  });
}

/** Clean old versions. */
export function cleanVersions(taskId, keepLast) {
  return apiPost('/api/versions/clean', {
    task_id: taskId, keep_last: keepLast || 20,
  });
}

// ── KB API ──────────────────────────────────────────────────────────────

/** Query knowledge base. */
export function queryKB(kb, question, topK, filters) {
  return apiPost('/api/kb/query', { kb, question, top_k: topK, filters });
}

/** List KB documents. */
export function listKBDocs(kb) {
  return apiGet(`/api/kb/docs?kb=${encodeURIComponent(kb)}`);
}

/** List all KB names. */
export function listKB() {
  return apiGet('/api/kb/list');
}

/** Delete a KB. */
export function deleteKB(kb) {
  return apiPost('/api/kb/delete', { kb });
}

/** Delete a KB document. */
export function deleteKBDoc(kb, docId) {
  return apiPost('/api/kb/delete', { kb, doc_id: docId });
}

/** Upload to KB. */
export function uploadKB(files, kb) {
  const fd = new FormData();
  for (const f of files) fd.append('files', f);
  if (kb) fd.append('kb', kb);
  return apiPost('/api/kb/upload', fd);
}

// ── Template API ────────────────────────────────────────────────────────

/** List custom templates. */
export function listTemplates() {
  return apiGet('/api/template/custom/list');
}

/** Get a template by name. */
export function getTemplate(name) {
  return apiGet(`/api/template/custom/${encodeURIComponent(name)}.md`);
}

/** Save a template. */
export function saveTemplate(name, content) {
  return apiPost('/api/template/custom/save', { name, content });
}

/** Delete a template. */
export function deleteTemplate(name) {
  return apiDelete(`/api/template/custom/${encodeURIComponent(name)}.md`);
}
