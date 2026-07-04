/** Bridge: expose ES module exports on `window.api` and `window.store`
 *  so that legacy app.js (global scope) can gradually adopt them.
 *
 *  Load AFTER api.js / state.js via <script type="module">.
 *  Once all callers use `window.api.*`, the legacy apiGet/apiPost in
 *  app.js can be removed.
 */

import { apiGet, apiPost, apiDelete } from './api.js';
import { store } from './state.js';

// ── API bridge ──────────────────────────────────────────────────────────

window.api = {
  get: apiGet,
  post: apiPost,
  delete: apiDelete,

  // Convenience wrappers
  uploadFiles(files, taskId) {
    const fd = new FormData();
    for (const f of files) fd.append('files', f);
    fd.append('task_id', taskId || '');
    return apiPost('/api/upload', fd);
  },

  getStatus(taskId) {
    return apiGet(`/api/status?task_id=${encodeURIComponent(taskId)}`);
  },

  sendChat(taskId, message, history, action) {
    return apiPost('/api/chat', {
      task_id: taskId, message, history: history || [], action,
    });
  },

  streamChat(taskId, message, history, action) {
    return fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ task_id: taskId, message, history, action }),
    });
  },

  submitSatisfaction(taskId, stage, satisfied, feedback, scope) {
    return apiPost('/api/satisfaction', {
      task_id: taskId, stage, satisfied, feedback, scope,
    });
  },

  listTasks() { return apiGet('/api/tasks'); },
  listVersions(taskId, type) {
    return apiGet(`/api/versions?task_id=${encodeURIComponent(taskId)}&type=${type}`);
  },

  queryKB(kb, question, topK) {
    return apiPost('/api/kb/query', { kb, question, top_k: topK });
  },
  listKB() { return apiGet('/api/kb/list'); },
  listKBDocs(kb) { return apiGet(`/api/kb/docs?kb=${encodeURIComponent(kb)}`); },
  deleteKB(kb) { return apiPost('/api/kb/delete', { kb }); },
  uploadKB(files, kb) {
    const fd = new FormData();
    for (const f of files) fd.append('files', f);
    if (kb) fd.append('kb', kb);
    return apiPost('/api/kb/upload', fd);
  },
};

// ── State bridge ─────────────────────────────────────────────────────────

window.store = store;

console.log('[bridge] api + store exposed on window (api.* / store.*)');
