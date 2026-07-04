/** Shared DOM and rendering utilities.
 *
 *  These are extracted from app.js helpers section.
 *  During gradual migration, both app.js globals and ES module exports exist.
 */

// ── DOM shortcuts ──────────────────────────────────────────────────────

/** Shorthand for document.getElementById. */
export const $ = (id) => document.getElementById(id);

/** HTML-escape a string. */
export const esc = (s) => String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;')
  .replace(/>/g, '&gt;').replace(/"/g, '&quot;');

// ── Toast notification ─────────────────────────────────────────────────

/** Show a temporary toast notification. */
export function toast(msg, kind = 'info', ms = 4000) {
  const box = $('toastBox');
  if (!box) return;
  const el = document.createElement('div');
  el.className = 'toast ' + kind;
  el.innerHTML = `<span>${esc(msg)}</span><span class="toast-close">&times;</span>`;
  const closeBtn = el.querySelector('.toast-close');
  if (closeBtn) closeBtn.onclick = () => el.remove();
  box.appendChild(el);
  if (ms > 0) setTimeout(() => {
    el.style.opacity = '0';
    el.style.transition = 'opacity .3s';
    setTimeout(() => el.remove(), 300);
  }, ms);
}

// ── Status bar helpers ─────────────────────────────────────────────────

/** Update the status text and dot color. */
export function setStatus(text, kind) {
  const st = $('statusText');
  if (st) st.textContent = text || '就绪';
  const dot = $('statusDot');
  if (dot) {
    dot.className = 'status-dot';
    if (kind) dot.classList.add(kind);
  }
}

/** Update the task badge display. */
export function setTaskBadge(tid) {
  const badge = $('taskBadge');
  if (badge) badge.textContent = tid || '无任务';
  const tidEl = $('taskId');
  if (tidEl) tidEl.textContent = tid || '—';
  const tidInput = $('taskIdInput');
  if (tidInput) tidInput.value = tid || '';
}
