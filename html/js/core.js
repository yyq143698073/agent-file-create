/* ═══════════════════════════════════════════════════════════════════════
   Core: global state + shared DOM / rendering helpers
   Loaded FIRST as a plain <script> (not module) so every subsequent
   script can use S / $ / esc / toast / setStatus / renderMD / etc.
   ═══════════════════════════════════════════════════════════════════════ */

const S = {
  taskId: "",
  pollTimer: null,
  history: [],
  kb: "default",
  lastAbText: "",
  previewTab: "outline",
  kbTab: "kb-query",
  generating: false,
  _outlineDone: false,
  _sending: false,
  _evalData: null,
  _evalTab: "combined",
  _waitingFeedback: false,
  _feedbackStage: "",
  _satisfactionKey: "",
  _waitingRedo: false,
  _redoType: "",
  _redoBaseVersion: 0,
  _versions: { outline: [], content: [] },
  _selectedVersion: { outline: null, content: null },
  _compareData: null,
  _compareSourceTab: "content",
  _showDiff: true,
  _cachedDiff: null,
  _lastSatisfactionStage: "",
  _lastSatisfactionPreview: "",
  _lastSatisfactionVersion: 0,
  _editingSection: null,
  _contentStreaming: false,
  _tplEditor: {
    currentName: "",
    content: "",
    variables: [],
    dirty: false,
  },
  _historyDirty: false,
  _satisfactionSubmitting: false,
  _pendingTemplate: null,
};

/* ── UI state machine ─────────────────────────────────────────────── */
const UI_STATES = {
  IDLE: "idle",
  GENERATING: "generating",
  SENDING_CHAT: "sending_chat",
  WAITING_SATISFACTION: "waiting_satisfaction",
  WAITING_SATISFACTION_FEEDBACK: "waiting_satisfaction_feedback",
  WAITING_REGEN_FEEDBACK: "waiting_regen_feedback",
  WAITING_REDO: "waiting_redo",
  EDITING_SECTION: "editing_section",
};

function deriveUIState() {
  if (S._sending) return UI_STATES.SENDING_CHAT;
  if (S._editingSection) return UI_STATES.EDITING_SECTION;
  if (S._waitingRedo) return UI_STATES.WAITING_REDO;
  if (S._waitingFeedback) return UI_STATES.WAITING_SATISFACTION_FEEDBACK;
  if (S._waitingRegenFeedback) return UI_STATES.WAITING_REGEN_FEEDBACK;
  if (S._satisfactionSubmitting) return UI_STATES.WAITING_SATISFACTION;
  if (S.generating) return UI_STATES.GENERATING;
  return UI_STATES.IDLE;
}

function syncUIState() {
  const next = deriveUIState();
  if (S.uiState === next) return;
  S.uiState = next;
  try {
    if (typeof window !== "undefined" && window.store && typeof window.store.set === "function") {
      window.store.set({ uiState: next });
    }
  } catch (e) { /* store is optional */ }
}

/* ── DOM shortcuts ────────────────────────────────────────────────── */
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

/* ── Toast ────────────────────────────────────────────────────────── */
function toast(msg, kind = "info", ms = 4000) {
  const box = $("toastBox");
  if (!box) return null;
  const el = document.createElement("div");
  el.className = "toast " + kind;
  el.innerHTML = `<span>${esc(msg)}</span><span class="toast-close">&times;</span>`;
  el.querySelector(".toast-close").onclick = () => el.remove();
  box.appendChild(el);
  if (ms > 0) setTimeout(() => { el.style.opacity = "0"; el.style.transition = "opacity .3s"; setTimeout(() => el.remove(), 300); }, ms);
  return el;
}

/* ── Status ───────────────────────────────────────────────────────── */
function setStatus(text, kind) {
  $("statusText").textContent = text || LANG.status_ready;
  const dot = $("statusDot");
  dot.className = "status-dot";
  if (kind) dot.classList.add(kind);
}

function setTaskBadge(tid) {
  S.taskId = tid || "";
  $("taskBadge").textContent = tid || LANG.chat_no_task;
  $("taskId").textContent = tid || "—";
  $("taskIdInput").value = tid || "";
  try {
    if (window.store && typeof window.store.set === "function") {
      window.store.set({ taskId: S.taskId });
    }
  } catch (e) { /* store is optional */ }
}

/* ── Markdown rendering ────────────────────────────────────────────── */
const _SAFE_URI = /^(https?:|mailto:|tel:|ftp:|\/|#|data:image\/)/i;
let _markedReady = false;
try { _markedReady = typeof marked !== "undefined" && typeof DOMPurify !== "undefined"; } catch (e) { _markedReady = false; }

// Per-tab preview render cache. Keys: "outline" | "content".
const _previewCache = new Map();

function renderMD(text) {
  if (!text) return "";
  if (_markedReady) {
    let html;
    try {
      html = marked.parse(String(text), { mangle: false, headerIds: false, breaks: true });
    } catch (e) {
      return esc(text).replace(/\n/g, "<br>");
    }
    try {
      html = DOMPurify.sanitize(html, {
        ALLOWED_URI_REGEXP: _SAFE_URI,
        ADD_ATTR: ["target", "rel", "data-lang"],
      });
    } catch (e) { /* keep marked output */ }
    return html.replace(/<a\s+href=/gi, '<a target="_blank" rel="noopener" href=');
  }
  // Fallback
  let h = esc(text);
  h = h.replace(/^#### (.+)$/gm, "<h4>$1</h4>");
  h = h.replace(/^### (.+)$/gm, "<h3>$1</h3>");
  h = h.replace(/^## (.+)$/gm, "<h2>$1</h2>");
  h = h.replace(/^# (.+)$/gm, "<h1>$1</h1>");
  h = h.replace(/`([^`]+)`/g, "<code>$1</code>");
  h = h.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  h = h.replace(/\*(.+?)\*/g, "<em>$1</em>");
  h = h.replace(/\n/g, "<br>");
  return h;
}

/* ── Drag & Drop ──────────────────────────────────────────────────── */
function setupDropZone() {
  const zone = $("dropZone");
  const input = $("files");
  const list = $("fileList");
  const btn = $("btnPickFiles");

  btn.onclick = () => input.click();
  zone.onclick = (e) => { if (e.target === zone || e.target.classList.contains("drop-hint")) input.click(); };

  zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("drag-over"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("drag-over");
    const dt = e.dataTransfer;
    if (dt.files.length) {
      const newFiles = new DataTransfer();
      for (const f of input.files) newFiles.items.add(f);
      for (const f of dt.files) newFiles.items.add(f);
      input.files = newFiles.files;
      showFileTags(list, input.files, true);
    }
  });

  input.addEventListener("change", () => showFileTags(list, input.files, true));

  const tplInput = $("templates");
  const tplList = $("tplList");
  tplInput.addEventListener("change", () => showFileTags(tplList, tplInput.files, false));
}

function showFileTags(container, fileList, removable) {
  container.innerHTML = "";
  for (const f of fileList || []) {
    const tag = document.createElement("span");
    tag.className = "file-tag";
    tag.textContent = f.name;
    if (removable) {
      const rm = document.createElement("span");
      rm.className = "file-tag-rm";
      rm.textContent = "×";
      rm.onclick = (e) => { e.stopPropagation(); tag.remove(); };
      tag.appendChild(rm);
    }
    container.appendChild(tag);
  }
}

/* ── Tab switching ────────────────────────────────────────────────── */
function setupTabs(containerId) {
  const container = $(containerId);
  if (!container) return;
  container.addEventListener("click", (e) => {
    const tab = e.target.closest(".tab");
    if (!tab) return;
    const panelId = "tab-" + tab.dataset.tab;
    container.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    tab.classList.add("active");
    const parent = container.parentElement;
    parent.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
    const panel = $(panelId);
    if (panel) panel.classList.add("active");
  });
}
