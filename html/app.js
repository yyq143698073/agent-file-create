/* ═══════════════════════════════════════════════════════════════════════
   State
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
  _waitingFeedback: false,         // user clicked "不满意", next chat msg is feedback
  _feedbackStage: "",              // "outline" | "content" — stage for feedback
  _satisfactionKey: "",            // "outline_v2" — prevents duplicate satisfaction msgs
  _waitingRedo: false,             // user clicked "基于此版重做", next chat msg is redo feedback
  _redoType: "",                   // "outline" | "content"
  _redoBaseVersion: 0,             // base version number for redo
  _versions: { outline: [], content: [] },
  _selectedVersion: { outline: null, content: null },
  _compareData: null,            // { oldText, newText, type }
  _compareSourceTab: "content",  // which tab was active when entering compare
  _showDiff: true,               // toggle diff highlighting in compare view
  _cachedDiff: null,             // cached diff result {oldText, newText, oldDiff, newDiff}
  _lastSatisfactionStage: "",    // saved for "返回" from structured feedback
  _lastSatisfactionPreview: "",
  _lastSatisfactionVersion: 0,
  _editingSection: null,         // { headingEl, toHide, editDiv, headingText } or null
  _contentStreaming: false,      // true while content sections are being generated
  _tplEditor: {                  // template editor state
    currentName: "",             // currently loaded template name (without .md)
    content: "",                 // current editor content
    variables: [],               // cached variable list
    dirty: false,                // unsaved changes flag
  },
  _historyDirty: false,          // pending chat history sync to backend
  _satisfactionSubmitting: false, // guard against double-submit
};

/* ═══════════════════════════════════════════════════════════════════════
   Helpers
   ═══════════════════════════════════════════════════════════════════════ */

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

/* ── Toast ──────────────────────────────────────────────────────────── */

function toast(msg, kind = "info", ms = 4000) {
  const box = $("toastBox");
  const el = document.createElement("div");
  el.className = "toast " + kind;
  el.innerHTML = `<span>${esc(msg)}</span><span class="toast-close">&times;</span>`;
  el.querySelector(".toast-close").onclick = () => el.remove();
  box.appendChild(el);
  if (ms > 0) setTimeout(() => { el.style.opacity = "0"; el.style.transition = "opacity .3s"; setTimeout(() => el.remove(), 300); }, ms);
}

/* ── Status ─────────────────────────────────────────────────────────── */

function setStatus(text, kind) {
  $("statusText").textContent = text || "就绪";
  const dot = $("statusDot");
  dot.className = "status-dot";
  if (kind) dot.classList.add(kind);
}

function setTaskBadge(tid) {
  S.taskId = tid || "";
  $("taskBadge").textContent = tid || "无任务";
  $("taskId").textContent = tid || "—";
  $("taskIdInput").value = tid || "";
}

/* ── Markdown rendering ─────────────────────────────────────────────── */

function renderMD(text) {
  if (!text) return "";
  let h = esc(text);

  // ── Phase 1: Code blocks (fenced) ──
  const codeBlocks = [];
  h = h.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
    const idx = codeBlocks.length;
    codeBlocks.push({ lang: esc(lang || ""), code: code.replace(/\n$/, "") });
    return `\x00CODEBLOCK_${idx}\x00`;
  });

  // ── Phase 2: Escape inline HTML in non-code content ──
  // (already escaped via esc(), but we need to protect the placeholders)

  // ── Phase 3: Block-level elements ──

  // Headings (must come before paragraphs)
  h = h.replace(/^#### (.+)$/gm, "<h4>$1</h4>");
  h = h.replace(/^### (.+)$/gm, "<h3>$1</h3>");
  h = h.replace(/^## (.+)$/gm, "<h2>$1</h2>");
  h = h.replace(/^# (.+)$/gm, "<h1>$1</h1>");

  // Horizontal rules
  h = h.replace(/^(---|\*\*\*|___)\s*$/gm, "<hr>");

  // Tables
  h = h.replace(/((?:^\|.+?\|[ \t]*\n)+(?:^\|[-: |]+\|[ \t]*\n)(?:^\|.+?\|[ \t]*\n)+)/gm, (match) => {
    const lines = match.trim().split("\n");
    if (lines.length < 2) return match;
    // Skip separator line (|---|---|)
    const headerLine = lines[0];
    const bodyLines = lines.slice(2);
    const buildRow = (line, tag) => {
      const cells = line.replace(/^\||\|$/g, "").split("|").map(c => `<${tag}>${c.trim()}</${tag}>`);
      return `<tr>${cells.join("")}</tr>`;
    };
    let tbl = "<table><thead>" + buildRow(headerLine, "th") + "</thead><tbody>";
    for (const bl of bodyLines) {
      if (bl.trim()) tbl += buildRow(bl, "td");
    }
    tbl += "</tbody></table>";
    return tbl;
  });

  // Blockquotes
  h = h.replace(/^(> .+(?:\n> .+)*)/gm, (m) => {
    const content = m.replace(/^> /gm, "");
    return "<blockquote>" + content + "</blockquote>";
  });

  // Unordered lists (multi-line)
  h = h.replace(/((?:^[-*] .+(?:\n|$))+)/gm, (m) => {
    const items = m.trim().split("\n").filter(l => /^[-*] /.test(l)).map(l => "<li>" + l.replace(/^[-*] /, "") + "</li>");
    return "<ul>" + items.join("") + "</ul>";
  });

  // Ordered lists (multi-line)
  h = h.replace(/((?:^\d+\. .+(?:\n|$))+)/gm, (m) => {
    const items = m.trim().split("\n").filter(l => /^\d+\. /.test(l)).map(l => "<li>" + l.replace(/^\d+\. /, "") + "</li>");
    return "<ol>" + items.join("") + "</ol>";
  });

  // Paragraphs: consecutive non-empty, non-tag lines
  h = h.replace(/(\n\n+)/g, "\n\n");
  const blocks = h.split(/\n\n+/);
  h = blocks.map(b => {
    const trimmed = b.trim();
    if (!trimmed) return "";
    // If already wrapped in a block tag, leave as is
    if (/^<(h[1-4]|table|ul|ol|blockquote|hr|pre|div)/.test(trimmed)) return trimmed;
    // Wrap in paragraph
    return "<p>" + trimmed.replace(/\n/g, "<br>") + "</p>";
  }).join("\n");

  // ── Phase 4: Inline elements ──
  // Bold and italic
  h = h.replace(/\*\*\*(.+?)\*\*\*/g, "<strong><em>$1</em></strong>");
  h = h.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  h = h.replace(/\*(.+?)\*/g, "<em>$1</em>");

  // Inline code (single backtick)
  h = h.replace(/`([^`]+)`/g, "<code>$1</code>");

  // Links [text](url)
  h = h.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

  // Images ![alt](url)
  h = h.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, '<img src="$2" alt="$1" style="max-width:100%">');

  // ── Phase 5: Restore code blocks ──
  h = h.replace(/\x00CODEBLOCK_(\d+)\x00/g, (_, idx) => {
    const cb = codeBlocks[parseInt(idx, 10)];
    if (!cb) return "";
    const langAttr = cb.lang ? ` data-lang="${cb.lang}"` : "";
    return `<pre${langAttr}><code>${cb.code}</code></pre>`;
  });

  return h;
}

/* ── Drag & Drop ────────────────────────────────────────────────────── */

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

/* ── Tab switching ──────────────────────────────────────────────────── */

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

/* ═══════════════════════════════════════════════════════════════════════
   API helpers
   ═══════════════════════════════════════════════════════════════════════ */

async function apiGet(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || `${r.status}`);
  return r.json();
}

async function apiPost(url, body) {
  const opts = { method: "POST" };
  if (body instanceof FormData) {
    opts.body = body;
  } else {
    opts.headers = { "Content-Type": "application/json" };
    opts.body = JSON.stringify(body);
  }
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || `${r.status}`);
  return r.json();
}

/* ═══════════════════════════════════════════════════════════════════════
   Task upload & append
   ═══════════════════════════════════════════════════════════════════════ */

async function uploadAndGenerate() {
  if (S.generating) { toast("任务正在进行中，请等待完成", "info", 2000); return; }
  const files = $("files").files;
  if (!files.length) { toast("请选择至少一个文件", "err"); return; }

  const fd = new FormData();
  for (const f of files) fd.append("files", f, f.name);
  for (const t of ($("templates").files || [])) fd.append("templates", t, t.name);
  const up = $("userPrompt").value.trim();
  if (up) fd.append("user_prompt", up);
  fd.append("target_words", $("targetWords").value || "0");
  if ($("abEval").checked) fd.append("ab_eval", "1");
  if ($("addToKb").checked) {
    fd.append("add_to_kb", "true");
    const kbn = ($("kbNameInUpload").value || "").trim();
    if (kbn) fd.append("kb_name", kbn);
  }
  const kbDocIds = getKbDocIdsStr();
  if (kbDocIds) fd.append("kb_doc_ids", kbDocIds);

  S.generating = true;
  S._lastOperation = "generate";
  S._contentStreaming = false;
  S._satisfactionKey = "";
  S._outlineDone = false;
  $("btnUpload").disabled = true;
  $("btnAppend").disabled = true;
  setStatus("上传中…", "busy");
  S.lastAbText = "";
  $("abBox").textContent = "";
  $("abCard").style.display = "none";
  $("evalCard").style.display = "none";
  S._evalData = null;
  S._outlineDone = false;

  try {
    const data = await apiPost("/api/upload", fd);
    setTaskBadge(data.task_id || "");
    S.history = [];
    clearChat();
    $("previewBox").textContent = "";
    setDownloads(data.downloads || {});
    setStatus("生成中…", "busy");
    startPolling();
    await pollStatus();
  } catch (e) {
    setStatus(e.message, "err");
    toast("上传失败: " + e.message, "err");
    S.generating = false;
    $("btnUpload").disabled = false;
    $("btnAppend").disabled = false;
  }
}

async function appendToTask() {
  if (S.generating) { toast("任务正在进行中，请等待完成", "info", 2000); return; }
  const tid = S.taskId;
  if (!tid) { toast("请先生成或加载一个任务", "err"); return; }
  const files = $("files").files;
  const templates = $("templates").files;
  if (!files.length && !templates.length) { toast("请选择文件或模板", "err"); return; }

  const fd = new FormData();
  fd.append("task_id", tid);
  for (const f of files) fd.append("files", f, f.name);
  for (const t of templates) fd.append("templates", t, t.name);
  const up = $("userPrompt").value.trim();
  if (up) fd.append("user_prompt", up);
  fd.append("target_words", $("targetWords").value || "0");
  if ($("abEval").checked) fd.append("ab_eval", "1");
  if ($("addToKb").checked) {
    fd.append("add_to_kb", "true");
    const kbn = ($("kbNameInUpload").value || "").trim();
    if (kbn) fd.append("kb_name", kbn);
  }
  const kbDocIds = getKbDocIdsStr();
  if (kbDocIds) fd.append("kb_doc_ids", kbDocIds);

  S.generating = true;
  S._lastOperation = "generate";
  S._contentStreaming = false;
  $("btnUpload").disabled = true;
  $("btnAppend").disabled = true;
  setStatus("追加中…", "busy");
  S._outlineDone = false;
  $("evalCard").style.display = "none";
  S._evalData = null;
  try {
    const data = await apiPost("/api/append", fd);
    setTaskBadge(data.task_id || tid);
    setDownloads(data.downloads || {});
    setStatus("重新生成中…", "busy");
    startPolling();
    await pollStatus();
  } catch (e) {
    setStatus(e.message, "err");
    toast("追加失败: " + e.message, "err");
    S.generating = false;
    $("btnUpload").disabled = false;
    $("btnAppend").disabled = false;
  }
}

/* ═══════════════════════════════════════════════════════════════════════
   Polling
   ═══════════════════════════════════════════════════════════════════════ */

function startPolling() {
  if (S.pollTimer) clearTimeout(S.pollTimer);
  function next() {
    S.pollTimer = setTimeout(async () => {
      await pollStatus();
      next(); // Always schedule next, like setInterval
    }, 2000);
  }
  next();
}

function stopPolling() {
  if (S.pollTimer) { clearTimeout(S.pollTimer); S.pollTimer = null; }
}

async function pollStatus() {
  if (!S.taskId) return;
  try {
    const data = await apiGet(`/api/status?task_id=${encodeURIComponent(S.taskId)}`);
    const st = data.status || "unknown";
    const stage = data.stage || "";
    const msg = data.message || "";
    const total = data.total_files || 0;
    const done = data.done_files || 0;

    // Progress bar
    if (total > 0 && st === "processing") {
      $("progressWrap").style.display = "block";
      $("progressFill").style.width = Math.round((done / total) * 100) + "%";
      $("progressMsg").textContent = `${stage} ${done}/${total}` + (msg ? " — " + msg : "");
    } else if (st === "processing" && stage === "document") {
      $("progressWrap").style.display = "block";
      $("progressFill").style.width = "95%";
      $("progressMsg").textContent = msg || "正在生成文档…";
    } else if (st === "queued") {
      $("progressWrap").style.display = "block";
      $("progressFill").style.width = "5%";
      $("progressMsg").textContent = "排队中…";
    } else {
      $("progressWrap").style.display = "none";
    }

    // Toast when outline is done
    if (st === "processing" && stage === "document" && msg && msg.includes("大纲生成完成")) {
      if (!S._outlineDone) { S._outlineDone = true; toast("大纲已生成，正在撰写正文…", "info", 3000); }
    }

    if (st === "need_user") {
      if (stage === "satisfaction_outline" || stage === "satisfaction_content") {
        setStatus(msg || "请审阅并反馈", "busy");
        const previewText = data.preview_text || "";
        const previewVersion = data.preview_version || 1;
        showSatisfactionInChat(stage === "satisfaction_outline" ? "outline" : "content", previewText, previewVersion);
      } else {
        setStatus(msg || "需要补充信息", "busy");
        showClarifyInChat(data.clarify_questions || []);
      }
    } else if (st === "processing") {
      setStatus(`生成中 (${stage || "处理中"})`, "busy");
      // Only hide satisfaction if we're genuinely past it (keyguard protects active prompts)
      if (!S._satisfactionKey) {
        hideSatisfaction();
      }
      _clarifyMsgShown = false;
      // Load partial content preview during document generation
      if (stage === "document") {
        const content = await fetchText(`/result/${encodeURIComponent(S.taskId)}/content.md`);
        if (content && content.trim()) {
          S._content = content;
          // Auto-switch to content tab once when streaming sections begin
          if (!S._contentStreaming && content.includes("（生成中…）")) {
            S._contentStreaming = true;
            if (S.previewTab !== "content") {
              S.previewTab = "content";
              const container = $("previewTabs");
              container.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
              const activeTab = container.querySelector('[data-tab="content"]');
              if (activeTab) activeTab.classList.add("active");
              $("compareBox").style.display = "none";
              $("compareToolbar").style.display = "none";
              $("previewBox").style.display = "";
            }
          }
          showPreview();
        }
      }
    } else if (st === "queued") {
      setStatus("排队等待…", "busy");
      hideSatisfaction();
      S._satisfactionKey = "";
      _clarifyMsgShown = false;
    } else if (st === "finished") {
      setStatus("完成", "ok");
      hideSatisfaction();
      S._satisfactionKey = "";
      _clarifyMsgShown = false;
      S._contentStreaming = false;
      stopPolling();
      // Load preview
      const outline = await fetchText(`/result/${encodeURIComponent(S.taskId)}/outline.md`);
      const content = await fetchText(`/result/${encodeURIComponent(S.taskId)}/content.md`);
      S._outline = outline;
      S._content = content;
      showPreview();
      // Load versions
      await loadVersions("outline");
      await loadVersions("content");
      // Show version bar for current tab and compare tab if any type has ≥2 versions
      const ov = S._versions.outline || [];
      const cv = S._versions.content || [];
      if (ov.length > 1 || cv.length > 1) {
        $("tabCompare").style.display = "";
      }
      // Show version tags for the currently active preview tab
      showVersionTags(S.previewTab === "compare" ? (ov.length > 1 ? "outline" : "content") : S.previewTab);
      // Eval metrics bar (from generate_document pipeline)
      if (data.eval_metrics && typeof data.eval_metrics === "object") {
        renderEvalMetricsBar(data.eval_metrics);
      }
      // Eval results
      if (data.eval && typeof data.eval === "object") {
        S._evalData = data.eval;
        renderEval();
      }
      // Chat notification on completion
      if (S._lastOperation) {
        const opMessages = {
          generate: "报告已生成完成，请查看下方预览。",
          regen_section: "章节已重新生成，请查看下方预览。",
          edit_section: "章节已根据编辑内容重写完成，请查看下方预览。",
          regenerate: "报告已重新生成，请查看下方预览。",
          redo_version: "报告已基于选定版本重新生成，请查看下方预览。",
        };
        const notifMsg = opMessages[S._lastOperation];
        if (notifMsg) { addMsg("assistant", notifMsg); S.history.push({ role: "assistant", content: notifMsg }); }
        S._lastOperation = "";
      }
      S.generating = false;
      $("btnUpload").disabled = false;
      $("btnAppend").disabled = false;
      loadTaskList();
    } else if (st === "failed") {
      setStatus("失败", "err");
      hideSatisfaction();
      S._satisfactionKey = "";
      S._contentStreaming = false;
      stopPolling();
      toast("生成失败: " + (data.error || msg || "未知"), "err");
      S.generating = false;
      $("btnUpload").disabled = false;
      $("btnAppend").disabled = false;
      loadTaskList();
    } else if (st === "canceled" || st === "cancelled") {
      setStatus("已取消", "ok");
      hideSatisfaction();
      S._satisfactionKey = "";
      S._contentStreaming = false;
      stopPolling();
      S.generating = false;
      $("btnUpload").disabled = false;
      $("btnAppend").disabled = false;
      loadTaskList();
    } else {
      setStatus(msg || st, "ok");
    }

    setDownloads(data.downloads || {});

    // A/B results
    const ab = data.ab_results || [];
    if (ab.length) {
      const lines = [];
      for (const it of ab) {
        if (it.error) { lines.push(`${it.file}: ERROR ${it.error}`); continue; }
        const a = it.a || {}, b = it.b || {};
        lines.push(`${it.file} | A: ${a.filled_fields ?? "-"}/7 r=${(a.field_ratio ?? -1).toFixed(2)} | B: ${b.filled_fields ?? "-"}/7 r=${(b.field_ratio ?? -1).toFixed(2)} | choose=${it.chosen || "-"}`);
      }
      S.lastAbText = lines.join("\n");
      $("abBox").textContent = S.lastAbText;
      $("abCard").style.display = "";
    } else if (data.ab_eval) {
      $("abBox").textContent = "评估中…";
      $("abCard").style.display = "";
    }
  } catch (e) {
    // Silently retry on next poll
  }
}

async function fetchText(url) {
  try { const r = await fetch(url); return r.ok ? await r.text() : ""; } catch { return ""; }
}

function showPreview() {
  // Cancel any active section edit
  if (S._editingSection) cancelSectionEdit();
  const tab = S.previewTab;
  const raw = tab === "outline" ? (S._outline || "（大纲尚未生成）") : (S._content || "（正文尚未生成）");
  $("previewBox").innerHTML = renderMD(raw);
  attachSectionActions(tab);
  // Show/hide full-screen button when content exists
  const btn = $("btnFullPreview");
  if (btn) btn.style.display = (S._content && S._content.trim()) ? "" : "none";
}

/* ── Section-level interactive editing ────────────────────────────────── */

function extractSectionBody(md, headingText) {
  const lines = (md || "").split("\n");
  const blocks = [];
  let cur = null;
  for (let i = 0; i < lines.length; i++) {
    const m = lines[i].match(/^(#{1,6})\s+(.+?)\s*$/);
    if (m) {
      if (cur) { cur.endIdx = i; blocks.push(cur); }
      cur = { heading: m[2].trim(), level: m[1].length, startIdx: i, endIdx: lines.length };
    }
  }
  if (cur) blocks.push(cur);

  let idx = blocks.findIndex(b => b.heading === headingText);
  if (idx < 0) {
    idx = blocks.findIndex(b => b.heading.includes(headingText) || headingText.includes(b.heading));
  }
  if (idx < 0) return "";

  const target = blocks[idx];
  let end = target.endIdx;
  for (let j = idx + 1; j < blocks.length; j++) {
    if (blocks[j].level <= target.level) { end = blocks[j].startIdx; break; }
  }
  return lines.slice(target.startIdx + 1, end).join("\n").trim();
}

function attachSectionActions(tabType) {
  const box = $("previewBox");
  const headings = box.querySelectorAll("h2, h3");
  headings.forEach(h => {
    // Remove existing action buttons (in case of re-render)
    const existing = h.querySelector(".sect-actions");
    if (existing) existing.remove();

    const headingText = h.textContent.trim();
    const wrap = document.createElement("span");
    wrap.className = "sect-actions";
    wrap.style.cssText = "margin-left:8px;display:inline-flex;gap:4px;vertical-align:middle";

    // Regenerate button (both tabs)
    const regenBtn = document.createElement("button");
    regenBtn.className = "btn btn-sm sect-regen-btn";
    regenBtn.textContent = "重生成";
    regenBtn.title = "重新生成此章节";
    regenBtn.onclick = (e) => { e.stopPropagation(); regenSection(headingText); };
    wrap.appendChild(regenBtn);

    // Edit button (content tab only)
    if (tabType === "content") {
      const editBtn = document.createElement("button");
      editBtn.className = "btn btn-sm sect-edit-btn";
      editBtn.textContent = "编辑";
      editBtn.title = "编辑此章节内容";
      editBtn.onclick = (e) => { e.stopPropagation(); startSectionEdit(h, headingText); };
      wrap.appendChild(editBtn);
    }

    h.appendChild(wrap);
  });
}

const REGEN_OPTIONS = ["更详细", "更简短", "换种结构", "增加数据", "优化表达", "换个角度"];

function regenSection(headingText) {
  if (!S.taskId) { toast("请先生成或加载一个任务", "err"); return; }
  if (S.generating) { toast("任务正在进行中，请等待完成", "info", 2000); return; }

  S.generating = true;
  $("btnUpload").disabled = true;
  $("btnAppend").disabled = true;

  // Cancel any active edit
  if (S._editingSection) cancelSectionEdit();

  // Show feedback prompt with quick-select chips
  const safeHeading = headingText.replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const chipsHTML = REGEN_OPTIONS.map(opt =>
    `<span class="feedback-chip regen-chip" data-opt="${esc(opt)}">${esc(opt)}</span>`
  ).join("");

  const html = `<div id="regenFeedbackBubble">
    <p>您希望在「<b>${safeHeading}</b>」章节进行哪些方面的更改？</p>
    <p style="font-size:12px;color:var(--c-text2);margin:4px 0">可点选快捷选项（可多选），也可直接输入修改意见：</p>
    <div class="feedback-chips">${chipsHTML}</div>
    <p style="font-size:11px;color:var(--c-text3);margin-top:6px">输入意见后按回车发送，或输入"跳过"直接重生成。</p>
  </div>`;

  const msg = addActionMsg(html);

  // Attach chip toggle handlers
  msg.querySelectorAll(".regen-chip").forEach(chip => {
    chip.addEventListener("click", function () { this.classList.toggle("active"); });
  });

  S._waitingRegenFeedback = { heading: headingText, msg: msg };
}

function startSectionEdit(headingEl, headingText) {
  // Cancel any previous edit
  if (S._editingSection) cancelSectionEdit();

  const raw = S._content || "";
  const body = extractSectionBody(raw, headingText);
  if (!body && !raw) {
    toast("无法提取章节内容", "err");
    return;
  }

  const hLevel = parseInt(headingEl.tagName.charAt(1));
  const toHide = [];
  let el = headingEl.nextElementSibling;
  while (el) {
    const tagMatch = el.tagName && el.tagName.match(/^H([1-6])$/);
    if (tagMatch) {
      if (parseInt(tagMatch[1]) <= hLevel) break;
    }
    // Don't hide action buttons or edit wrappers
    if (!el.classList.contains("sect-actions") && !el.classList.contains("section-edit-wrap")) {
      toHide.push(el);
    }
    el = el.nextElementSibling;
  }
  toHide.forEach(e => { e._prevDisplay = e.style.display; e.style.display = "none"; });

  const editDiv = document.createElement("div");
  editDiv.className = "section-edit-wrap";
  editDiv.innerHTML = `
    <textarea class="section-edit-textarea" rows="12">${esc(body)}</textarea>
    <div class="section-edit-actions">
      <button class="btn btn-primary btn-sm sect-save-btn">保存并重写</button>
      <button class="btn btn-outline btn-sm sect-save-direct-btn">直接保存</button>
      <button class="btn btn-outline btn-sm sect-cancel-btn">取消</button>
      <span class="sect-edit-hint">「保存并重写」由 AI 润色后写入；「直接保存」原样写入不经过 AI</span>
    </div>
  `;
  headingEl.insertAdjacentElement("afterend", editDiv);

  S._editingSection = { headingEl, toHide, editDiv, headingText };
  editDiv.querySelector(".sect-cancel-btn").onclick = () => cancelSectionEdit();
  editDiv.querySelector(".sect-save-btn").onclick = () => submitSectionEdit(headingText, editDiv.querySelector("textarea"));
  editDiv.querySelector(".sect-save-direct-btn").onclick = () => saveSectionDirect(headingText, editDiv.querySelector("textarea"));

  // Scroll to edit area
  editDiv.scrollIntoView({ behavior: "smooth", block: "center" });
}

function cancelSectionEdit() {
  if (!S._editingSection) return;
  const s = S._editingSection;
  s.toHide.forEach(e => { e.style.display = e._prevDisplay || ""; });
  if (s.editDiv && s.editDiv.parentNode) s.editDiv.remove();
  S._editingSection = null;
}

async function submitSectionEdit(headingText, textarea) {
  const edited = textarea.value.trim();
  if (!edited) { toast("编辑内容不能为空", "err"); return; }
  if (!S.taskId) { toast("请先生成或加载一个任务", "err"); return; }

  S.generating = true;
  S._lastOperation = "edit_section";
  $("btnUpload").disabled = true;
  $("btnAppend").disabled = true;
  setStatus(`重写章节「${headingText}」…`, "busy");

  // Clean up edit UI
  cancelSectionEdit();

  try {
    const data = await apiPost("/api/section/edit", {
      task_id: S.taskId,
      section_name: headingText,
      edited_content: edited,
    });
    toast(data.message || "已启动编辑重写", "ok", 3000);
    $("previewBox").textContent = "";
    startPolling();
  } catch (e) {
    setStatus(e.message, "err");
    toast("编辑重写失败: " + e.message, "err");
    S.generating = false;
    $("btnUpload").disabled = false;
    $("btnAppend").disabled = false;
  }
}

async function saveSectionDirect(headingText, textarea) {
  const edited = textarea.value.trim();
  if (!edited) { toast("编辑内容不能为空", "err"); return; }
  if (!S.taskId) { toast("请先生成或加载一个任务", "err"); return; }

  setStatus("直接保存中…", "busy");

  // Clean up edit UI immediately
  cancelSectionEdit();

  try {
    const data = await apiPost("/api/section/save", {
      task_id: S.taskId,
      section_name: headingText,
      edited_content: edited,
    });
    if (data && data.ok) {
      toast(data.message || "已保存", "ok", 3000);
      // Reload content from disk
      const content = await fetchText(`/result/${encodeURIComponent(S.taskId)}/content.md`);
      if (content && content.trim()) {
        S._content = content;
        showPreview();
      }
      const notifMsg = "章节已直接保存，请查看下方预览。";
      addMsg("assistant", notifMsg);
      S.history.push({ role: "assistant", content: notifMsg });
    } else {
      toast(data.message || "保存失败", "err");
    }
    setStatus("已完成", "ok");
  } catch (e) {
    setStatus(e.message, "err");
    toast("直接保存失败: " + e.message, "err");
  }
}

function openFullPreview() {
  if (!S._content || !S._content.trim()) return;
  const win = window.open("", "_blank");
  win.document.write(`<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>报告正文 — ${esc(S.taskId)}</title>
<style>
  body { max-width:900px; margin:40px auto; padding:20px; font-family:system-ui,-apple-system,sans-serif; line-height:1.8; color:#1a1a1a; }
  h1 { font-size:1.8em; border-bottom:2px solid #e0e0e0; padding-bottom:12px; }
  h2 { font-size:1.4em; margin-top:32px; border-bottom:1px solid #eee; padding-bottom:8px; }
  h3 { font-size:1.15em; margin-top:24px; }
  h4 { font-size:1.05em; margin-top:20px; }
  table { border-collapse:collapse; width:100%; margin:16px 0; }
  th,td { border:1px solid #ddd; padding:10px 14px; text-align:left; }
  th { background:#f5f5f5; font-weight:600; }
  pre { background:#f8f8f8; border:1px solid #e0e0e0; border-radius:6px; padding:14px 18px; overflow-x:auto; font-size:0.9em; line-height:1.5; }
  code { background:#f0f0f0; padding:2px 6px; border-radius:3px; font-size:0.9em; }
  pre code { background:none; padding:0; }
  blockquote { border-left:4px solid #ccc; padding-left:16px; color:#666; margin:16px 0; }
  hr { border:none; border-top:1px solid #e0e0e0; margin:24px 0; }
  ul,ol { padding-left:24px; }
  a { color:#2563eb; }
  img { max-width:100%; }
</style></head><body>${renderMD(S._content)}</body></html>`);
  win.document.close();
}

/* ── Eval rendering ─────────────────────────────────────────────────── */

function colorClass(v) {
  if (v >= 0.7) return "high";
  if (v >= 0.4) return "mid";
  return "low";
}

function avgBadgeColor(avg) {
  if (avg >= 0.7) return "var(--c-green)";
  if (avg >= 0.4) return "var(--c-amber)";
  return "var(--c-red)";
}

function renderScoreBar(label, score) {
  const pct = Math.round(score * 100);
  return `<div class="eval-dim">
    <span class="eval-label">${esc(label)}</span>
    <div class="eval-bar-wrap"><div class="eval-bar-fill ${colorClass(score)}" style="width:${pct}%"></div></div>
    <span class="eval-score">${pct}</span>
  </div>`;
}

function renderEvalPanel(scores) {
  if (!scores) return '<div class="eval-empty">暂无数据</div>';
  const dims = [
    ["相关性", scores.relevance || 0],
    ["忠实度", scores.faithfulness || 0],
    ["连贯性", scores.coherence || 0],
    ["完整性", scores.completeness || 0],
  ];
  const avg = (dims.reduce((s, d) => s + d[1], 0) / 4);
  let html = dims.map(d => renderScoreBar(d[0], d[1])).join("");
  html += `<div class="eval-avg">
    <span>综合均分</span>
    <span class="eval-avg-badge" style="background:${avgBadgeColor(avg)}">${(avg * 100).toFixed(0)}</span>
  </div>`;
  return html;
}

function renderEvalMetricsBar(metrics) {
  if (!metrics) return;
  const bar = $("evalMetricsBar");
  bar.style.display = "flex";

  // Warn count
  const warns = metrics.warns !== undefined ? metrics.warns :
    (metrics.facts_total || 0) - (metrics.facts_verified || 0);
  const warnBadge = $("evalWarns");
  if (warns > 0) {
    warnBadge.style.background = "#fef2f2"; warnBadge.style.color = "#dc2626";
    warnBadge.textContent = `⚠️ ${warns} 条警告`;
  } else {
    warnBadge.style.background = "#f0fdf4"; warnBadge.style.color = "#16a34a";
    warnBadge.textContent = "✅ 无警告";
  }

  // FActScore
  const fs = metrics.factscore;
  const fsBadge = $("evalFActScore");
  if (fs != null && fs !== undefined) {
    const pct = Math.round(fs * 100);
    fsBadge.style.background = pct >= 70 ? "#f0fdf4" : pct >= 40 ? "#fffbeb" : "#fef2f2";
    fsBadge.style.color = pct >= 70 ? "#16a34a" : pct >= 40 ? "#b45309" : "#dc2626";
    fsBadge.textContent = `FAct: ${pct}%`;
  } else {
    fsBadge.textContent = "";
  }

  // Coverage
  const cov = metrics.coverage;
  const covBadge = $("evalCoverage");
  if (cov != null && cov !== undefined) {
    const pct = Math.round(cov * 100);
    covBadge.style.background = pct >= 70 ? "#f0fdf4" : pct >= 40 ? "#fffbeb" : "#fef2f2";
    covBadge.style.color = pct >= 70 ? "#16a34a" : pct >= 40 ? "#b45309" : "#dc2626";
    covBadge.textContent = `覆盖: ${pct}%`;
  } else {
    covBadge.textContent = "";
  }

  // Citation estimate (from citation verification results if available)
  const citeBadge = $("evalCiteAcc");
  if (metrics.facts_verified != null && metrics.facts_total != null && metrics.facts_total > 0) {
    citeBadge.style.background = "#f0fdf4"; citeBadge.style.color = "#16a34a";
    citeBadge.textContent = `${metrics.facts_verified}/${metrics.facts_total} 事实已验证`;
  } else {
    citeBadge.textContent = "";
  }
}

function renderEval() {
  const data = S._evalData;
  if (!data) {
    $("evalCard").style.display = "none";
    return;
  }
  $("evalCard").style.display = "";
  const tab = S._evalTab;
  const content = $("evalContent");
  const tabs = document.querySelectorAll("#evalTabs .eval-tab");
  tabs.forEach(t => t.classList.toggle("active", t.dataset.eval === tab));

  let html = "";
  if (tab === "combined") {
    html = renderEvalPanel(data.combined || data.decomposed);
    if (data.warnings && data.warnings.length) {
      html += `<div class="eval-warnings">⚠ ${data.warnings.map(esc).join("<br>")}</div>`;
    }
  } else if (tab === "decomposed") {
    html = renderEvalPanel(data.decomposed);
    if (data.decomposed_details) {
      const dd = data.decomposed_details;
      if (dd.faithfulness_warnings && dd.faithfulness_warnings.length) {
        html += `<div class="eval-warnings">⚠ ${dd.faithfulness_warnings.map(esc).join("<br>")}</div>`;
      }
    }
  } else if (tab === "llm") {
    html = renderEvalPanel(data.llm_judge);
  } else if (tab === "reasoning") {
    const reasoning = data.llm_reasoning || "";
    html = reasoning
      ? `<div class="eval-reasoning">${esc(reasoning)}</div>`
      : '<div class="eval-empty">LLM 评判未启用或暂无评语</div>';
  }
  content.innerHTML = html || '<div class="eval-empty">暂无数据</div>';
}

/* ═══════════════════════════════════════════════════════════════════════
   Downloads
   ═══════════════════════════════════════════════════════════════════════ */

function setDownloads(data) {
  const list = $("downloadList");
  list.innerHTML = "";
  let docxUrl = "";
  let pdfUrl = "";
  for (const f of (data && data.files) || []) {
    const li = document.createElement("li");
    const a = document.createElement("a");
    a.href = f.url;
    a.textContent = `${f.name} (${Math.round((f.size || 0) / 1024)} KB)`;
    a.target = "_blank";
    li.appendChild(a);
    list.appendChild(li);
    // Track first DOCX and PDF for download buttons
    if (!docxUrl && /\.docx$/i.test(f.name)) docxUrl = f.url;
    if (!pdfUrl && /\.pdf$/i.test(f.name)) pdfUrl = f.url;
  }
  if (!list.children.length) list.innerHTML = "<li style='color:var(--c-text3);font-size:12px'>暂无输出文件</li>";

  // Update prominent download buttons
  const btnDocx = $("btnDownloadDocx");
  const btnPdf = $("btnDownloadPdf");
  if (btnDocx) {
    if (docxUrl) { btnDocx.href = docxUrl; btnDocx.style.display = ""; }
    else btnDocx.style.display = "none";
  }
  if (btnPdf) {
    if (pdfUrl) { btnPdf.href = pdfUrl; btnPdf.style.display = ""; }
    else btnPdf.style.display = "none";
  }
}

/* ═══════════════════════════════════════════════════════════════════════
   Chat action messages (clarify & satisfaction)
   ═══════════════════════════════════════════════════════════════════════ */

function addActionMsg(html, buttons) {
  const log = $("chatLog");
  const empty = log.querySelector(".chat-empty");
  if (empty) empty.remove();

  const wrap = document.createElement("div");
  wrap.className = "msg assistant";

  const r = document.createElement("div");
  r.className = "role";
  r.textContent = "助手";

  const b = document.createElement("div");
  b.className = "bubble";
  b.innerHTML = html;

  if (buttons && buttons.length) {
    const btnRow = document.createElement("div");
    btnRow.style.cssText = "margin-top:10px;display:flex;gap:8px;flex-wrap:wrap";
    for (const btn of buttons) {
      const el = document.createElement("button");
      el.className = btn.cls || "btn btn-sm";
      el.textContent = btn.text;
      el.addEventListener("click", btn.onClick);
      btnRow.appendChild(el);
    }
    b.appendChild(btnRow);
  }

  wrap.appendChild(r);
  wrap.appendChild(b);
  log.appendChild(wrap);
  log.scrollTop = log.scrollHeight;
  return wrap;
}

/* ── Clarify in chat ──────────────────────────────────────────────────── */

let _clarifyMsgShown = false;

function showClarifyInChat(questions) {
  if (_clarifyMsgShown) return;
  _clarifyMsgShown = true;
  const qs = questions.length ? questions : ["请补充你希望生成文档的侧重点/受众/篇幅/风格等信息。"];
  const items = qs.map((q, i) => `${i + 1}. ${esc(q)}`).join("<br>");
  addActionMsg(
    `<p>📋 <strong>需要补充信息</strong></p><div style="font-size:13px;line-height:1.7;margin:8px 0">${items}</div><p style="font-size:11px;color:var(--c-text3)">请在聊天框输入回答后发送，或发送 <code>跳过</code></p>`
  );
}

function resetClarifyMsg() { _clarifyMsgShown = false; }

/* ── Satisfaction in chat ─────────────────────────────────────────────── */

let _satisfactionChatMsg = null;  // track the satisfaction message element

function showSatisfactionInChat(stage, previewText, versionNum) {
  const key = stage + "_v" + (versionNum || 1);
  if (S._satisfactionKey === key) return;
  S._satisfactionSubmitting = false;
  S._feedbackStage = stage;
  S._waitingFeedback = false;
  S._lastSatisfactionStage = stage;
  S._lastSatisfactionPreview = previewText;
  S._lastSatisfactionVersion = versionNum;

  const label = stage === "outline" ? "大纲" : "报告正文";
  const ver = versionNum || 1;

  // Show preview in right panel
  const previewTitle = $("satisfactionPreviewTitle");
  const previewContent = $("satisfactionPreviewContent");
  previewTitle.textContent = `📝 ${label}已生成 — V${ver}`;
  const text = previewText || (stage === "outline" ? (S._outline || "") : (S._content || ""));
  previewContent.innerHTML = text ? renderMD(text) : "<p style='color:var(--c-text3)'>（内容加载中…）</p>";

  // Clean up old UI before building new one
  hideSatisfaction();

  // Build buttons
  _satisfactionChatMsg = _renderSatisfactionButtons(label, stage, ver);
  // Set guard AFTER everything is built — prevents duplicate on next poll
  S._satisfactionKey = key;
}

function _renderSatisfactionButtons(label, stage, ver) {
  // Remove old DOM only (don't touch state flags — caller manages _satisfactionKey)
  if (_satisfactionChatMsg && _satisfactionChatMsg.parentNode) {
    _satisfactionChatMsg.remove();
  }
  document.querySelectorAll('[id^="satisfactionBubble_"]').forEach(el => {
    const wrap = el.closest(".msg");
    if (wrap) wrap.remove(); else el.remove();
  });
  _satisfactionChatMsg = null;

  const bubbleId = "satisfactionBubble_" + stage;
  const html = `<div id="${bubbleId}">
    <p>📝 <strong>${label}已生成 — V${ver}</strong></p>
    <p>请查看下方预览区的内容，然后选择：</p>
    <div class="satisf-action-row" style="display:flex;gap:8px;flex-wrap:wrap;margin-top:10px">
      <button class="btn btn-primary btn-sm satisf-satisfied-btn">满意，继续</button>
      <button class="btn btn-outline btn-sm satisf-unsatisfied-btn">不满意，重新生成</button>
    </div>
  </div>`;

  const msg = addActionMsg(html);

  // Find buttons within the newly created wrapper (avoids global getElementById pollution)
  const satisfiedBtn = msg.querySelector(".satisf-satisfied-btn");
  const unsatisfiedBtn = msg.querySelector(".satisf-unsatisfied-btn");
  if (satisfiedBtn) {
    satisfiedBtn.addEventListener("click", function () {
      if (S._satisfactionSubmitting) return; // prevent double-submit
      S._satisfactionSubmitting = true;
      satisfiedBtn.disabled = true;
      if (unsatisfiedBtn) unsatisfiedBtn.disabled = true;
      $("satisfactionPreviewCard").style.display = "none";
      submitSatisfactionChat(true, stage, "");
    });
  }
  if (unsatisfiedBtn) {
    unsatisfiedBtn.addEventListener("click", function () {
      if (S._satisfactionSubmitting) return;
      S._satisfactionSubmitting = true;
      satisfiedBtn.disabled = true;
      unsatisfiedBtn.disabled = true;
      var previewCard = $("satisfactionPreviewCard");
      if (previewCard) previewCard.style.display = "none";
      S._waitingFeedback = true;
      S._feedbackStage = stage;
      _showFeedbackChipsInPlace(stage);
    });
  }

  // Show preview card
  $("satisfactionPreviewCard").style.display = "";

  return msg;
}

/* ── Inline feedback chips (replaces buttons inside satisfaction message) ─ */

let _feedbackChips = new Set();

const FEEDBACK_OPTIONS = [
  "结构不合理", "数据太少/缺失", "语言太官方/生硬",
  "内容偏题/不相关", "篇幅不合适", "逻辑不连贯",
];

function _showFeedbackChipsInPlace(stage) {
  const label = stage === "outline" ? "大纲" : "报告正文";
  _feedbackChips = new Set();
  _selectedScope = "content_only";

  // Find the bubble from the latest satisfaction message wrapper; fall back to global lookup
  var bubble = _satisfactionChatMsg ? _satisfactionChatMsg.querySelector("#satisfactionBubble_" + stage) : null;
  if (!bubble) bubble = document.getElementById("satisfactionBubble_" + stage);
  if (!bubble) return;

  const chipsHTML = FEEDBACK_OPTIONS.map(opt =>
    `<span class="feedback-chip" data-opt="${esc(opt)}">${esc(opt)}</span>`
  ).join("");

  let html = `<p>📋 <strong>对 ${label} 不满意</strong></p>`;
  html += `<p style="font-size:12px;margin:4px 0">请选择不满意的原因（可多选）：</p>`;
  html += `<div class="feedback-chips">${chipsHTML}</div>`;
  if (stage === "content") {
    html += `<p style="font-size:12px;margin:8px 0 4px">重新生成范围：</p>`;
    html += `<div class="feedback-chips">`;
    html += `<span class="feedback-chip feedback-chip-scope active" data-scope="content_only">仅重做正文</span>`;
    html += `<span class="feedback-chip feedback-chip-scope" data-scope="outline">从大纲重来</span>`;
    html += `</div>`;
  }
  html += `<p style="font-size:11px;color:var(--c-text3);margin-top:6px">也可以在聊天框输入具体改进意见后按回车发送。</p>`;
  html += `<div class="satisf-action-row" style="display:flex;gap:8px;flex-wrap:wrap;margin-top:10px">
    <button class="btn btn-primary btn-sm satisf-submit-fb-btn">提交反馈，重新生成</button>
    <button class="btn btn-outline btn-sm satisf-skip-fb-btn">跳过，直接重做</button>
    <button class="btn btn-sm satisf-back-fb-btn">返回</button>
  </div>`;

  bubble.innerHTML = html;

  // Attach click handlers to chips (within bubble only)
  bubble.querySelectorAll(".feedback-chip:not(.feedback-chip-scope)").forEach(chip => {
    chip.addEventListener("click", function () {
      this.classList.toggle("active");
      if (this.classList.contains("active")) {
        _feedbackChips.add(this.dataset.opt);
      } else {
        _feedbackChips.delete(this.dataset.opt);
      }
    });
  });

  // Scope chips
  bubble.querySelectorAll(".feedback-chip-scope").forEach(chip => {
    chip.addEventListener("click", function () {
      const row = this.parentElement;
      row.querySelectorAll(".feedback-chip-scope").forEach(c => c.classList.remove("active"));
      this.classList.add("active");
      _selectedScope = this.dataset.scope;
    });
  });

  // Submit button
  const submitBtn = bubble.querySelector(".satisf-submit-fb-btn");
  if (submitBtn) {
    submitBtn.addEventListener("click", function () {
      if (S._satisfactionSubmitting) return;
      S._satisfactionSubmitting = true;
      const activeChips = bubble.querySelectorAll(".feedback-chip.active:not(.feedback-chip-scope)");
      const checkedOpts = Array.from(activeChips).map(c => c.dataset.opt).filter(Boolean);
      const chatText = $("chatInput").value.trim();
      $("chatInput").value = "";
      const parts = [...checkedOpts];
      if (chatText) parts.push(chatText);
      const fb = parts.join("；");
      S._waitingFeedback = false;
      submitSatisfactionChat(false, stage, fb);
    });
  }

  // Skip button
  const skipBtn = bubble.querySelector(".satisf-skip-fb-btn");
  if (skipBtn) {
    skipBtn.addEventListener("click", function () {
      if (S._satisfactionSubmitting) return;
      S._satisfactionSubmitting = true;
      S._waitingFeedback = false;
      submitSatisfactionChat(false, stage, "");
    });
  }

  // Back button — re-render satisfaction buttons without guard reset
  const backBtn = bubble.querySelector(".satisf-back-fb-btn");
  if (backBtn) {
    backBtn.addEventListener("click", function () {
      S._waitingFeedback = false;
      $("satisfactionPreviewCard").style.display = "";
      // Re-render buttons (guard still active from original showSatisfactionInChat call)
      _satisfactionChatMsg = _renderSatisfactionButtons(label, stage, S._lastSatisfactionVersion || 1);
    });
  }
}

function hideSatisfaction() {
  $("satisfactionPreviewCard").style.display = "none";
  S._satisfactionSubmitting = false;
  S._feedbackStage = "";
  S._waitingFeedback = false;
  // Remove stale DOM nodes to prevent duplicate IDs
  if (_satisfactionChatMsg && _satisfactionChatMsg.parentNode) {
    _satisfactionChatMsg.remove();
  }
  document.querySelectorAll('[id^="satisfactionBubble_"]').forEach(el => {
    const wrap = el.closest(".msg");
    if (wrap) wrap.remove(); else el.remove();
  });
  _satisfactionChatMsg = null;
  _feedbackChips = new Set();
}

function toggleFeedbackChip(el) {
  // Kept for backwards compatibility with any inline onclick handlers
  el.classList.toggle("active");
  if (el.classList.contains("active")) {
    _feedbackChips.add(el.dataset.opt);
  } else {
    _feedbackChips.delete(el.dataset.opt);
  }
}

function toggleScopeChip(el) {
  const row = el.parentElement;
  row.querySelectorAll(".feedback-chip-scope").forEach(c => c.classList.remove("active"));
  el.classList.add("active");
  _selectedScope = el.dataset.scope;
}

let _selectedScope = "content_only";

async function submitSatisfactionChat(satisfied, stage, feedback) {
  const fb = (feedback || "").trim();
  let scope = stage === "outline" ? "outline" : "content_only";
  // Use structured scope selection if available
  if (!satisfied && stage === "content") {
    if (_selectedScope === "outline") {
      scope = "outline";
    } else if (/从大纲|全部重|重新开始|重头/.test(fb)) {
      scope = "outline";
    }
  }
  // Reset structured feedback state
  _selectedScope = "content_only";

  // Save version before regenerating
  if (!satisfied && fb) {
    if (stage === "outline" && S._outline) {
      S._versions.outline = S._versions.outline || [];
      S._versions.outline.push({ version: (S._versions.outline.length + 1), content: S._outline, feedback: fb, ts: Date.now() / 1000 });
    } else if (stage === "content" && S._content) {
      S._versions.content = S._versions.content || [];
      S._versions.content.push({ version: (S._versions.content.length + 1), content: S._content, feedback: fb, ts: Date.now() / 1000 });
    }
  }

  // Show confirmation in chat
  if (satisfied) {
    addMsg("assistant", "✅ 已收到：满意，继续生成…");
  } else {
    addMsg("assistant", `✅ 已收到反馈：${fb || "重新生成"}\n\n🔄 正在重新生成…`);
  }

  hideSatisfaction();

  try {
    await apiPost("/api/satisfaction", {
      task_id: S.taskId,
      stage: stage,
      satisfied: satisfied,
      feedback: fb,
      scope: scope,
    });
    if (satisfied) {
      setStatus("满意，继续生成…", "busy");
    } else {
      setStatus("已提交反馈，重新生成中…", "busy");
      S._outlineDone = false;
      S._lastOperation = "regenerate";
      $("previewBox").textContent = "";
      S._compareData = null;
      startPolling();
    }
    // Don't call pollStatus() here — the 2s interval will pick up changes.
    // Calling it immediately would race with the backend status update and
    // could re-trigger showSatisfactionInChat for the same version.
  } catch (e) {
    toast("提交失败: " + e.message, "err");
  }
}

/* ═══════════════════════════════════════════════════════════════════════
   Version management
   ═══════════════════════════════════════════════════════════════════════ */

async function loadVersions(type) {
  if (!S.taskId) return [];
  try {
    const data = await apiGet(`/api/versions?task_id=${encodeURIComponent(S.taskId)}&type=${type}`);
    const versions = Array.isArray(data.versions) ? data.versions : [];
    S._versions[type] = versions;
    showVersionTags(type);
    return versions;
  } catch {
    return [];
  }
}

function showVersionTags(type) {
  const bar = $("versionBar");
  const tags = $("versionTags");
  const versions = S._versions[type] || [];

  if (versions.length <= 1) {
    bar.style.display = "none";
    return;
  }

  bar.style.display = "flex";
  tags.innerHTML = "";
  for (const v of versions) {
    const vn = v.version || 1;
    const tag = document.createElement("span");
    tag.className = "version-tag";
    if (v.selected) tag.classList.add("selected");
    if (S._selectedVersion[type] === vn) tag.classList.add("active");
    const fbShort = (v.feedback || "").slice(0, 30);
    tag.title = v.feedback ? `反馈: ${v.feedback}` : `版本 ${vn}` + (v.selected ? " (已选中)" : "");

    // Version number (click to preview)
    const vLabel = document.createElement("span");
    vLabel.textContent = `V${vn}`;
    vLabel.style.cursor = "pointer";
    vLabel.onclick = () => {
      switchVersion(type, vn);
      if (S.previewTab === "compare") {
        const selRight = $("compareVerRight");
        if (selRight && vn) {
          const options = Array.from(selRight.options).map(o => parseInt(o.value));
          if (options.includes(vn)) {
            selRight.value = vn;
            S._cachedDiff = null;
            refreshBothCompareSides();
          }
        }
      } else {
        showPreview();
      }
    };
    tag.appendChild(vLabel);

    // Delete button (×) — only if more than 1 version exists
    if (versions.length > 1) {
      const delBtn = document.createElement("span");
      delBtn.className = "version-del-btn";
      delBtn.textContent = "×";
      delBtn.title = "删除此版本";
      delBtn.onclick = async (e) => {
        e.stopPropagation();
        if (!confirm(`确定要删除 V${vn} 吗？此操作不可撤销。`)) return;
        try {
          const data = await apiPost("/api/versions/delete", {
            task_id: S.taskId, type: type, version: vn,
          });
          if (data && data.ok) {
            toast(`已删除 V${vn}`, "ok", 2000);
            await loadVersions(type);
            showVersionTags(type);
            showPreview();
          } else {
            toast(data.message || "删除失败", "err");
          }
        } catch (e) {
          toast("删除失败: " + e.message, "err");
        }
      };
      tag.appendChild(delBtn);
    }

    tags.appendChild(tag);
  }
}

function switchVersion(type, versionNum) {
  const versions = S._versions[type] || [];
  const found = versions.find(v => v.version === versionNum);
  if (!found || !found.content) return;

  S._selectedVersion[type] = versionNum;
  showVersionTags(type);

  if (type === "outline") {
    S._outline = found.content;
  } else {
    S._content = found.content;
  }

  if (S.previewTab === type) {
    showPreview();
  } else if (S.previewTab === "compare") {
    updateCompareFromSelection(type);
  }
}

/* ── Cross-version comparison ─────────────────────────────────────────── */

function populateCompareSelectors() {
  const type = (S.previewTab === "outline" || S.previewTab === "compare_outline") ? "outline" : "content";
  const versions = S._versions[type] || [];
  const selLeft = $("compareVerLeft");
  const selRight = $("compareVerRight");

  if (!selLeft || !selRight) return;

  selLeft.innerHTML = "";
  selRight.innerHTML = "";

  for (const v of versions) {
    const vn = v.version || 1;
    const fbShort = (v.feedback || "").slice(0, 25);
    const label = fbShort ? `V${vn} — ${fbShort}` : `V${vn}`;
    const optL = document.createElement("option");
    optL.value = vn;
    optL.textContent = label;
    selLeft.appendChild(optL);
    const optR = document.createElement("option");
    optR.value = vn;
    optR.textContent = label;
    selRight.appendChild(optR);
  }

  if (versions.length >= 2) {
    selLeft.value = versions[versions.length - 2].version;
    selRight.value = versions[versions.length - 1].version;
    $("diffStats").textContent = "";
  } else if (versions.length === 1) {
    selLeft.value = versions[0].version;
    selRight.value = versions[0].version;
    $("diffStats").textContent = "（该类型仅有 1 个版本，无法对比差异）";
  } else {
    $("diffStats").textContent = "（该类型暂无版本）";
  }
}

/* ── Line diff algorithm (LCS-based) ─────────────────────────────────── */

function computeLineDiff(oldText, newText) {
  const oldLines = (oldText || "").split("\n");
  const newLines = (newText || "").split("\n");
  const MAX = 400;
  if (oldLines.length > MAX || newLines.length > MAX) {
    // Fallback: simple line-by-line comparison within bounds
    const minLen = Math.min(oldLines.length, newLines.length);
    const result = { old: [], new: [] };
    for (let i = 0; i < Math.max(oldLines.length, newLines.length); i++) {
      if (i < minLen && oldLines[i] === newLines[i]) {
        result.old.push({ type: "same", text: oldLines[i] });
        result.new.push({ type: "same", text: newLines[i] });
      } else {
        if (i < oldLines.length) result.old.push({ type: "removed", text: oldLines[i] });
        if (i < newLines.length) result.new.push({ type: "added", text: newLines[i] });
      }
    }
    return result;
  }

  const m = oldLines.length, n = newLines.length;
  // LCS DP with typed array for speed
  const dp = new Uint16Array((m + 1) * (n + 1));
  const idx = (i, j) => i * (n + 1) + j;

  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (oldLines[i - 1] === newLines[j - 1]) {
        dp[idx(i, j)] = dp[idx(i - 1, j - 1)] + 1;
      } else {
        dp[idx(i, j)] = Math.max(dp[idx(i - 1, j)], dp[idx(i, j - 1)]);
      }
    }
  }

  // Backtrack
  const oldResult = [], newResult = [];
  let i = m, j = n;
  const stack = [];
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && oldLines[i - 1] === newLines[j - 1]) {
      stack.push({ type: "same", text: oldLines[i - 1] });
      i--; j--;
    } else if (j > 0 && (i === 0 || dp[idx(i, j - 1)] >= dp[idx(i - 1, j)])) {
      stack.push({ type: "added", text: newLines[j - 1] });
      j--;
    } else {
      stack.push({ type: "removed", text: oldLines[i - 1] });
      i--;
    }
  }

  // Build separate old/new line arrays by replaying the diff
  for (const item of stack.reverse()) {
    if (item.type === "same") {
      oldResult.push({ type: "same", text: item.text });
      newResult.push({ type: "same", text: item.text });
    } else if (item.type === "removed") {
      oldResult.push({ type: "removed", text: item.text });
      // new: skip (was removed)
    } else { // added
      // old: skip
      newResult.push({ type: "added", text: item.text });
    }
  }

  return { old: oldResult, new: newResult };
}

/* ── Diff-aware markdown rendering ───────────────────────────────────── */

function renderDiffContent(rawText, lineTypes, side) {
  // Group consecutive lines of same display type, render as markdown blocks
  const blocks = [];
  let currentType = null;
  let currentLines = [];

  for (const dl of lineTypes) {
    const show = (side === "old")
      ? (dl.type === "added" ? "skip" : dl.type)   // old side: skip added lines
      : (dl.type === "removed" ? "skip" : dl.type); // new side: skip removed lines

    if (show === "skip") {
      if (currentLines.length > 0) {
        blocks.push({ type: currentType, text: currentLines.join("\n") });
        currentLines = [];
      }
      currentType = null;
      continue;
    }

    if (show !== currentType && currentLines.length > 0) {
      blocks.push({ type: currentType, text: currentLines.join("\n") });
      currentLines = [];
    }
    currentType = show;
    currentLines.push(dl.text);
  }
  if (currentLines.length > 0) {
    blocks.push({ type: currentType, text: currentLines.join("\n") });
  }

  // Render blocks with diff styling
  let html = "";
  for (const block of blocks) {
    const rendered = renderMD(block.text);
    if (block.type === "removed") {
      html += `<div class="diff-removed">${rendered}</div>`;
    } else if (block.type === "added") {
      html += `<div class="diff-added">${rendered}</div>`;
    } else {
      html += rendered;
    }
  }
  return html || "<p style='color:var(--c-text3)'>（无内容）</p>";
}

function updateDiffCache() {
  // Recompute diff when either side changes
  const type = getCompareType();
  const versions = S._versions[type] || [];
  const lv = parseInt($("compareVerLeft").value) || 1;
  const rv = parseInt($("compareVerRight").value) || 1;
  const oldVer = versions.find(v => v.version === lv);
  const newVer = versions.find(v => v.version === rv);
  const oldText = oldVer ? (oldVer.content || "") : "";
  const newText = newVer ? (newVer.content || "") : "";

  if (S._cachedDiff && S._cachedDiff.oldText === oldText && S._cachedDiff.newText === newText) {
    return S._cachedDiff;
  }

  const diff = computeLineDiff(oldText, newText);
  // Count stats
  let added = 0, removed = 0;
  for (const d of diff.old) { if (d.type === "removed") removed++; }
  for (const d of diff.new) { if (d.type === "added") added++; }
  diff.addedCount = added;
  diff.removedCount = removed;

  S._cachedDiff = { oldText, newText, diff };
  return S._cachedDiff;
}

function onDiffToggle() {
  S._showDiff = $("showDiff").checked;
  $("diffStats").style.display = S._showDiff ? "" : "none";
  S._cachedDiff = null;
  refreshBothCompareSides();
}

function getCompareType() {
  // Use the dropdown selector in the compare toolbar when in compare mode
  if (S.previewTab === "compare") {
    const sel = $("compareTypeSelect");
    if (sel && sel.value) return sel.value;
    // Fallback to stored source tab
    return S._compareSourceTab === "outline" ? "outline" : "content";
  }
  return S.previewTab === "outline" ? "outline" : "content";
}

function updateCompareFromSelection(type) {
  populateCompareSelectors();
  refreshBothCompareSides();
}

function showCompare() {
  // Auto-select type with most versions; prefer the source tab if both have ≥2
  const ov = S._versions.outline || [];
  const cv = S._versions.content || [];
  let bestType = S._compareSourceTab || "outline";
  if (ov.length > cv.length && ov.length >= 2) bestType = "outline";
  else if (cv.length > ov.length && cv.length >= 2) bestType = "content";
  else if (ov.length < 2 && cv.length >= 2) bestType = "content";
  else if (cv.length < 2 && ov.length >= 2) bestType = "outline";

  // Set the type selector
  const typeSel = $("compareTypeSelect");
  if (typeSel) {
    typeSel.value = bestType;
    // Show counts in dropdown
    typeSel.querySelectorAll("option").forEach(opt => {
      const cnt = opt.value === "outline" ? ov.length : cv.length;
      opt.textContent = opt.value === "outline" ? `大纲 (${cnt}版)` : `正文 (${cnt}版)`;
    });
  }

  const versions = S._versions[bestType] || [];
  if (versions.length < 1) {
    toast("暂无版本可对比", "info", 3000);
    return;
  }

  // Show version bar for the current compare type
  showVersionTags(bestType);

  populateCompareSelectors();

  // Show diff toolbar
  $("compareToolbar").style.display = "flex";
  $("showDiff").checked = S._showDiff;
  $("showDiff").onchange = onDiffToggle;
  $("diffStats").style.display = S._showDiff ? "" : "none";

  // Set up change handlers — refresh both sides when either selector changes
  $("compareVerLeft").onchange = () => refreshBothCompareSides();
  $("compareVerRight").onchange = () => refreshBothCompareSides();

  // Type selector change: reload versions and re-render compare view
  if (typeSel) typeSel.onchange = () => { populateCompareSelectors(); refreshBothCompareSides(); showVersionTags(getCompareType()); };

  // Load initial content
  refreshBothCompareSides();

  // Set up accept/redo buttons
  $("btnAcceptLeft").onclick = () => acceptVersionFromCompare("left");
  $("btnAcceptRight").onclick = () => acceptVersionFromCompare("right");
  $("btnRedoLeft").onclick = () => redoVersionFromCompare("left");
  $("btnRedoRight").onclick = () => redoVersionFromCompare("right");

  // Show compare view
  $("compareBox").style.display = "grid";
  $("previewBox").style.display = "none";
}

function refreshBothCompareSides() {
  const type = getCompareType();
  const versions = S._versions[type] || [];
  const lv = parseInt($("compareVerLeft").value) || 1;
  const rv = parseInt($("compareVerRight").value) || 1;

  // Update labels
  const lVer = versions.find(v => v.version === lv);
  const rVer = versions.find(v => v.version === rv);
  const lbl = (v, vn) => v ? `V${vn}${v.feedback ? " — " + v.feedback.slice(0, 30) : ""}` : `V${vn}`;
  $("compareLabelLeft").textContent = lbl(lVer, lv);
  $("compareLabelRight").textContent = lbl(rVer, rv);

  // Invalidate diff cache
  S._cachedDiff = null;

  const oldText = lVer ? (lVer.content || "") : "";
  const newText = rVer ? (rVer.content || "") : "";

  if (S._showDiff && lv !== rv) {
    const cached = updateDiffCache();
    const diff = cached.diff;
    $("diffStats").textContent = `+${diff.addedCount || 0} 行 −${diff.removedCount || 0} 行`;
    $("compareOld").innerHTML = renderDiffContent(cached.oldText, diff.old, "old");
    $("compareNew").innerHTML = renderDiffContent(cached.newText, diff.new, "new");
  } else {
    $("diffStats").textContent = lv === rv ? "（同一版本）" : "";
    $("compareOld").innerHTML = renderMD(oldText || "（无内容）");
    $("compareNew").innerHTML = renderMD(newText || "（无内容）");
  }
}

function acceptVersionFromCompare(side) {
  const type = getCompareType();
  const sel = side === "left" ? $("compareVerLeft") : $("compareVerRight");
  const versionNum = parseInt(sel.value) || 1;
  selectVersion(type, versionNum);
}

function redoVersionFromCompare(side) {
  const type = getCompareType();
  const sel = side === "left" ? $("compareVerLeft") : $("compareVerRight");
  const versionNum = parseInt(sel.value) || 1;

  S._waitingRedo = true;
  S._redoType = type;
  S._redoBaseVersion = versionNum;

  const label = type === "outline" ? "大纲" : "报告正文";
  const versions = S._versions[type] || [];
  const baseVer = versions.find(v => v.version === versionNum);
  const labelExtra = baseVer && baseVer.feedback ? `（${baseVer.feedback.slice(0, 40)}）` : "";

  addActionMsg(
    `<p>🔄 <strong>基于 ${label} V${versionNum} 重新生成</strong> ${labelExtra}</p>
     <p>请输入<strong>改进意见</strong>，例如：请增加数据分析章节、减少技术术语。</p>
     <p style="font-size:11px;color:var(--c-text3)">在聊天框输入后发送</p>`,
    [
      {
        text: "跳过，直接重做", cls: "btn btn-outline btn-sm",
        onClick: function() {
          const row = this.parentElement;
          row.querySelectorAll("button").forEach(b => b.disabled = true);
          doRedoVersion(type, versionNum, "");
        }
      }
    ]
  );
}

async function doRedoVersion(type, versionNum, feedback) {
  const fb = (feedback || "").trim();
  addMsg("assistant", `✅ 基于 ${type === "outline" ? "大纲" : "报告正文"} V${versionNum} 重新生成` + (fb ? `\n\n反馈：${fb}` : ""));

  S._waitingRedo = false;

  try {
    await apiPost("/api/versions/redo", {
      task_id: S.taskId,
      type: type,
      base_version: versionNum,
      feedback: fb,
    });
    setStatus("重新生成中…", "busy");
    S._outlineDone = false;
    S.generating = true;
    S._lastOperation = "redo_version";
    $("previewBox").textContent = "";
    $("compareBox").style.display = "none";
    $("compareToolbar").style.display = "none";
    $("previewBox").style.display = "";
    $("btnUpload").disabled = true;
    $("btnAppend").disabled = true;
    S._satisfactionKey = "";
    startPolling();
  } catch (e) {
    toast("重新生成失败: " + e.message, "err");
    S.generating = false;
    $("btnUpload").disabled = false;
    $("btnAppend").disabled = false;
  }
}

async function selectVersion(type, versionNum) {
  if (!S.taskId) return;
  try {
    await apiPost("/api/versions/select", {
      task_id: S.taskId,
      type: type,
      version: versionNum,
    });
    S._selectedVersion[type] = versionNum;
    // Update versions metadata
    const versions = S._versions[type] || [];
    for (const v of versions) {
      v.selected = (v.version === versionNum);
    }
    showVersionTags(type);
    toast(`已选择 V${versionNum} 作为最终版本`, "ok", 3000);

    // Reload preview with selected version content
    const found = versions.find(v => v.version === versionNum);
    if (found && found.content) {
      if (type === "outline") {
        S._outline = found.content;
      } else {
        S._content = found.content;
      }
    }
    S.previewTab = type;
    // Switch preview tab button
    const container = $("previewTabs");
    container.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    const activeTab = container.querySelector(`[data-tab="${type}"]`);
    if (activeTab) activeTab.classList.add("active");
    showPreview();
    $("compareBox").style.display = "none";
    $("compareToolbar").style.display = "none";
    $("previewBox").style.display = "";
  } catch (e) {
    toast("选择版本失败: " + e.message, "err");
  }
}

/* ═══════════════════════════════════════════════════════════════════════
   Chat
   ═══════════════════════════════════════════════════════════════════════ */

function clearChat() {
  $("chatLog").innerHTML = "";
}

function showChatSummary(text) {
  const log = $("chatLog");
  // Remove any existing summary banner
  const existing = log.querySelector(".chat-summary-banner");
  if (existing) existing.remove();

  const banner = document.createElement("div");
  banner.className = "chat-summary-banner";
  banner.innerHTML = `<span class="summary-label">对话摘要</span><span class="summary-text">${esc(text)}</span>`;
  banner.onclick = () => { banner.classList.toggle("collapsed"); };
  const first = log.firstChild;
  if (first) {
    log.insertBefore(banner, first);
  } else {
    log.appendChild(banner);
  }
}

function addMsg(role, text) {
  const log = $("chatLog");
  // Remove empty-state if present
  const empty = log.querySelector(".chat-empty");
  if (empty) empty.remove();

  const wrap = document.createElement("div");
  wrap.className = "msg " + role;
  const r = document.createElement("div");
  r.className = "role";
  r.textContent = role === "user" ? "用户" : "助手";
  const b = document.createElement("div");
  b.className = "bubble";
  b.innerHTML = renderMD(text || "");
  wrap.appendChild(r);
  wrap.appendChild(b);
  log.appendChild(wrap);
  log.scrollTop = log.scrollHeight;

  // Persist to history
  S.history.push({ role: role, content: text || "" });
  S._historyDirty = true;
  _debouncedSyncHistory();
}

let _syncTimer = null;
function _debouncedSyncHistory() {
  if (_syncTimer) clearTimeout(_syncTimer);
  _syncTimer = setTimeout(_syncHistory, 800);
}

async function _syncHistory() {
  _syncTimer = null;
  if (!S._historyDirty || !S.taskId) return;
  // Only sync the last unsaved message to avoid re-sending everything
  const items = S.history;
  if (!items.length) return;
  const last = items[items.length - 1];
  try {
    await apiPost("/api/chat/history/save", {
      task_id: S.taskId, role: last.role, content: last.content,
    });
    S._historyDirty = false;
  } catch (e) { /* silently ignore save failures */ }
}

// Streaming helpers
let _streamWrap = null;
let _streamBubble = null;
let _streamRaw = "";

function ensureStreamBubble() {
  if (_streamBubble) return;
  const log = $("chatLog");
  const empty = log.querySelector(".chat-empty");
  if (empty) empty.remove();

  _streamWrap = document.createElement("div");
  _streamWrap.className = "msg assistant";
  const r = document.createElement("div");
  r.className = "role";
  r.textContent = "助手";
  _streamBubble = document.createElement("div");
  _streamBubble.className = "bubble";
  _streamBubble.innerHTML = "";
  _streamWrap.appendChild(r);
  _streamWrap.appendChild(_streamBubble);
  log.appendChild(_streamWrap);
}

function appendStreamToken(text) {
  ensureStreamBubble();
  _streamRaw += text;
  _streamBubble.innerHTML = renderMD(_streamRaw);
  $("chatLog").scrollTop = $("chatLog").scrollHeight;
}

function finalizeStreamBubble() {
  const text = _streamRaw;
  _streamBubble = null;
  _streamWrap = null;
  _streamRaw = "";
  return text;
}

function renderHistory(items) {
  clearChat();
  for (const it of items || []) {
    if (it && it.role && it.content) addMsg(it.role, it.content);
  }
}

/* ── Send chat ──────────────────────────────────────────────────────── */

async function sendChat() {
  if (S._sending) return;
  const msg = $("chatInput").value.trim();
  if (!msg) return;
  if (!S.taskId) setStatus("通用助手模式（无任务）", "ok");

  S._sending = true;
  $("chatInput").value = "";
  addMsg("user", msg);

  // If waiting for section regen feedback, route to section regen
  if (S._waitingRegenFeedback) {
    const info = S._waitingRegenFeedback;
    S._waitingRegenFeedback = null;

    // Collect checked chips + typed message
    const activeChips = document.querySelectorAll(".regen-chip.active");
    const chipTexts = Array.from(activeChips).map(c => c.dataset.opt).filter(Boolean);
    const parts = [...chipTexts];
    if (msg && msg !== "跳过") parts.push(msg);
    const feedback = parts.join("；");

    // Remove the chips bubble
    if (info.msg && info.msg.parentNode) info.msg.remove();

    S._lastOperation = "regen_section";
    $("btnSend").disabled = true;
    setStatus(`重新生成章节「${info.heading}」…`, "busy");
    $("previewBox").textContent = "";

    const action = { type: "regenerate", scope: "section", section: info.heading, feedback: feedback };
    S.history.push({ role: "user", content: msg || "（快捷选项：" + chipTexts.join("、") + "）" });

    try {
      const data = await apiPost("/api/chat", { task_id: S.taskId, message: msg, history: S.history, action: action });
      if (data && data.reply) {
        addMsg("assistant", data.reply);
        S.history.push({ role: "assistant", content: data.reply });
      }
      startPolling();
    } catch (e) {
      setStatus(e.message, "err");
      toast("重生成失败: " + e.message, "err");
      S.generating = false;
      $("btnUpload").disabled = false;
      $("btnAppend").disabled = false;
    }
    S._sending = false;
    $("btnSend").disabled = false;
    return;
  }

  // If waiting for satisfaction feedback, route to satisfaction
  if (S._waitingFeedback) {
    S._waitingFeedback = false;
    // Collect checked feedback chips + typed message
    const chips = document.querySelectorAll(".feedback-chip.active:not(.feedback-chip-scope)");
    const checkedOpts = Array.from(chips).map(c => c.dataset.opt).filter(Boolean);
    const parts = [...checkedOpts];
    if (msg) parts.push(msg);
    const fb = parts.join("；");
    $("btnSend").disabled = true;
    setStatus("提交反馈中…", "busy");
    await submitSatisfactionChat(false, S._feedbackStage, fb);
    S._sending = false;
    $("btnSend").disabled = false;
    return;
  }

  // If waiting for redo feedback, route to redo
  if (S._waitingRedo) {
    S._waitingRedo = false;
    $("btnSend").disabled = true;
    setStatus("重新生成中…", "busy");
    await doRedoVersion(S._redoType, S._redoBaseVersion, msg);
    S._sending = false;
    $("btnSend").disabled = false;
    return;
  }

  S.history.push({ role: "user", content: msg });
  setStatus("思考中…", "busy");
  $("btnSend").disabled = true;

  try {
    const r = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task_id: S.taskId, message: msg, history: S.history }),
    });
    if (!r.ok) {
      const d = await r.json().catch(() => ({}));
      const err = d.error || "对话失败";
      addMsg("assistant", "❌ " + err);
      setStatus(err, "err");
      $("btnSend").disabled = false;
      return;
    }

    // Detect action responses (JSON) vs SSE streaming
    const contentType = r.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
      const data = await r.json();
      if (data.reply) {
        addMsg("assistant", data.reply);
        S.history.push({ role: "assistant", content: data.reply });
      }
      if (data.action && data.action.type === "regenerate") {
        setStatus("重新生成中…", "busy");
        S._outlineDone = false;
        $("previewBox").textContent = "";
        startPolling();
        await pollStatus();
      } else {
        setStatus("", "ok");
      }
      $("btnSend").disabled = false;
      return;
    }

    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    setStatus("", "ok");
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const payload = line.slice(6).trim();
        if (payload === "[DONE]") {
          const fullReply = finalizeStreamBubble();
          if (fullReply) S.history.push({ role: "assistant", content: fullReply });
          setStatus("", "ok");
          break;
        }
        try {
          const obj = JSON.parse(payload);
          if (obj.token) appendStreamToken(obj.token);
          else if (obj.status === "streaming") { setStatus("输出中…", "ok"); ensureStreamBubble(); }
          else if (obj.error) { addMsg("assistant", "❌ " + obj.error); setStatus(obj.error, "err"); }
        } catch {}
      }
    }
    await pollStatus();
  } catch (e) {
    addMsg("assistant", "❌ 请求异常: " + e.message);
    setStatus(e.message, "err");
  } finally {
    $("btnSend").disabled = false;
    S._sending = false;
  }
}

/* ═══════════════════════════════════════════════════════════════════════
   Task load by ID
   ═══════════════════════════════════════════════════════════════════════ */

async function loadTaskById() {
  const tid = $("taskIdInput").value.trim();
  if (!tid) { toast("请输入 task_id", "err"); return; }
  setTaskBadge(tid);
  $("previewBox").textContent = "";
  setDownloads({ files: [] });
  setStatus("加载中…", "busy");
  stopPolling();
  $("evalCard").style.display = "none";
  S._evalData = null;

  try {
    const hr = await fetch(`/api/chat/history?task_id=${encodeURIComponent(tid)}`);
    const hd = await hr.json();
    if (hr.ok && hd && Array.isArray(hd.history)) {
      S.history = hd.history;
      renderHistory(S.history);
      // Show conversation summary if it exists
      if (hd.summary && String(hd.summary).trim()) {
        showChatSummary(String(hd.summary).trim());
      }
    } else {
      S.history = [];
      clearChat();
    }
  } catch {
    S.history = [];
    clearChat();
  }

  await pollStatus();
  startPolling();
}

/* ═══════════════════════════════════════════════════════════════════════
   Task List
   ═══════════════════════════════════════════════════════════════════════ */

async function loadTaskList() {
  try {
    const data = await apiGet("/api/tasks");
    const tasks = Array.isArray(data.tasks) ? data.tasks : [];
    const sel = $("taskListSelect");
    sel.innerHTML = '<option value="">— 历史任务 —</option>';
    if (!tasks.length) {
      sel.style.display = "none";
      $("btnRefreshTasks").style.display = "none";
      return;
    }
    sel.style.display = "";
    $("btnRefreshTasks").style.display = "";
    for (const t of tasks) {
      const opt = document.createElement("option");
      opt.value = t.task_id || "";
      const statusLabel = { finished: "✓", processing: "⏳", queued: "⌛", failed: "✗", canceled: "⊘", paused: "⏸" }[t.status] || "?";
      const stage = t.stage ? ` [${t.stage}]` : "";
      const chatLabel = t.has_chat ? " 💬" : "";
      opt.textContent = `${statusLabel} ${t.task_id}${stage}${chatLabel}`;
      sel.appendChild(opt);
    }
  } catch {
    // Silently fail, task list is non-critical
  }
}

function onTaskListSelect() {
  const tid = $("taskListSelect").value;
  if (!tid) return;
  $("taskIdInput").value = tid;
  loadTaskById();
}

/* ═══════════════════════════════════════════════════════════════════════
   Knowledge Base
   ═══════════════════════════════════════════════════════════════════════ */

function setKbCurrent(name) {
  S.kb = (name || "").trim() || "default";
  $("kbCurrent").textContent = S.kb;
  $("kbName").value = S.kb;
}

/* ── KB Doc selector ───────────────────────────────────────────────── */

let _kbDocSelected = new Set();

async function loadKbDocs() {
  const kbName = (($("kbDocSelect").value || "").trim() || S.kb || "default");
  const list = $("kbDocList");
  list.innerHTML = "<span style='color:var(--c-text3);font-size:12px'>加载中…</span>";
  $("btnKbDocLoad").disabled = true;
  try {
    const data = await apiGet(`/api/kb/docs?kb=${encodeURIComponent(kbName)}`);
    const docs = Array.isArray(data.docs) ? data.docs : [];
    list.innerHTML = "";
    if (!docs.length) {
      list.innerHTML = "<span style='color:var(--c-text3);font-size:12px'>该知识库暂无文档</span>";
      return;
    }
    for (const d of docs) {
      const did = d.doc_id || "";
      const title = d.title || d.source || did;
      const meta = [d.doc_type, d.file_ext, d.chunk_count + "块"].filter(Boolean).join(" · ");
      const item = document.createElement("label");
      item.className = "kb-doc-item";
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.value = did;
      cb.checked = _kbDocSelected.has(did);
      cb.addEventListener("change", () => {
        if (cb.checked) _kbDocSelected.add(did);
        else _kbDocSelected.delete(did);
        updateKbDocCount();
      });
      item.appendChild(cb);
      const nameSpan = document.createElement("span");
      nameSpan.className = "doc-name";
      nameSpan.textContent = title;
      item.appendChild(nameSpan);
      const metaSpan = document.createElement("span");
      metaSpan.className = "doc-meta";
      metaSpan.textContent = meta;
      item.appendChild(metaSpan);
      list.appendChild(item);
    }
    updateKbDocCount();
  } catch (e) {
    list.innerHTML = "<span style='color:var(--c-red);font-size:12px'>加载失败: " + esc(e.message) + "</span>";
  } finally {
    $("btnKbDocLoad").disabled = false;
  }
}

function updateKbDocCount() {
  const el = $("kbDocCount");
  el.style.display = _kbDocSelected.size ? "" : "none";
  el.textContent = "已选 " + _kbDocSelected.size + " 个文档";
}

function getKbDocIdsStr() {
  return [..._kbDocSelected].join(",");
}

async function loadKbList() {
  setKbStatus("加载中…");
  try {
    const data = await apiGet("/api/kb/list");
    const items = (data && Array.isArray(data.kbs)) ? data.kbs : ["default"];
    const sel = $("kbSelect");
    sel.innerHTML = "";
    for (const name of items) {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      sel.appendChild(opt);
    }
    sel.value = S.kb;
    setKbStatus("");
  } catch (e) {
    setKbStatus("加载失败: " + e.message);
  }
}

function useKbFromUI() {
  const typed = ($("kbName").value || "").trim();
  const chosen = ($("kbSelect").value || "").trim();
  setKbCurrent(typed || chosen || "default");
  setKbStatus("");
}

function setKbStatus(s) { const el = $("kbStatus"); if (el) el.textContent = s || ""; }

async function kbUpload() {
  const files = $("kbFiles").files;
  if (!files.length) { setKbStatus("请选择文件"); return; }
  useKbFromUI();
  const fd = new FormData();
  fd.append("kb", S.kb);
  const dt = ($("kbDocType").value || "").trim();
  if (dt) fd.append("doc_type", dt);
  for (const f of files) fd.append("files", f, f.name);

  setKbStatus("上传中…");
  $("btnKbUpload").disabled = true;
  try {
    const data = await apiPost("/api/kb/upload", fd);
    const results = Array.isArray(data.results) ? data.results : [];
    const ok = results.filter(r => r && r.ok).length;
    const fail = results.filter(r => r && !r.ok).length;
    setKbStatus(`完成：成功 ${ok}，失败 ${fail}`);
    $("kbAnswer").textContent = JSON.stringify(data, null, 2);
    $("kbCitations").textContent = "";
    await loadKbList();
    $("kbSelect").value = S.kb;
  } catch (e) {
    setKbStatus("失败: " + e.message);
  } finally {
    $("btnKbUpload").disabled = false;
  }
}

async function kbQuery() {
  const q = ($("kbQuestion").value || "").trim();
  if (!q) { setKbStatus("请输入问题"); return; }
  useKbFromUI();

  const topK = parseInt(($("kbTopK").value || "6").trim(), 10) || 6;
  let filters = null;
  const fRaw = ($("kbFilters").value || "").trim();
  if (fRaw) {
    try {
      const obj = JSON.parse(fRaw);
      if (obj && typeof obj === "object" && !Array.isArray(obj)) filters = obj;
      else { setKbStatus("filters 需为 JSON 对象"); return; }
    } catch { setKbStatus("filters JSON 解析失败"); return; }
  }

  setKbStatus("检索中…");
  $("btnKbQuery").disabled = true;
  try {
    const data = await apiPost("/api/kb/query", { kb: S.kb, question: q, top_k: topK, filters });
    $("kbAnswer").textContent = data.answer || "";
    renderKbCitations(data.citations || []);
    setKbStatus("");
  } catch (e) {
    setKbStatus("查询失败: " + e.message);
  } finally {
    $("btnKbQuery").disabled = false;
  }
}

function renderKbCitations(items) {
  const box = $("kbCitations");
  if (!box) return;
  const cits = Array.isArray(items) ? items : [];
  if (!cits.length) { box.textContent = "（无引用）"; return; }
  const lines = [];
  let i = 1;
  for (const c of cits) {
    if (!c) continue;
    const score = c.score != null ? Number(c.score).toFixed(3) : "-";
    lines.push(`${i}. [score=${score}] ${c.doc_id || "-"} | ${c.section_path || "-"}`);
    if (c.snippet) lines.push("   " + c.snippet);
    lines.push("");
    i++;
  }
  box.textContent = lines.join("\n").trim();
}

/* ═══════════════════════════════════════════════════════════════════════
   Init
   ═══════════════════════════════════════════════════════════════════════ */

/* ═══════════════════════════════════════════════════════════════════════
   Template Editor
   ═══════════════════════════════════════════════════════════════════════ */

async function loadTplList() {
  try {
    const data = await apiGet("/api/template/custom/list");
    const items = Array.isArray(data.templates) ? data.templates : [];
    // Populate editor dropdown
    const sel = $("tplEditorSelect");
    sel.innerHTML = '<option value="">— 新建模板 —</option>';
    for (const it of items) {
      const opt = document.createElement("option");
      opt.value = it.stripped;
      opt.textContent = `${it.stripped} (${it.variable_count || 0} 变量)`;
      sel.appendChild(opt);
    }
    // Populate quick select
    const qs = $("tplQuickSelect");
    qs.innerHTML = '<option value="">— 或选择已保存的模板 —</option>';
    for (const it of items) {
      const opt = document.createElement("option");
      opt.value = it.stripped;
      opt.textContent = it.stripped;
      qs.appendChild(opt);
    }
  } catch (e) {
    // Silently ignore — template dir may not exist yet
  }
}

async function loadTemplate(name) {
  if (!name) return;
  try {
    const data = await apiGet(`/api/template/custom/${encodeURIComponent(name)}.md`);
    $("tplEditorName").value = name;
    $("tplEditorTextarea").value = data.content || "";
    $("tplEditorSelect").value = name;
    S._tplEditor.currentName = name;
    S._tplEditor.content = data.content || "";
    S._tplEditor.dirty = false;
    refreshTplPreview();
    toast("已加载模板: " + name, "ok", 2000);
  } catch (e) {
    toast("加载失败: " + e.message, "err");
  }
}

async function saveTemplate() {
  const name = $("tplEditorName").value.trim();
  if (!name) { toast("请输入模板名称", "info"); return; }
  const content = $("tplEditorTextarea").value;
  try {
    const res = await apiPost("/api/template/custom/save", { name: name, content: content });
    $("tplEditorSelect").value = name;
    S._tplEditor.currentName = name;
    S._tplEditor.content = content;
    S._tplEditor.dirty = false;
    toast("已保存: " + res.name, "ok", 2000);
    loadTplList();
  } catch (e) {
    toast("保存失败: " + e.message, "err");
  }
}

async function deleteTemplate() {
  let name = $("tplEditorSelect").value;
  if (!name) { toast("请选择要删除的模板", "info"); return; }
  if (!confirm("确定要删除模板「" + name + "」吗？此操作不可撤销。")) return;
  try {
    await apiDelete(`/api/template/custom/${encodeURIComponent(name)}.md`);
    toast("已删除: " + name, "ok", 2000);
    if (S._tplEditor.currentName === name) newTemplate();
    loadTplList();
  } catch (e) {
    toast("删除失败: " + e.message, "err");
  }
}

const TPL_FULL_PRESETS = {
  blank: {
    label: "空白模板",
    content: "# {{title}}\n\n{{document_outline}}\n\n{{document_content}}\n\n---\n任务ID：{{task_id}}",
  },
  academic: {
    label: "学术论文",
    content: "# {{title}}\n\n## 摘要\n{{摘要}}\n\n## 引言\n{{引言}}\n\n## 研究背景\n{{研究背景}}\n\n## 文献综述\n{{文献综述}}\n\n## 研究目的与意义\n{{研究目的与意义}}\n\n## 研究方法\n{{研究方法}}\n\n## 实验设计与数据采集\n{{实验设计与数据采集}}\n\n## 结果分析\n{{结果分析}}\n\n## 讨论\n{{讨论}}\n\n## 局限性\n{{局限性}}\n\n## 结论\n{{结论}}\n\n## 未来研究方向\n{{未来研究方向}}\n\n---\n任务ID：{{task_id}}",
  },
  tech: {
    label: "技术报告",
    content: "# {{title}}\n\n## 项目概述\n{{项目概述}}\n\n## 需求分析\n{{需求分析}}\n\n## 技术方案\n{{技术方案}}\n\n## 系统架构\n{{系统架构}}\n\n## 核心模块设计\n{{核心模块设计}}\n\n## 性能评估\n{{性能评估}}\n\n## 安全设计\n{{安全设计}}\n\n## 部署方案\n{{部署方案}}\n\n## 风险评估\n{{风险评估}}\n\n## 运维建议\n{{运维建议}}\n\n## 总结\n{{总结}}\n\n---\n任务ID：{{task_id}}",
  },
  business: {
    label: "商业计划",
    content: "# {{title}}\n\n## 执行摘要\n{{执行摘要}}\n\n## 市场分析\n{{市场分析}}\n\n## 目标用户\n{{目标用户}}\n\n## 竞争分析\n{{竞争分析}}\n\n## 产品介绍\n{{产品介绍}}\n\n## 商业模式\n{{商业模式}}\n\n## 营销策略\n{{营销策略}}\n\n## 运营计划\n{{运营计划}}\n\n## 团队介绍\n{{团队介绍}}\n\n## 财务预测\n{{财务预测}}\n\n## 融资需求\n{{融资需求}}\n\n## 风险与对策\n{{风险与对策}}\n\n---\n任务ID：{{task_id}}",
  },
};

function newTemplate() {
  const presetKey = ($("tplNewPreset").value) || "blank";
  const preset = TPL_FULL_PRESETS[presetKey] || TPL_FULL_PRESETS.blank;
  $("tplEditorName").value = presetKey === "blank" ? "" : preset.label;
  $("tplEditorTextarea").value = preset.content;
  $("tplEditorSelect").value = "";
  S._tplEditor.currentName = presetKey === "blank" ? "" : preset.label;
  S._tplEditor.content = preset.content;
  S._tplEditor.dirty = presetKey !== "blank";
  refreshTplPreview();
}

async function loadVariables() {
  try {
    const data = await apiGet("/api/template/variables");
    S._tplEditor.variables = data.system_variables || [];
    renderVarList(data.system_variables || [], data.section_variables_note || "");
  } catch (e) {
    $("tplVarList").innerHTML = "<span style='font-size:10px;color:var(--c-text3)'>加载失败</span>";
  }
}

function renderVarList(sysVars, note) {
  const list = $("tplVarList");
  list.innerHTML = "";
  for (const v of sysVars) {
    const chip = document.createElement("div");
    chip.className = "tpl-var-chip";
    chip.textContent = `{{${v.key}}}`;
    chip.title = v.description || v.label;
    chip.innerHTML = `{{<b>${esc(v.key)}</b>}}<span class="var-desc">${esc(v.label)}</span>`;
    chip.addEventListener("click", () => insertVariableAtCursor(v.key));
    list.appendChild(chip);
  }
  $("tplVarNote").textContent = note || "";
}

function insertVariableAtCursor(key) {
  const ta = $("tplEditorTextarea");
  const token = `{{${key}}}`;
  const start = ta.selectionStart;
  const end = ta.selectionEnd;
  ta.value = ta.value.substring(0, start) + token + ta.value.substring(end);
  ta.selectionStart = ta.selectionEnd = start + token.length;
  ta.focus();
  S._tplEditor.dirty = true;
  S._tplEditor.content = ta.value;
  refreshTplPreview();
}

const TPL_PRESETS = {
  academic: {
    label: "学术论文",
    sections: ["摘要", "引言", "研究背景", "文献综述", "研究目的与意义",
               "研究方法", "实验设计", "数据采集", "结果分析", "讨论",
               "局限性", "结论", "未来研究方向"],
  },
  tech: {
    label: "技术报告",
    sections: ["项目概述", "需求分析", "技术方案", "系统架构",
               "核心模块设计", "性能评估", "安全设计", "部署方案",
               "测试报告", "风险评估", "运维建议", "总结"],
  },
  business: {
    label: "商业计划",
    sections: ["执行摘要", "市场分析", "目标用户", "竞争分析",
               "产品介绍", "商业模式", "营销策略", "运营计划",
               "团队介绍", "财务预测", "融资需求", "风险与对策"],
  },
};

function insertPresetSection(sectionName) {
  const ta = $("tplEditorTextarea");
  const token = `## ${sectionName}\n{{${sectionName}}}\n\n`;
  const start = ta.selectionStart;
  const end = ta.selectionEnd;
  ta.value = ta.value.substring(0, start) + token + ta.value.substring(end);
  ta.selectionStart = ta.selectionEnd = start + token.length;
  ta.focus();
  S._tplEditor.dirty = true;
  S._tplEditor.content = ta.value;
  refreshTplPreview();
}

function renderPresetList() {
  const cat = ($("tplPresetCat").value) || "academic";
  const preset = TPL_PRESETS[cat] || TPL_PRESETS.academic;
  const list = $("tplPresetList");
  list.innerHTML = "";
  for (const s of preset.sections) {
    const chip = document.createElement("div");
    chip.className = "tpl-var-chip tpl-preset-chip";
    chip.textContent = "📋 " + s;
    chip.title = "点击插入章节：## " + s;
    chip.addEventListener("click", () => insertPresetSection(s));
    list.appendChild(chip);
  }
}

function refreshTplPreview() {
  const raw = $("tplEditorTextarea").value;
  // Highlight placeholders before markdown rendering
  const highlighted = raw.replace(
    /\{\{(.+?)\}\}/g,
    '<span class="tpl-placeholder">{{$1}}</span>'
  );
  $("tplPreviewBox").innerHTML = renderMD(highlighted);
}

async function useTemplateForTask(name) {
  if (!name) { toast("请先选择或保存模板", "info"); return; }
  if (!S.taskId) { toast("请先生成一个任务或加载已有任务", "info"); return; }
  try {
    const res = await apiPost("/api/template/custom/use", { name: name, task_id: S.taskId });
    toast("已复制到任务: " + res.name, "ok", 2500);
    // Refresh the file list display in the upload card
    const tplList = $("tplList");
    const existing = tplList.querySelectorAll(".file-tag");
    let found = false;
    for (const tag of existing) {
      if (tag.textContent.replace("×", "").trim() === res.name) found = true;
    }
    if (!found) {
      const tag = document.createElement("span");
      tag.className = "file-tag";
      tag.textContent = res.name;
      tplList.appendChild(tag);
    }
  } catch (e) {
    toast("使用失败: " + e.message, "err");
  }
}

async function quickUseTemplate() {
  const name = $("tplQuickSelect").value;
  if (!name) { toast("请选择一个模板", "info"); return; }
  if (!S.taskId) {
    toast("请先上传文件生成任务，或加载已有任务", "info");
    return;
  }
  await useTemplateForTask(name);
}

function initTemplateEditor() {
  loadTplList();
  loadVariables();

  // Tab switching inside the editor card
  $("tplEditorTabBar").addEventListener("click", (e) => {
    const tab = e.target.closest(".tab");
    if (!tab) return;
    const tabName = tab.dataset.tab;
    $("tplEditorTabBar").querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    tab.classList.add("active");
    const parent = $("tplEditorTabBar").parentElement;
    parent.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
    const panel = document.getElementById("tab-" + tabName);
    if (panel) {
      panel.classList.add("active");
      if (tabName === "tpl-preview") refreshTplPreview();
    }
  });

  // Editor textarea changes
  $("tplEditorTextarea").addEventListener("input", () => {
    S._tplEditor.dirty = true;
    S._tplEditor.content = $("tplEditorTextarea").value;
    // Update preview in real-time if visible
    if ($("tab-tpl-preview").classList.contains("active")) {
      refreshTplPreview();
    }
  });

  // CRUD buttons
  $("btnTplLoad").addEventListener("click", () => {
    const name = $("tplEditorSelect").value;
    if (name) loadTemplate(name);
  });
  $("btnTplSave").addEventListener("click", saveTemplate);
  $("btnTplDelete").addEventListener("click", deleteTemplate);
  $("btnTplNew").addEventListener("click", newTemplate);

  // Import local .md file into editor
  $("btnTplImport").addEventListener("click", () => $("tplImportFile").click());
  $("tplImportFile").addEventListener("change", () => {
    const file = $("tplImportFile").files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      $("tplEditorTextarea").value = reader.result || "";
      const baseName = file.name.replace(/\.(md|txt|markdown)$/i, "");
      $("tplEditorName").value = baseName;
      S._tplEditor.currentName = baseName;
      S._tplEditor.content = reader.result || "";
      S._tplEditor.dirty = false;
      refreshTplPreview();
      toast("已导入: " + file.name, "ok", 2000);
    };
    reader.readAsText(file);
    $("tplImportFile").value = "";
  });
  $("btnTplUse").addEventListener("click", () => {
    const name = S._tplEditor.currentName || $("tplEditorSelect").value;
    if (name) useTemplateForTask(name);
    else toast("请先选择或保存一个模板", "info");
  });
  $("btnTplQuickUse").addEventListener("click", quickUseTemplate);

  // Template name input changed — update currentName
  $("tplEditorName").addEventListener("input", () => {
    S._tplEditor.currentName = $("tplEditorName").value.trim();
  });
}

// Simple DELETE wrapper (not in the original helpers)
async function apiDelete(url) {
  const r = await fetch(url, { method: "DELETE" });
  const data = await r.json();
  if (!r.ok) throw new Error((data && data.detail) || r.statusText);
  return data;
}


function init() {
  setupDropZone();
  setupTabs("previewTabs");
  setupTabs("kbTabs");

  // Template preset sections
  $("tplPresetCat").addEventListener("change", renderPresetList);
  renderPresetList();

  // Eval tabs
  $("evalTabs").addEventListener("click", (e) => {
    const tab = e.target.closest(".eval-tab");
    if (!tab) return;
    S._evalTab = tab.dataset.eval;
    renderEval();
  });

  // Upload
  $("btnUpload").addEventListener("click", uploadAndGenerate);
  $("btnAppend").addEventListener("click", appendToTask);

  // Chat
  $("btnSend").addEventListener("click", sendChat);
  $("chatInput").addEventListener("keydown", (e) => { if (e.key === "Enter") sendChat(); });

  // Task
  $("btnLoadTask").addEventListener("click", loadTaskById);
  $("taskIdInput").addEventListener("keydown", (e) => { if (e.key === "Enter") loadTaskById(); });

  // Preview tabs
  $("previewTabs").addEventListener("click", (e) => {
    const tab = e.target.closest(".tab");
    if (!tab) return;
    const tabName = tab.dataset.tab;
    if (tabName === "compare") {
      // Keep track of source type for compare
      if (S.previewTab !== "compare") {
        S._compareSourceTab = S.previewTab;
      }
      S.previewTab = "compare";
      showCompare();
      return;
    }
    S.previewTab = tabName;
    $("compareBox").style.display = "none";
    $("compareToolbar").style.display = "none";
    $("previewBox").style.display = "";
    showPreview();
    // Update version tags for this tab
    const versions = S._versions[tabName] || [];
    if (versions.length > 1) {
      $("tabCompare").style.display = "";
    }
    showVersionTags(tabName);
  });
  // Full-screen preview button
  $("btnFullPreview").addEventListener("click", openFullPreview);

  // KB
  $("btnKbRefresh").addEventListener("click", loadKbList);
  $("btnKbUse").addEventListener("click", useKbFromUI);
  $("btnKbUpload").addEventListener("click", kbUpload);
  $("btnKbQuery").addEventListener("click", kbQuery);
  $("kbQuestion").addEventListener("keydown", (e) => { if (e.key === "Enter") kbQuery(); });
  $("kbSelect").addEventListener("change", () => { const v = ($("kbSelect").value || "").trim(); if (v) setKbCurrent(v); });

  // KB doc selector
  $("btnKbDocLoad").addEventListener("click", loadKbDocs);
  // Sync KB doc selector dropdown with main KB list
  const syncKbDocSelect = async () => {
    try {
      const data = await apiGet("/api/kb/list");
      const items = (data && Array.isArray(data.kbs)) ? data.kbs : ["default"];
      const sel = $("kbDocSelect");
      sel.innerHTML = "";
      for (const name of items) {
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name;
        sel.appendChild(opt);
      }
      sel.value = S.kb;
    } catch {}
  };
  syncKbDocSelect();

  // Toggle KB name field when "add to KB" checkbox changes
  $("addToKb").addEventListener("change", () => {
    $("kbNameField").style.display = $("addToKb").checked ? "" : "none";
  });

  // Startup
  initTemplateEditor();
  setKbCurrent("default");
  loadKbList();
  loadTaskList();
  $("btnRefreshTasks").addEventListener("click", loadTaskList);
  const qs = new URLSearchParams(window.location.search);
  const tid = (qs.get("task_id") || "").trim();
  if (tid) {
    $("taskIdInput").value = tid;
    loadTaskById();
  }
}

document.addEventListener("DOMContentLoaded", init);
