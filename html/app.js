/* ═══════════════════════════════════════════════════════════════════════
   State
   ═══════════════════════════════════════════════════════════════════════ */

const S = {
  taskId: "",
  pollTimer: null,
  history: [],
  clarifyShown: false,
  kb: "default",
  lastAbText: "",
  previewTab: "outline",
  kbTab: "kb-query",
  generating: false,
  _outlineDone: false,
  _sending: false,
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

/* ── Simple Markdown-ish rendering ──────────────────────────────────── */

function renderMD(text) {
  let h = esc(text);
  // Bold: **text**
  h = h.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  // Inline code: `text`
  h = h.replace(/`([^`]+)`/g, "<code>$1</code>");
  // Unordered lists: lines starting with - or *
  h = h.replace(/(^|\n)[-*] (.+?)(?=\n|$)/g, (m, nl, item) => nl + "<li>" + item + "</li>");
  // Wrap consecutive <li> in <ul>
  h = h.replace(/((?:<li>.*?<\/li>\n?)+)/g, "<ul>$1</ul>");
  // Line breaks
  h = h.replace(/\n/g, "<br>");
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
  if ($("abEval").checked) fd.append("ab_eval", "1");
  if ($("addToKb").checked) {
    fd.append("add_to_kb", "true");
    const kbn = ($("kbNameInUpload").value || "").trim();
    if (kbn) fd.append("kb_name", kbn);
  }
  const kbDocIds = getKbDocIdsStr();
  if (kbDocIds) fd.append("kb_doc_ids", kbDocIds);

  S.generating = true;
  $("btnUpload").disabled = true;
  $("btnAppend").disabled = true;
  setStatus("上传中…", "busy");
  S.lastAbText = "";
  $("abBox").textContent = "";
  $("abCard").style.display = "none";
  hideClarify();
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
  if ($("abEval").checked) fd.append("ab_eval", "1");
  if ($("addToKb").checked) {
    fd.append("add_to_kb", "true");
    const kbn = ($("kbNameInUpload").value || "").trim();
    if (kbn) fd.append("kb_name", kbn);
  }
  const kbDocIds = getKbDocIdsStr();
  if (kbDocIds) fd.append("kb_doc_ids", kbDocIds);

  S.generating = true;
  $("btnUpload").disabled = true;
  $("btnAppend").disabled = true;
  setStatus("追加中…", "busy");
  S._outlineDone = false;
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
  if (S.pollTimer) clearInterval(S.pollTimer);
  S.pollTimer = setInterval(pollStatus, 2000);
}

function stopPolling() {
  if (S.pollTimer) { clearInterval(S.pollTimer); S.pollTimer = null; }
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
      setStatus(msg || "需要补充信息", "busy");
      showClarify(data.clarify_questions || [], data.clarify_answers || "");
    } else if (st === "processing") {
      setStatus(`生成中 (${stage || "处理中"})`, "busy");
      hideClarify();
      // Load partial content preview during document generation
      if (stage === "document") {
        const content = await fetchText(`/result/${encodeURIComponent(S.taskId)}/content.md`);
        if (content && content.trim()) {
          S._content = content;
          showPreview();
        }
      }
    } else if (st === "queued") {
      setStatus("排队等待…", "busy");
      hideClarify();
    } else if (st === "finished") {
      setStatus("完成", "ok");
      hideClarify();
      stopPolling();
      // Load preview
      const outline = await fetchText(`/result/${encodeURIComponent(S.taskId)}/outline.md`);
      const content = await fetchText(`/result/${encodeURIComponent(S.taskId)}/content.md`);
      S._outline = outline;
      S._content = content;
      showPreview();
      S.generating = false;
      $("btnUpload").disabled = false;
      $("btnAppend").disabled = false;
      loadTaskList();
    } else if (st === "failed") {
      setStatus("失败", "err");
      hideClarify();
      stopPolling();
      toast("生成失败: " + (data.error || msg || "未知"), "err");
      S.generating = false;
      $("btnUpload").disabled = false;
      $("btnAppend").disabled = false;
      loadTaskList();
    } else if (st === "canceled" || st === "cancelled") {
      setStatus("已取消", "ok");
      hideClarify();
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
  const tab = S.previewTab;
  const raw = tab === "outline" ? (S._outline || "（无）") : (S._content || "（无）");
  $("previewBox").innerHTML = tab === "outline" ? renderMD(raw) : renderMD(raw);
}

/* ═══════════════════════════════════════════════════════════════════════
   Downloads
   ═══════════════════════════════════════════════════════════════════════ */

function setDownloads(data) {
  const list = $("downloadList");
  list.innerHTML = "";
  for (const f of (data && data.files) || []) {
    const li = document.createElement("li");
    const a = document.createElement("a");
    a.href = f.url;
    a.textContent = `${f.name} (${Math.round((f.size || 0) / 1024)} KB)`;
    a.target = "_blank";
    li.appendChild(a);
    list.appendChild(li);
  }
  if (!list.children.length) list.innerHTML = "<li style='color:var(--c-text3);font-size:12px'>暂无输出文件</li>";
}

/* ═══════════════════════════════════════════════════════════════════════
   Clarify
   ═══════════════════════════════════════════════════════════════════════ */

function showClarify(questions, answers) {
  const wrap = $("clarifyWrap");
  const qBox = $("clarifyQuestions");
  qBox.innerHTML = "";
  for (const q of (questions.length ? questions : ["请补充你希望生成文档的侧重点/受众/篇幅/风格等信息。"])) {
    const d = document.createElement("div");
    d.className = "qitem";
    d.textContent = q;
    qBox.appendChild(d);
  }
  if (!S.clarifyShown) $("clarifyAnswers").value = answers || "";
  wrap.style.display = "";
  S.clarifyShown = true;
}

function hideClarify() {
  $("clarifyWrap").style.display = "none";
  S.clarifyShown = false;
}

async function submitClarify(skip) {
  if (!S.taskId) return;
  const answers = $("clarifyAnswers").value.trim();
  $("btnClarifySubmit").disabled = $("btnClarifySkip").disabled = true;
  try {
    await apiPost("/api/clarify", { task_id: S.taskId, answers, skip: !!skip });
    hideClarify();
    setStatus("已提交，继续生成…", "busy");
    await pollStatus();
  } catch (e) {
    toast("提交失败: " + e.message, "err");
  } finally {
    $("btnClarifySubmit").disabled = $("btnClarifySkip").disabled = false;
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

function init() {
  setupDropZone();
  setupTabs("previewTabs");
  setupTabs("kbTabs");

  // Upload
  $("btnUpload").addEventListener("click", uploadAndGenerate);
  $("btnAppend").addEventListener("click", appendToTask);

  // Clarify
  $("btnClarifySubmit").addEventListener("click", () => submitClarify(false));
  $("btnClarifySkip").addEventListener("click", () => submitClarify(true));

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
    S.previewTab = tab.dataset.tab;
    showPreview();
  });

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
