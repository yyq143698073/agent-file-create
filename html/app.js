/* ═══════════════════════════════════════════════════════════════════════
   Core infrastructure loaded from /static/js/core.js (loaded first):
   S, UI_STATES, deriveUIState, syncUIState, $, esc, toast, setStatus,
   setTaskBadge, renderMD, _previewCache, setupDropZone, showFileTags,
   setupTabs
   ═══════════════════════════════════════════════════════════════════════ */

/* ═══════════════════════════════════════════════════════════════════════
   API helpers (with global error interception)
   ═══════════════════════════════════════════════════════════════════════ */

async function apiGet(url) {
  try {
    const r = await fetch(url);
    if (!r.ok) {
      const body = await r.json().catch(() => ({}));
      const msg = body.detail || body.error || `服务器错误 (${r.status})`;
      if (r.status >= 500) toast(msg, "error", 6000);
      throw new Error(msg);
    }
    return r.json();
  } catch (e) {
    if (e.name === "TypeError" && e.message.includes("fetch")) {
      toast(LANG.toast_network_error, "error", 8000);
    }
    throw e;
  }
}

async function apiPost(url, body) {
  const opts = { method: "POST" };
  if (body instanceof FormData) {
    opts.body = body;
  } else {
    opts.headers = { "Content-Type": "application/json" };
    opts.body = JSON.stringify(body);
  }
  try {
    const r = await fetch(url, opts);
    if (!r.ok) {
      const resp = await r.json().catch(() => ({}));
      const msg = resp.detail || resp.error || `服务器错误 (${r.status})`;
      if (r.status >= 500) toast(msg, "error", 6000);
      else if (r.status === 409) toast(msg, "warning", 5000);
      throw new Error(msg);
    }
    return r.json();
  } catch (e) {
    if (e.name === "TypeError" && e.message.includes("fetch")) {
      toast(LANG.toast_network_error, "error", 8000);
    }
    throw e;
  }
}

/* ═══════════════════════════════════════════════════════════════════════
   Task upload & append
   ═══════════════════════════════════════════════════════════════════════ */

// ── Generation helpers (shared by uploadAndGenerate & appendToTask) ──

// Build the FormData shared by both upload and append. Includes files,
// templates, user prompt, target words, A/B eval, KB options, and selected
// KB doc ids. Pass `withTaskId` to include the existing task_id (append),
// and `withRetrievalKb` to include the optional retrieval KB selector.
function _buildGenerationFormData(withTaskId, withRetrievalKb) {
  const fd = new FormData();
  if (withTaskId) fd.append("task_id", S.taskId);
  for (const f of ($("files").files || [])) fd.append("files", f, f.name);
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
  if (withRetrievalKb) {
    const retrievalKb = ($("retrievalKbSelect").value || "").trim();
    if (retrievalKb) fd.append("retrieval_kb", retrievalKb);
  }
  return fd;
}

// Common "busy" UI state for generation: disable buttons, hide eval/warning
// cards, reset outline/eval state. `busyText` is the status label to show.
function _setGeneratingBusy(busyText) {
  S.generating = true;
  S._lastOperation = "generate";
  S._contentStreaming = false;
  S._outlineDone = false;
  S._evalData = null;
  $("btnUpload").disabled = true;
  $("btnAppend").disabled = true;
  setStatus(busyText, "busy");
  $("evalCard").style.display = "none";
  $("warningsCard").style.display = "none";
  syncUIState();
}

// Restore interactive UI after a failed generation.
function _endGeneratingBusy() {
  S.generating = false;
  $("btnUpload").disabled = false;
  $("btnAppend").disabled = false;
  syncUIState();
}

async function uploadAndGenerate() {
  if (S.generating) { toast(LANG.toast_task_busy, "info", 2000); return; }
  const files = $("files").files;
  if (!files.length) { toast(LANG.toast_select_files, "error"); return; }

  const fd = _buildGenerationFormData(false, true);

  _setGeneratingBusy(LANG.status_uploading);
  // Upload-only reset: clear A/B panel and satisfaction key
  S._satisfactionKey = "";
  S.lastAbText = "";
  $("abBox").textContent = "";
  $("abCard").style.display = "none";

  try {
    const data = await apiPost("/api/upload", fd);
    setTaskBadge(data.task_id || "");
    // Apply pending template if user selected one before generating
    if (S._pendingTemplate) {
      try {
        await useTemplateForTask(S._pendingTemplate);
        S._pendingTemplate = null;
      } catch (e) {
        toast(e.message, "error");
      }
    }
    S.history = [];
    clearChat();
    $("previewBox").textContent = "";
    setDownloads(data.downloads || {});
    setStatus(LANG.status_generating, "busy");
    startPolling();
    await pollStatus();
  } catch (e) {
    setStatus(e.message, "error");
    toast(LANG.toast_upload_failed + ': ' + e.message, "error");
    _endGeneratingBusy();
  }
}

async function appendToTask() {
  if (S.generating) { toast(LANG.toast_task_busy, "info", 2000); return; }
  const tid = S.taskId;
  if (!tid) { toast(LANG.toast_need_task, "error"); return; }
  const files = $("files").files;
  const templates = $("templates").files;
  if (!files.length && !templates.length) { toast("请选择文件或模板", "error"); return; }

  const fd = _buildGenerationFormData(true, false);

  _setGeneratingBusy("追加中…");

  try {
    const data = await apiPost("/api/append", fd);
    setTaskBadge(data.task_id || tid);
    if (S._pendingTemplate) {
      try {
        await useTemplateForTask(S._pendingTemplate);
        S._pendingTemplate = null;
      } catch (e) {
        toast(e.message, "error");
      }
    }
    setDownloads(data.downloads || {});
    setStatus(LANG.status_regenerating, "busy");
    startPolling();
    await pollStatus();
  } catch (e) {
    setStatus(e.message, "error");
    toast(LANG.toast_append_failed + ': ' + e.message, "error");
    _endGeneratingBusy();
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
      $("progressMsg").textContent = LANG.status_queued;
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
      } else if (stage === "final_confirm") {
        setStatus("请进行最终确认", "busy");
        S._isFinalConfirmStage = true;
        await loadFinalConfirmPreview(data);
        showFinalConfirmInChat(msg || LANG.fc_title);
      } else if (stage === "quality_gate") {
        setStatus(LANG.chat_report_done, "busy");
        showQualityGateInChat(msg || LANG.qg_title);
      } else {
        setStatus(msg || LANG.chat_need_info, "busy");
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
      setStatus(LANG.status_queued, "busy");
      hideSatisfaction();
      S._satisfactionKey = "";
      _clarifyMsgShown = false;
    } else if (st === "finished") {
      setStatus(LANG.status_done, "ok");
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
      // ── Unified eval card ──
      S._evalData = data.eval || null;
      S._evalMetrics = data.eval_metrics || {};
      const merged = Object.assign(
        {}, S._evalMetrics,
        (S._evalData || {}).combined || S._evalData || {}
      );
      const hasEvalData = (data.eval && Object.keys(data.eval).length > 0) ||
        (data.eval_metrics && Object.keys(data.eval_metrics).length > 0);
      if (hasEvalData) {
        $("evalCard").style.display = "";
        renderEvalMetricsBar(merged);
        renderEvalContent(merged);
      } else {
        $("evalCard").style.display = "none";
      }
      // ── Warnings card ──
      if (data.warnings && data.warnings.length) {
        renderWarningsList(data.warnings);
      } else {
        $("warningsCard").style.display = "none";
      }
      // Chat notification on completion
      if (S._lastOperation) {
        const opMessages = {
          generate: LANG.chat_report_generated,
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
      setStatus(LANG.status_failed, "error");
      hideSatisfaction();
      S._satisfactionKey = "";
      S._contentStreaming = false;
      stopPolling();
      toast(LANG.toast_gen_failed + ': ' + (data.error || msg || "\u672A\u77E5"), "error");
      S.generating = false;
      $("btnUpload").disabled = false;
      $("btnAppend").disabled = false;
      loadTaskList();
    } else if (st === "canceled" || st === "cancelled") {
      setStatus(LANG.status_canceled, "ok");
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
      $("abBox").textContent = LANG.status_evaluating;
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
  const raw = tab === "outline" ? (S._outline || "（大纲尚未生成）") : (S._content || "（正文尚未生成）");
  if (S._editingSection) return;
  if (_previewCache.get(tab) === raw) return;
  _previewCache.set(tab, raw);
  const box = $("previewBox");
  const st = box.scrollTop;  // save scroll position
  box.innerHTML = renderMD(raw);
  box.scrollTop = Math.min(st, box.scrollHeight);  // restore (clamped)
  attachSectionActions(tab);
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
  if (!S.taskId) { toast("请先生成或加载一个任务", "error"); return; }
  if (S.generating && !S._isFinalConfirmStage) { toast("任务正在进行中，请等待完成", "info", 2000); return; }

  S.generating = true;
  S._isFinalConfirmStage = false;
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
  syncUIState();
}

function startSectionEdit(headingEl, headingText) {
  // Cancel any previous edit
  if (S._editingSection) cancelSectionEdit();

  const raw = S._content || "";
  const body = extractSectionBody(raw, headingText);
  if (!body && !raw) {
    toast("无法提取章节内容", "error");
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
  syncUIState();
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
  syncUIState();
}

async function submitSectionEdit(headingText, textarea) {
  const edited = textarea.value.trim();
  if (!edited) { toast("编辑内容不能为空", "error"); return; }
  if (!S.taskId) { toast("请先生成或加载一个任务", "error"); return; }

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
    setStatus(e.message, "error");
    toast("编辑重写失败: " + e.message, "error");
    S.generating = false;
    $("btnUpload").disabled = false;
    $("btnAppend").disabled = false;
  }
}

async function saveSectionDirect(headingText, textarea) {
  const edited = textarea.value.trim();
  if (!edited) { toast("编辑内容不能为空", "error"); return; }
  if (!S.taskId) { toast("请先生成或加载一个任务", "error"); return; }

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
      toast(data.message || "保存失败", "error");
    }
    setStatus("已完成", "ok");
  } catch (e) {
    setStatus(e.message, "error");
    toast("直接保存失败: " + e.message, "error");
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

/* ── Eval rendering ── loaded from /static/js/eval.js */
/* ── Downloads ────────────────────────────────────────────────────── */

/* ═══════════════════════════════════════════════════════════════════════
   Chat action messages (clarify & satisfaction)
   ═══════════════════════════════════════════════════════════════════════ */

/* ── Chat module ── loaded from /static/js/chat.js */
/* ── Task load / KB / Template / Init ── loaded from /static/js/features.js */
