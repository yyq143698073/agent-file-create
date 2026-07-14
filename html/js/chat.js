/* ── Chat module ── loaded from /static/js/chat.js */

function addActionMsg(html, buttons) {
  const log = $("chatLog");
  const empty = log.querySelector(".chat-empty");
  if (empty) empty.remove();

  const wrap = document.createElement("div");
  wrap.className = "msg assistant";

  const r = document.createElement("div");
  r.className = "role";
  r.textContent = LANG.role_assistant;

  const b = document.createElement("div");
  b.className = "bubble";
  b.innerHTML = html;

  if (buttons && buttons.length) {
    const btnRow = document.createElement("div");
    btnRow.className = "action-row";
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
    '<p>\uD83D\uDCCB <strong>' + LANG.chat_need_info + '</strong></p><div style="font-size:13px;line-height:1.7;margin:8px 0">' + items + '</div><p class="text-hint">\u8BF7\u5728\u804A\u5929\u6846\u8F93\u5165\u56DE\u7B54\u540E\u53D1\u9001\uFF0C\u6216\u53D1\u9001 <code>\u8DF3\u8FC7</code></p>'
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
  const text = previewText || (stage === "outline" ? (S._outline || "") : (S._content || ""));

  if (stage === "outline") {
    S._outline = text;
  } else {
    S._content = text;
  }

  const previewTitle = $("satisfactionPreviewTitle");
  const previewContent = $("satisfactionPreviewContent");
  previewTitle.textContent = `📝 ${label}已生成 — V${ver}`;
  previewContent.innerHTML = text ? renderMD(text) : '<p class="text-muted">（内容加载中…）</p>';

  hideSatisfaction();

  _satisfactionChatMsg = _renderSatisfactionButtons(label, stage, ver);
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
  const isContent = stage === "content";
  const extraHint = isContent
    ? `<p class="text-hint" style="margin-top:6px">
         💡 可在下方预览区直接<b>编辑段落</b>，或切换<b>历史版本</b>进行对比。<br>
         👍 满意后进入模板渲染和可选的质量评估。</p>`
    : "";

  const html = `<div id="${bubbleId}">
    <p>📝 <strong>${label}已生成 — V${ver}</strong></p>
    <p>请查看下方预览区的内容，然后选择：</p>${extraHint}
    <div class="satisf-action-row action-row">
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
      S._satisfactionSubmitting = true;
      satisfiedBtn.disabled = true;
      if (unsatisfiedBtn) unsatisfiedBtn.disabled = true;
      $("satisfactionPreviewCard").style.display = "none";
      submitSatisfactionChat(true, stage, "");
    });
  }
  if (unsatisfiedBtn) {
    unsatisfiedBtn.addEventListener("click", function () {
      S._satisfactionSubmitting = true;
      satisfiedBtn.disabled = true;
      unsatisfiedBtn.disabled = true;
      var previewCard = $("satisfactionPreviewCard");
      if (previewCard) previewCard.style.display = "none";
      S._waitingFeedback = true;
      S._feedbackStage = stage;
      syncUIState();
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
  // Release submitting guard — we're now in a new UI state (chips)
  S._satisfactionSubmitting = false;

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
  html += `<p class="text-hint" style="margin-top:6px">也可以在聊天框输入具体改进意见后按回车发送。</p>`;
  html += `<div class="action-row">
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
      S._satisfactionSubmitting = true;
      const activeChips = bubble.querySelectorAll(".feedback-chip.active:not(.feedback-chip-scope)");
      const checkedOpts = Array.from(activeChips).map(c => c.dataset.opt).filter(Boolean);
      const chatText = $("chatInput").value.trim();
      $("chatInput").value = "";
      const parts = [...checkedOpts];
      if (chatText) parts.push(chatText);
      const fb = parts.join("；");
      S._waitingFeedback = false;
      syncUIState();
      submitSatisfactionChat(false, stage, fb);
    });
  }

  // Skip button
  const skipBtn = bubble.querySelector(".satisf-skip-fb-btn");
  if (skipBtn) {
    skipBtn.addEventListener("click", function () {
      S._satisfactionSubmitting = true;
      S._waitingFeedback = false;
      syncUIState();
      submitSatisfactionChat(false, stage, "");
    });
  }

  // Back button — re-render satisfaction buttons without guard reset
  const backBtn = bubble.querySelector(".satisf-back-fb-btn");
  if (backBtn) {
    backBtn.addEventListener("click", function () {
      S._waitingFeedback = false;
      syncUIState();
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
  syncUIState();
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

/* ── Quality gate (after render) ────────────────────────────────────────── */

let _qualityGateKey = "";

// Generic helper for the three "yes/no/confirm" prompts in chat:
//   - dedup by a per-prompt key (set via setDedupKey)
//   - remove any prior _satisfactionChatMsg
//   - build a bubble with a row of buttons and call addActionMsg
//   - wire each button's click handler
// Used by showQualityGateInChat, showFinalConfirmInChat (and could absorb
// satisfaction buttons later). Does NOT touch state flags — caller decides.
function _showPromptInChat({ dedupKey, setDedupKey, bubbleId, title, intro, listHTML, buttons }) {
  if (dedupKey === S.taskId) return null;
  if (_satisfactionChatMsg && _satisfactionChatMsg.parentNode) {
    _satisfactionChatMsg.remove();
  }
  const buttonsHTML = buttons.map((b) =>
    `<button class="btn ${b.cls || "btn-outline"} btn-sm">${b.text}</button>`
  ).join("");
  let html = `<div id="${bubbleId}"><p>${title}</p>`;
  if (intro) html += `<p>${intro}</p>`;
  if (listHTML) html += listHTML;
  html += `<div class="action-row">${buttonsHTML}</div></div>`;
  _satisfactionChatMsg = addActionMsg(html);
  const btnEls = _satisfactionChatMsg.querySelectorAll("button");
  // Disable every button in this prompt — used by handlers to lock UI.
  function disableAll() { btnEls.forEach(b => { b.disabled = true; }); }
  buttons.forEach((b, i) => {
    if (btnEls[i]) btnEls[i].addEventListener("click", () => b.handler(btnEls[i], disableAll));
  });
  setDedupKey(S.taskId);
  return _satisfactionChatMsg;
}

// Common: post to /api/satisfaction and remove the prompt bubble from chat.
async function _submitPromptAction(stage, satisfied, feedback) {
  if (_satisfactionChatMsg && _satisfactionChatMsg.parentNode) _satisfactionChatMsg.remove();
  _satisfactionChatMsg = null;
  try {
    await apiPost("/api/satisfaction", {
      task_id: S.taskId, stage: stage, satisfied: satisfied, feedback: feedback || "",
    });
  } catch (e) { /* best-effort; status toast already shown by apiPost */ }
}

function showQualityGateInChat(msg) {
  _showPromptInChat({
    dedupKey: _qualityGateKey,
    setDedupKey: (v) => { _qualityGateKey = v; },
    bubbleId: "qualityGateBubble",
    title: `📋 <strong>${msg}</strong>`,
    intro: "开启后将核查事实准确性并修正可疑内容。",
    buttons: [
      { text: "要，开启评估", cls: "btn-primary", handler: async (_btn, disableAll) => {
        disableAll();
        setStatus("正在进行质量评估…", "busy");
        addMsg("assistant", "✅ 已收到：开启质量评估。正在核查事实准确性…");
        await _submitPromptAction("quality", true, "");
      }},
      { text: "不要，跳过", handler: async (_btn, disableAll) => {
        disableAll();
        setStatus("报告生成完成", "done");
        addMsg("assistant", "✅ 已收到：跳过质量评估。报告生成完成！");
        await _submitPromptAction("quality", false, "");
      }},
    ],
  });
}

/* ── Final confirm (before render) ────────────────────────────────────────── */

let _finalConfirmKey = "";

async function loadFinalConfirmPreview(data) {
  const outline = await fetchText(`/result/${encodeURIComponent(S.taskId)}/outline.md`);
  const content = await fetchText(`/result/${encodeURIComponent(S.taskId)}/content.md`);
  S._outline = outline;
  S._content = content;
  showPreview();
  await loadVersions("outline");
  await loadVersions("content");
  const ov = S._versions.outline || [];
  const cv = S._versions.content || [];
  if (ov.length > 1 || cv.length > 1) {
    $("tabCompare").style.display = "";
  }
  showVersionTags(S.previewTab === "compare" ? (ov.length > 1 ? "outline" : "content") : S.previewTab);
}

function showFinalConfirmInChat(msg) {
  _showPromptInChat({
    dedupKey: _finalConfirmKey,
    setDedupKey: (v) => { _finalConfirmKey = v; },
    bubbleId: "finalConfirmBubble",
    title: `✅ <strong>${msg}</strong>`,
    intro: "您可以在右侧预览框中进行以下操作：",
    listHTML: `<ul style="margin:8px 0;padding-left:20px">
      <li><strong>版本对比</strong>：切换查看历史版本差异</li>
      <li><strong>段落重生成</strong>：点击段落旁的 🔄 按钮让 AI 重写该段</li>
      <li><strong>编辑段落</strong>：点击段落旁的 ✏️ 按钮直接编辑</li>
    </ul>
    <p>确认无误后点击下方按钮：</p>`,
    buttons: [
      { text: "✅ 最终确认，开始渲染", cls: "btn-primary", handler: async (_btn, disableAll) => {
        disableAll();
        setStatus("正在渲染最终报告…", "busy");
        addMsg("assistant", "✅ 已收到：最终确认。正在渲染最终报告…");
        await _submitPromptAction("final", true, "");
      }},
    ],
  });
}

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
      setStatus(LANG.feedback_satisfied_progress, "busy");
      // Restore preview visibility after satisfaction prompt hides it
      $("previewBox").style.display = "";
      $("compareBox").style.display = "none";
      $("compareToolbar").style.display = "none";
      if (S._content) showPreview();
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
    toast("提交失败: " + e.message, "error");
  }
}

/* ── Version management + compare ── loaded from /static/js/versions.js */

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
  r.textContent = role === "user" ? LANG.role_user : LANG.role_assistant;
  const b = document.createElement("div");
  b.className = "bubble";
  b.innerHTML = renderMD(text || "");
  wrap.appendChild(r);
  wrap.appendChild(b);
  log.appendChild(wrap);
  log.scrollTop = log.scrollHeight;

  // Make follow-up suggestions clickable for assistant messages
  if (role === "assistant") {
    setTimeout(makeFollowupsClickable, 10);
  }

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
  r.textContent = LANG.role_assistant;
  _streamBubble = document.createElement("div");
  _streamBubble.className = "bubble";
  _streamBubble.innerHTML = "";
  _streamWrap.appendChild(r);
  _streamWrap.appendChild(_streamBubble);
  log.appendChild(_streamWrap);
}

let _streamLastRender = 0;

function appendStreamToken(text) {
  ensureStreamBubble();
  _streamRaw += text;
  // Throttle MD rendering to at most once per 50ms during streaming.
  // For typical backends this collapses ~25 tokens into one DOM update.
  const now = Date.now();
  if (now - _streamLastRender >= 50) {
    _streamBubble.innerHTML = renderMD(_streamRaw);
    $("chatLog").scrollTop = $("chatLog").scrollHeight;
    _streamLastRender = now;
  }
}

function finalizeStreamBubble() {
  // Force final render to catch any tokens that arrived after the
  // last throttle window.
  if (_streamBubble) {
    _streamBubble.innerHTML = renderMD(_streamRaw);
    $("chatLog").scrollTop = $("chatLog").scrollHeight;
  }
  const text = _streamRaw;
  _streamBubble = null;
  _streamWrap = null;
  _streamRaw = "";
  _streamLastRender = 0;

  // ── Post-process: make follow-up suggestions clickable ──────────
  makeFollowupsClickable();

  return text;
}

// ── Clickable follow-up suggestions ──────────────────────────────
function makeFollowupsClickable() {
  // Find the last assistant bubble in the chat log
  const bubbles = document.querySelectorAll("#chatLog .msg.assistant .bubble");
  const lastBubble = bubbles[bubbles.length - 1];
  if (!lastBubble) return;

  // Pattern: ---\n💡 intro\n- question1\n- question2\n- question3
  // marked@12 outputs `<hr>` or `<hr />`; support both `<hr>` and `<hr/>`.
  const HR_MATCH = /<hr\s*\/?>\s*<p>💡[^<]*<\/p>\s*(<ul>[\s\S]*?<\/ul>)/i;
  const followupSection = lastBubble.innerHTML.match(HR_MATCH);
  if (!followupSection) return;

  const ulMatch = followupSection[1];
  // Parse list items
  const items = ulMatch.match(/<li>(.+?)<\/li>/g);
  if (!items || !items.length) return;

  // Build clickable button row
  let buttonsHtml = '<div class="followup-btns">';
  for (const item of items) {
    const text = item.replace(/<\/?li>/g, "").replace(/"/g, "&quot;");
    buttonsHtml += `<button class="followup-btn" onclick="pickFollowup(this)" data-q="${text}">${text}</button>`;
  }
  buttonsHtml += "</div>";

  // Replace the follow-up section with clickable buttons
  lastBubble.innerHTML = lastBubble.innerHTML.replace(
    HR_MATCH,
    '<hr><p>💡 点击下方按钮快速提问：</p>' + buttonsHtml
  );
}

function pickFollowup(btn) {
  const q = btn.dataset.q;
  if (!q) return;
  $("chatInput").value = q;
  $("chatInput").focus();
  // Dismiss all followup buttons after picking
  document.querySelectorAll(".followup-btn").forEach(b => {
    b.disabled = true;
    b.style.opacity = "0.5";
  });
}

// ── Clickable empty-state hint chips ───────────────────────────────
function bindChatHints() {
  const hints = document.querySelectorAll(".chat-hints span");
  const hintMap = {
    "直接说需求": "请根据这些材料生成一份报告",
    "切换知识库": "切换到默认知识库",
    "在知识库里查": "在知识库里查一下",
    "查看进度": "查看当前进度",
    "补充修改": "这份报告需要补充更多数据",
  };
  for (const el of hints) {
        el.classList.add("clickable");
    el.addEventListener("click", () => {
      const text = el.textContent.trim();
      $("chatInput").value = hintMap[text] || text;
      $("chatInput").focus();
    });
  }
}

function renderHistory(items) {
  clearChat();
  const log = $("chatLog");
  const frag = document.createDocumentFragment();
  for (const it of items || []) {
    if (!it || !it.role || !it.content) continue;
    const wrap = document.createElement("div");
    wrap.className = "msg " + it.role;
    const r = document.createElement("div");
    r.className = "role";
    r.textContent = it.role === "user" ? LANG.role_user : LANG.role_assistant;
    const b = document.createElement("div");
    b.className = "bubble";
    b.innerHTML = renderMD(it.content || "");
    wrap.appendChild(r);
    wrap.appendChild(b);
    frag.appendChild(wrap);
    // Track in history for sync
    S.history.push({ role: it.role, content: it.content || "" });
  }
  log.appendChild(frag);
  log.scrollTop = log.scrollHeight;
}

/* ── Embedding health indicator ─────────────────────────────────────── */

async function checkEmbedHealth() {
  const dot = $("kbEmbedStatus");
  if (!dot) return;
  dot.className = "status-dot";
  dot.title = "嵌入服务状态检测中…";
  try {
    const r = await fetch("/api/kb/health", { method: "POST" });
    const data = await r.json();
    if (data.ok) {
      dot.classList.add("ok");
      dot.title = `嵌入服务正常 (${data.model || "?"}, ${data.dim || "?"}维)`;
    } else {
      dot.classList.add("err");
      dot.title = `嵌入服务异常：${data.error || "未知错误"}。KB 上传和检索功能不可用。`;
    }
  } catch (e) {
    dot.classList.add("err");
    dot.title = "嵌入服务检测失败，KB 功能可能不可用。";
  }
}

/* ── Chat-bar KB controls ───────────────────────────────────────────── */

async function loadChatKbList() {
  try {
    const data = await apiGet("/api/kb/list");
    const sel = $("chatKbSelect");
    const currentVal = sel.value;
    sel.innerHTML = '<option value="">不使用KB</option>';
    for (const kb of (data.kbs || [])) {
      // Skip registry-only placeholder entries
      const opt = document.createElement("option");
      opt.value = kb;
      opt.textContent = kb;
      if (currentVal && currentVal === kb) opt.selected = true;
      sel.appendChild(opt);
    }
  } catch (e) { /* ignore */ }
}

async function onChatKbChange() {
  const kb = $("chatKbSelect").value;
  if (!kb) {
    try {
      await apiPost("/api/chat", {
        task_id: S.taskId || "lobby",
        action: { type: "kb_clear" }
      });
    } catch (e) { /* ignore */ }
    toast("\u5DF2\u53D6\u6D88\u77E5\u8BC6\u5E93\u9009\u62E9", "info", 2000);
    return;
  }
  try {
    await apiPost("/api/chat", {
      task_id: S.taskId || "lobby",
      action: { type: "kb_use", kb: kb }
    });
    toast("已切换到知识库：" + kb, "ok", 2000);
  } catch (e) {
    toast("切换失败：" + e.message, "error", 3000);
  }
}

async function onCreateKb() {
  const name = prompt("输入新知识库名称（中英文、数字、下划线）：");
  if (!name || !name.trim()) return;
  try {
    await apiPost("/api/kb/create", { kb: name.trim() });
    await loadChatKbList();
    $("chatKbSelect").value = name.trim();
    await onChatKbChange();
    toast("知识库已创建：" + name.trim(), "ok");
  } catch (e) {
    toast("创建失败：" + e.message, "error");
  }
}

async function onChatKbUpload() {
  const files = $("chatKbFiles").files;
  if (!files.length) return;
  const kb = $("chatKbSelect").value;
  if (!kb) { toast(LANG.toast_select_kb, "warn"); return; }
  const fd = new FormData();
  for (const f of files) fd.append("files", f);
  fd.append("kb", kb);

  // Clear any stale "thinking" status from previous chat attempts
  if (S._sending) { S._sending = false; $("btnSend").disabled = false; $("btnSend").textContent = LANG.btn_send; $("btnSend").classList.remove("btn-stop"); }
  setStatus(LANG.status_uploading, "busy");

  // Show one persistent toast that transitions from "uploading" to "done" or "failed"
  const progressEl = toast("正在上传 " + files.length + " 个文件到 " + kb + "...", "info", 60000);

  try {
    const data = await apiPost("/api/kb/upload", fd);
    // Transition the SAME toast from "uploading" to "done"
    if (progressEl && progressEl.parentNode) {
      progressEl.className = "toast ok";
      progressEl.querySelector("span").textContent = "已上传 " + (data.count || files.length) + " 个文件到知识库 " + kb;
      setTimeout(() => { if (progressEl.parentNode) progressEl.remove(); }, 4000);
    } else {
      toast("已上传 " + (data.count || files.length) + " 个文件到知识库 " + kb, "ok");
    }
    $("chatKbFiles").value = "";
    loadChatKbList().catch(() => {});
  } catch (e) {
    // Transition the SAME toast from "uploading" to "failed"
    if (progressEl && progressEl.parentNode) {
      progressEl.className = "toast error";
      progressEl.querySelector("span").textContent = LANG.toast_upload_failed + ': ' + e.message;
      setTimeout(() => { if (progressEl.parentNode) progressEl.remove(); }, 5000);
    } else {
      toast(LANG.toast_upload_failed + ': ' + e.message, "error", 5000);
    }
  } finally {
    setStatus("", "ok");
  }
}

/* ── Slash command autocomplete ──────────────────────────────────────── */

const SLASH_COMMANDS = [
  { cmd: "/kb ask",    desc: "KB智能问答",   usage: "/kb ask <问题>" },
  { cmd: "/kb use",    desc: "选择知识库",   usage: "/kb use <名称>" },
  { cmd: "/kb list",   desc: "列出知识库",   usage: "/kb list" },
  { cmd: "/kb clear",  desc: "取消知识库",   usage: "/kb clear" },
  { cmd: "/kb stats",  desc: "KB统计信息",   usage: "/kb stats [名称]" },
  { cmd: "/regen",     desc: "重新生成报告", usage: "/regen [doc|all|章节]" },
  { cmd: "/gen",       desc: "开始生成报告", usage: "/gen <需求描述>" },
  { cmd: "/prompt",    desc: "更新生成需求", usage: "/prompt <文本>" },
  { cmd: "/prompt!",   desc: "更新并重跑",   usage: "/prompt! <文本>" },
  { cmd: "/status",    desc: "查看任务状态", usage: "/status" },
  { cmd: "/pause",     desc: "暂停任务",     usage: "/pause" },
  { cmd: "/resume",    desc: "继续任务",     usage: "/resume" },
  { cmd: "/cancel",    desc: "取消任务",     usage: "/cancel" },
  { cmd: "/help",      desc: "查看所有指令", usage: "/help" },
  { cmd: "/files",     desc: "查看上传文件", usage: "/files" },
  { cmd: "/templates", desc: "查看可用模板", usage: "/templates" },
  { cmd: "/append",    desc: "追加文件说明", usage: "/append" },
];

function looksLikeChatGenerationRequest(msg) {
  const text = (msg || "").trim();
  if (!text || text.startsWith("/")) return false;
  if (text.length < 8) return false;
  const patterns = [
    /请.*(?:生成|写|整理|输出|撰写)/,
    /帮我(?:生成|写|做|整理)/,
    /根据.*材料.*(?:生成|写|整理|输出)/,
    /(?:生成|写一份|做一份|整理成|输出一份).*(?:报告|文档|总结|方案|综述|分析|汇报|稿子)/,
  ];
  return patterns.some((re) => re.test(text));
}

async function triggerChatGeneration(prompt) {
  const text = (prompt || "").trim();
  if (!text) return false;

  if (!S.taskId) {
    const files = $("files").files;
    if (!files.length) return false;
    addMsg("assistant", "收到，我会按这个需求开始生成：\n\n" + text + "\n\n正在上传文件并启动生成…");
    $("userPrompt").value = text;
    S._sending = false;
    await uploadAndGenerate();
    return true;
  }

  addMsg("assistant", "收到，我会按这个需求开始生成：\n\n" + text + "\n\n正在启动报告生成…");
  S.generating = true;
  S._lastOperation = "generate";
  $("btnUpload").disabled = true;
  $("btnAppend").disabled = true;
  setStatus("启动生成中…", "busy");
  S._sending = false;
  try {
    const data = await apiPost("/api/gen", { task_id: S.taskId, prompt: text });
    if (data.ok) {
      addMsg("assistant", "任务已启动，任务ID：" + data.task_id + "\n\n等待报告生成完成后会在预览区显示。");
      startPolling();
    }
  } catch (e) {
    addMsg("assistant", "启动失败：" + e.message);
    S.generating = false;
    $("btnUpload").disabled = false;
    $("btnAppend").disabled = false;
  }
  return true;
}

function onChatInput() {
  const val = $("chatInput").value;
  if (!val.startsWith("/")) { hideSlashPanel(); return; }

  const query = val.slice(1).toLowerCase();
  const matches = SLASH_COMMANDS.filter(c => c.cmd.includes(query) || c.desc.includes(query));
  if (!matches.length) { hideSlashPanel(); return; }

  const panel = $("slashPanel");
  panel.innerHTML = matches.map((c, i) =>
    `<div class="slash-item${i === 0 ? ' active' : ''}" data-cmd="${esc(c.usage || c.cmd)}" data-idx="${i}">
      <span class="cmd">${esc(c.cmd)}</span>
      <span class="desc">${esc(c.desc)}</span>
    </div>`
  ).join("");
  panel.style.display = "block";

  // Click to select
  panel.querySelectorAll(".slash-item").forEach(el => {
    el.addEventListener("click", () => {
      $("chatInput").value = el.dataset.cmd + " ";
      hideSlashPanel();
      $("chatInput").focus();
    });
  });
}

function hideSlashPanel() { $("slashPanel").style.display = "none"; }
function slashPanelVisible() { return $("slashPanel").style.display === "block"; }

function moveSlashSelection(dir) {
  if (!slashPanelVisible()) return;
  const items = $("slashPanel").querySelectorAll(".slash-item");
  const active = $("slashPanel").querySelector(".slash-item.active");
  let idx = active ? parseInt(active.dataset.idx) : -1;
  idx += dir;
  if (idx < 0) idx = items.length - 1;
  if (idx >= items.length) idx = 0;
  items.forEach(el => el.classList.remove("active"));
  items[idx].classList.add("active");
  items[idx].scrollIntoView({ block: "nearest" });
}

function selectActiveSlashItem() {
  const active = $("slashPanel").querySelector(".slash-item.active");
  if (active) {
    $("chatInput").value = active.dataset.cmd + " ";
    hideSlashPanel();
    $("chatInput").focus();
  }
}

/* ── Send chat ──────────────────────────────────────────────────────── */

async function sendChat() {
  if (S._sending) return;
  const msg = $("chatInput").value.trim();
  if (!msg) return;
  if (!S.taskId) setStatus(LANG.chat_general_mode, "ok");

  S._sending = true;
  syncUIState();
  $("chatInput").value = "";
  addMsg("user", msg);

  // ── Generation request: support both /gen and natural language ──
  const genMatch = msg.match(/^\/(?:gen|generate|生成)(?:\s+(.+))?/i);
  if (genMatch) {
    const prompt = (genMatch[1] || "").trim();
    if (!prompt) {
      addMsg("assistant", "请提供需求描述，格式：\n\n<code>/gen 需求描述</code>\n\n例如：\n• /gen 帮我写一份RAG技术综述\n• /gen 根据材料生成市场分析报告\n• /gen 写一份项目总结，重点突出技术方案和性能评估");
      S._sending = false;
      return;
    }
    // Auto-create task if files selected but no task yet
    if (!S.taskId) {
      const files = $("files").files;
      if (!files.length) {
        addMsg("assistant", "请先在左侧面板选择要上传的材料文件。选好以后，你既可以输入 /gen，也可以直接用自然语言告诉我生成需求。");
        S._sending = false;
        return;
      }
      await triggerChatGeneration(prompt);
      return;
    }
    await triggerChatGeneration(prompt);
    return;
  }

  if (looksLikeChatGenerationRequest(msg) && (S.taskId || ($("files").files || []).length > 0)) {
    await triggerChatGeneration(msg);
    return;
  }

  // If waiting for section regen feedback, route to section regen
  if (S.uiState === UI_STATES.WAITING_REGEN_FEEDBACK) {
    const info = S._waitingRegenFeedback;
    S._waitingRegenFeedback = null;
    syncUIState();

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
      setStatus(e.message, "error");
      toast("重生成失败: " + e.message, "error");
      S.generating = false;
      $("btnUpload").disabled = false;
      $("btnAppend").disabled = false;
    }
    S._sending = false;
    $("btnSend").disabled = false;
    return;
  }

  // If waiting for satisfaction feedback, route to satisfaction
  if (S.uiState === UI_STATES.WAITING_SATISFACTION_FEEDBACK) {
    S._waitingFeedback = false;
    syncUIState();
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
  if (S.uiState === UI_STATES.WAITING_REDO) {
    S._waitingRedo = false;
    syncUIState();
    $("btnSend").disabled = true;
    setStatus("重新生成中…", "busy");
    await doRedoVersion(S._redoType, S._redoBaseVersion, msg);
    S._sending = false;
    syncUIState();
    $("btnSend").disabled = false;
    return;
  }

  S.history.push({ role: "user", content: msg });
  setStatus("思考中…", "busy");

  // ── Thinking indicator ──────────────────────────────────────────
  let thinkingEl = null;
  const chatBox = $("chatLog");
  try {
    thinkingEl = document.createElement("div");
    thinkingEl.className = "msg-thinking";
    thinkingEl.innerHTML = '<span>思考中</span><span class="dots"><span></span><span></span><span></span></span>';
    if (chatBox) {
      chatBox.appendChild(thinkingEl);
      chatBox.scrollTop = chatBox.scrollHeight;
    }
  } catch (e) { console.error("showThinking failed:", e); }

  // ── Cancel / stop button ─────────────────────────────────────────
  const btn = $("btnSend");
  const origText = btn.textContent;
  btn.textContent = LANG.btn_stop;
  btn.classList.add("btn-stop");
  btn.disabled = false;

  const abortCtrl = new AbortController();
  let userStopped = false;
  let timeoutFired = false;
  const chatTimeout = setTimeout(() => {
    timeoutFired = true;
    abortCtrl.abort();
  }, 60000);
  function stopStream() {
    userStopped = true;
    abortCtrl.abort();
    if (thinkingEl && thinkingEl.parentNode) thinkingEl.remove();
    thinkingEl = null;
    btn.textContent = origText;
    btn.classList.remove("btn-stop");
    btn.disabled = false;
    btn.onclick = sendChat;
    S._sending = false;
    setStatus("", "ok");
  }
  btn.onclick = stopStream;

  try {
    const r = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task_id: S.taskId || "lobby", message: msg, history: S.history }),
      signal: abortCtrl.signal,
    });
    if (!r.ok) {
      const d = await r.json().catch(() => ({}));
      const err = d.error || d.detail || `对话失败 (${r.status})`;
      addMsg("assistant", "❌ " + err);
      setStatus(err, "error");
      if (r.status >= 500) toast("对话服务异常: " + err, "error", 6000);
      return;
    }

    const contentType = r.headers.get("content-type") || "";
    // Remove thinking indicator as soon as we have a response
    if (thinkingEl && thinkingEl.parentNode) { thinkingEl.remove(); thinkingEl = null; }

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
      }
      return;
    }

    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
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
          break;
        }
        try {
          const obj = JSON.parse(payload);
          if (obj.token) appendStreamToken(obj.token);
          else if (obj.status === "streaming") { setStatus("输出中…", "ok"); ensureStreamBubble(); }
          else if (obj.error) { addMsg("assistant", "❌ " + obj.error); }
        } catch {}
      }
    }
    await pollStatus();
  } catch (e) {
    if (e.name === "AbortError") {
      if (timeoutFired && !userStopped) {
        const timeoutMsg = "对话响应超时（60s），请检查后端服务或模型连接后重试。";
        addMsg("assistant", "❌ " + timeoutMsg);
        setStatus("对话响应超时", "error");
        toast(timeoutMsg, "warning", 6000);
      }
    } else if (e.name === "TypeError" && e.message.includes("fetch")) {
      const netMsg = "网络连接失败，请确认服务已启动。";
      addMsg("assistant", "❌ " + netMsg);
      setStatus("连接失败", "error");
      toast(netMsg, "error", 8000);
    } else {
      console.error("sendChat error:", e);
      addMsg("assistant", "❌ 请求异常: " + e.message);
      toast("对话异常: " + e.message, "error", 5000);
    }
  } finally {
    clearTimeout(chatTimeout);
    if (thinkingEl && thinkingEl.parentNode) thinkingEl.remove();
    thinkingEl = null;
    btn.textContent = origText;
    btn.classList.remove("btn-stop");
    btn.disabled = false;
    btn.onclick = sendChat;
    S._sending = false;
    if (!timeoutFired || userStopped) setStatus("", "ok");
  }
}
