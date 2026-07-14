/* ── Version management ────────────────────────────────────────────── */

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
    tag.title = v.feedback ? LANG.label_feedback + ': ' + v.feedback : LANG.label_version + ' ' + vn + (v.selected ? ' (' + LANG.label_selected + ')' : '');

    const vLabel = document.createElement("span");
    vLabel.textContent = `V${vn}`;
    vLabel.className = "clickable";
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

    if (versions.length > 1) {
      const delBtn = document.createElement("span");
      delBtn.className = "version-del-btn";
      delBtn.textContent = "×";
      delBtn.title = LANG.btn_delete + ' ' + LANG.label_version;
      delBtn.onclick = async (e) => {
        e.stopPropagation();
        if (!confirm(LANG.ver_confirm_delete.replace('{ver}', vn))) return;
        try {
          const data = await apiPost("/api/versions/delete", {
            task_id: S.taskId, type: type, version: vn,
          });
          if (data && data.ok) {
            toast(LANG.ver_toast_deleted.replace('{ver}', vn), "ok", 2000);
            await loadVersions(type);
            showVersionTags(type);
            showPreview();
          } else {
            toast(data.message || LANG.toast_delete_failed, "error");
          }
        } catch (e) {
          toast(LANG.toast_delete_failed + ': ' + e.message, "error");
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

/* ── Cross-version comparison ─────────────────────────────────────── */

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
    $("diffStats").textContent = LANG.ver_only_one;
  } else {
    $("diffStats").textContent = LANG.ver_no_versions;
  }
}

/* ── Line diff algorithm (LCS-based) ───────────────────────────────── */

function computeLineDiff(oldText, newText) {
  const oldLines = (oldText || "").split("\n");
  const newLines = (newText || "").split("\n");
  const MAX = 400;
  if (oldLines.length > MAX || newLines.length > MAX) {
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

  for (const item of stack.reverse()) {
    if (item.type === "same") {
      oldResult.push({ type: "same", text: item.text });
      newResult.push({ type: "same", text: item.text });
    } else if (item.type === "removed") {
      oldResult.push({ type: "removed", text: item.text });
    } else {
      newResult.push({ type: "added", text: item.text });
    }
  }

  return { old: oldResult, new: newResult };
}

/* ── Diff-aware markdown rendering ─────────────────────────────────── */

function renderDiffContent(rawText, lineTypes, side) {
  const blocks = [];
  let currentType = null;
  let currentLines = [];

  for (const dl of lineTypes) {
    const show = (side === "old")
      ? (dl.type === "added" ? "skip" : dl.type)
      : (dl.type === "removed" ? "skip" : dl.type);

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
  return html || '<p class="empty-text">' + LANG.ver_no_content + '</p>';
}

function updateDiffCache() {
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
  if (S.previewTab === "compare") {
    const sel = $("compareTypeSelect");
    if (sel && sel.value) return sel.value;
    return S._compareSourceTab === "outline" ? "outline" : "content";
  }
  return S.previewTab === "outline" ? "outline" : "content";
}

function updateCompareFromSelection(type) {
  populateCompareSelectors();
  refreshBothCompareSides();
}

function showCompare() {
  const ov = S._versions.outline || [];
  const cv = S._versions.content || [];
  let bestType = S._compareSourceTab || "outline";
  if (ov.length > cv.length && ov.length >= 2) bestType = "outline";
  else if (cv.length > ov.length && cv.length >= 2) bestType = "content";
  else if (ov.length < 2 && cv.length >= 2) bestType = "content";
  else if (cv.length < 2 && ov.length >= 2) bestType = "outline";

  const typeSel = $("compareTypeSelect");
  if (typeSel) {
    typeSel.value = bestType;
    typeSel.querySelectorAll("option").forEach(opt => {
      const cnt = opt.value === "outline" ? ov.length : cv.length;
      opt.textContent = opt.value === "outline" ? (LANG.label_outline + ' (' + cnt + LANG.label_version + ')') : (LANG.label_report + ' (' + cnt + LANG.label_version + ')');
    });
  }

  const versions = S._versions[bestType] || [];
  if (versions.length < 1) {
    toast(LANG.ver_no_compare, "info", 3000);
    return;
  }

  showVersionTags(bestType);
  populateCompareSelectors();

  $("compareToolbar").style.display = "flex";
  $("showDiff").checked = S._showDiff;
  $("showDiff").onchange = onDiffToggle;
  $("diffStats").style.display = S._showDiff ? "" : "none";

  $("compareVerLeft").onchange = () => refreshBothCompareSides();
  $("compareVerRight").onchange = () => refreshBothCompareSides();

  if (typeSel) typeSel.onchange = () => { populateCompareSelectors(); refreshBothCompareSides(); showVersionTags(getCompareType()); };

  refreshBothCompareSides();

  $("btnAcceptLeft").onclick = () => acceptVersionFromCompare("left");
  $("btnAcceptRight").onclick = () => acceptVersionFromCompare("right");
  $("btnRedoLeft").onclick = () => redoVersionFromCompare("left");
  $("btnRedoRight").onclick = () => redoVersionFromCompare("right");

  $("compareBox").style.display = "grid";
  $("previewBox").style.display = "none";
}

function refreshBothCompareSides() {
  const type = getCompareType();
  const versions = S._versions[type] || [];
  const lv = parseInt($("compareVerLeft").value) || 1;
  const rv = parseInt($("compareVerRight").value) || 1;

  const lVer = versions.find(v => v.version === lv);
  const rVer = versions.find(v => v.version === rv);
  const lbl = (v, vn) => v ? `V${vn}${v.feedback ? " — " + v.feedback.slice(0, 30) : ""}` : `V${vn}`;
  $("compareLabelLeft").textContent = lbl(lVer, lv);
  $("compareLabelRight").textContent = lbl(rVer, rv);

  S._cachedDiff = null;

  const oldText = lVer ? (lVer.content || "") : "";
  const newText = rVer ? (rVer.content || "") : "";

  if (S._showDiff && lv !== rv) {
    const cached = updateDiffCache();
    const diff = cached.diff;
    $("diffStats").textContent = '+' + (diff.addedCount || 0) + LANG.label_lines + ' \u2212' + (diff.removedCount || 0) + LANG.label_lines;
    $("compareOld").innerHTML = renderDiffContent(cached.oldText, diff.old, "old");
    $("compareNew").innerHTML = renderDiffContent(cached.newText, diff.new, "new");
  } else {
    $("diffStats").textContent = lv === rv ? LANG.ver_same_version : "";
    $("compareOld").innerHTML = renderMD(oldText || LANG.ver_no_content);
    $("compareNew").innerHTML = renderMD(newText || LANG.ver_no_content);
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
  syncUIState();

  const label = type === "outline" ? LANG.label_outline : LANG.label_report;
  const versions = S._versions[type] || [];
  const baseVer = versions.find(v => v.version === versionNum);
  const labelExtra = baseVer && baseVer.feedback ? `（${baseVer.feedback.slice(0, 40)}）` : "";

  addActionMsg(
    `<p>🔄 <strong>基于 ${label} V${versionNum} 重新生成</strong> ${labelExtra}</p>
     <p>请输入<strong>改进意见</strong>，例如：请增加数据分析章节、减少技术术语。</p>
     <p class="text-hint">在聊天框输入后发送</p>`,
    [
      {
        text: LANG.feedback_skip, cls: "btn btn-outline btn-sm",
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
  const label = type === "outline" ? LANG.label_outline : LANG.label_report;
  addMsg("assistant", '\u2705 ' + LANG.label_version + ' ' + label + ' V' + versionNum + ' ' + LANG.btn_regen + (fb ? '\n\n' + LANG.label_feedback + '\uff1a' + fb : ''));

  S._waitingRedo = false;
  syncUIState();

  try {
    await apiPost("/api/versions/redo", {
      task_id: S.taskId, type: type, base_version: versionNum, feedback: fb,
    });
    setStatus(LANG.status_regenerating, "busy");
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
    toast(LANG.toast_regen_failed + ': ' + e.message, "error");
    S.generating = false;
    $("btnUpload").disabled = false;
    $("btnAppend").disabled = false;
  }
}

async function selectVersion(type, versionNum) {
  if (!S.taskId) return;
  try {
    await apiPost("/api/versions/select", {
      task_id: S.taskId, type: type, version: versionNum,
    });
    S._selectedVersion[type] = versionNum;
    const versions = S._versions[type] || [];
    for (const v of versions) {
      v.selected = (v.version === versionNum);
    }
    showVersionTags(type);
    toast(LANG.ver_toast_selected.replace('{ver}', versionNum), "ok", 3000);

    const found = versions.find(v => v.version === versionNum);
    if (found && found.content) {
      if (type === "outline") {
        S._outline = found.content;
      } else {
        S._content = found.content;
      }
    }
    S.previewTab = type;
    const container = $("previewTabs");
    container.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    const activeTab = container.querySelector(`[data-tab="${type}"]`);
    if (activeTab) activeTab.classList.add("active");
    showPreview();
    $("compareBox").style.display = "none";
    $("compareToolbar").style.display = "none";
    $("previewBox").style.display = "";
  } catch (e) {
    toast(LANG.ver_toast_select_failed + e.message, "error");
  }
}
