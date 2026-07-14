/* ── KB + Template + Init ── loaded from /static/js/features.js */

/* ── Downloads display ───────────────────────────────────────────── */

function setDownloads(data) {
  const list = $("downloadList");
  list.innerHTML = "";
  var docxUrl = "", pdfUrl = "";
  var files = (data && data.files) || [];
  for (var i = 0; i < files.length; i++) {
    var f = files[i];
    var li = document.createElement("li");
    var a = document.createElement("a");
    a.href = f.url;
    a.textContent = f.name + " (" + Math.round((f.size || 0) / 1024) + " KB)";
    a.target = "_blank";
    li.appendChild(a);
    list.appendChild(li);
    if (!docxUrl && /\.docx$/i.test(f.name)) docxUrl = f.url;
    if (!pdfUrl && /\.pdf$/i.test(f.name)) pdfUrl = f.url;
  }
  if (!list.children.length) list.innerHTML = '<li style="color:var(--c-text3);font-size:12px">暂无下载文件</li>';
  var btnDocx = $("btnDownloadDocx");
  var btnPdf = $("btnDownloadPdf");
  if (btnDocx) {
    if (docxUrl) { btnDocx.href = docxUrl; btnDocx.style.display = ""; }
    else btnDocx.style.display = "none";
  }
  if (btnPdf) {
    if (pdfUrl) { btnPdf.href = pdfUrl; btnPdf.style.display = ""; }
    else btnPdf.style.display = "none";
  }
}

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
  list.innerHTML = '<span class="empty-text">' + LANG.kb_loading_docs + '</span>';
  $("btnKbDocLoad").disabled = true;
  try {
    const data = await apiGet(`/api/kb/docs?kb=${encodeURIComponent(kbName)}`);
    const docs = Array.isArray(data.docs) ? data.docs : [];
    list.innerHTML = "";
    if (!docs.length) {
      list.innerHTML = '<span class="empty-text">' + LANG.kb_no_docs + '</span>';
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
      const delBtn = document.createElement("button");
      delBtn.className = "btn btn-sm kb-del-btn";
      delBtn.textContent = "删除";
      delBtn.addEventListener("click", (e) => { e.preventDefault(); e.stopPropagation(); deleteKbDoc(did); });
      item.appendChild(delBtn);
      list.appendChild(item);
    }
    updateKbDocCount();
  } catch (e) {
    list.innerHTML = '<span style="color:var(--c-red);font-size:12px">' + LANG.kb_loading_failed + ': ' + esc(e.message) + '</span>';
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
  setKbStatus(LANG.status_loading);
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
    refreshKbDocList();
  } catch (e) {
    setKbStatus(LANG.kb_loading_failed + ': ' + e.message);
  }
}

async function refreshKbDocList() {
  const box = $("kbDocListView");
  if (!box) return;
  const kb = ($("kbSelect").value || "").trim() || S.kb;
  if (!kb) { box.innerHTML = ""; return; }
  try {
    const data = await apiGet(`/api/kb/docs?kb=${encodeURIComponent(kb)}`);
    const docs = Array.isArray(data.docs) ? data.docs : [];
    if (!docs.length) {
      box.innerHTML = "<div class='kb-doc-empty'>知识库为空，请先上传文档</div>";
      return;
    }
    box.innerHTML = docs.slice(0, 30).map(d => {
      const name = esc(d.title || d.doc_id || "");
      const cnt = d.chunk_count || 0;
      return `<div class="kb-doc-item small">
        <span class="doc-name">${name}</span>
        <span class="doc-meta">${cnt} chunks</span>
      </div>`;
    }).join("");
  } catch { box.innerHTML = "<div class='kb-doc-empty'>" + LANG.kb_loading_failed + "</div>"; }
}

function useKbFromUI() {
  const typed = ($("kbName").value || "").trim();
  const chosen = ($("kbSelect").value || "").trim();
  setKbCurrent(typed || chosen || "default");
  setKbStatus("");
}

async function deleteKb() {
  const kb = ($("kbSelect").value || "").trim();
  if (!kb) return toast(LANG.toast_select_kb, "warn");
  if (!confirm(`确定要删除整个知识库 "${kb}" 及其所有文档吗？此操作不可撤销。`)) return;
  try {
    const resp = await apiPost("/api/kb/delete", { kb });
    if (resp.ok) {
      toast(`知识库 "${kb}" 已删除`, "info");
      loadKbList();
      loadKbDocs();
    } else {
      toast(LANG.toast_delete_failed + ': ' + (resp.error || "\u672A\u77E5\u9519\u8BEF"), "error");
    }
  } catch (e) {
    toast(LANG.toast_delete_failed + ': ' + e.message, "error");
  }
}

async function deleteKbDoc(docId) {
  const kb = ($("kbSelect").value || "").trim() || S.kb;
  if (!kb || !docId) return;
  if (!confirm(`确定要从 "${kb}" 中删除文档 "${docId}" 吗？`)) return;
  try {
    const resp = await apiPost("/api/kb/delete", { kb, doc_id: docId });
    if (resp.ok) {
      toast(`文档已删除`, "info");
      loadKbDocs();
    } else {
      toast(LANG.toast_delete_failed + ': ' + (resp.error || "\u672A\u77E5\u9519\u8BEF"), "error");
    }
  } catch (e) {
    toast(LANG.toast_delete_failed + ': ' + e.message, "error");
  }
}

function setKbStatus(s) { const el = $("kbStatus"); if (el) el.textContent = s || ""; }

async function kbUpload() {
  const files = $("kbFiles").files;
  if (!files.length) { setKbStatus(LANG.toast_select_file); return; }
  useKbFromUI();
  const fd = new FormData();
  fd.append("kb", S.kb);
  const dt = ($("kbDocType").value || "").trim();
  if (dt) fd.append("doc_type", dt);
  for (const f of files) fd.append("files", f, f.name);

  setKbStatus(LANG.status_uploading);
  $("btnKbUpload").disabled = true;
  try {
    const data = await apiPost("/api/kb/upload", fd);
    const results = Array.isArray(data.results) ? data.results : [];
    const ok = results.filter(r => r && r.ok).length;
    const fail = results.filter(r => r && !r.ok).length;
    setKbStatus(LANG.status_done + '\uff1a' + LANG.status_success + ' ' + ok + '\uff0c' + LANG.status_failed + ' ' + fail);
    $("kbAnswer").textContent = JSON.stringify(data, null, 2);
    $("kbCitations").textContent = "";
    await loadKbList();
    if (S._refreshRetrievalKbList) S._refreshRetrievalKbList();
    $("kbSelect").value = S.kb;
  } catch (e) {
    setKbStatus(LANG.status_failed + ': ' + e.message);
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

function _parseCitationMeta(text) {
  const meta = { abstract: "", keywords: [], doi: "", url: "",
    boilerplate: [], citation: "", bodyLines: [] };
  const lines = text.split("\n").map(l => l.trim()).filter(Boolean);
  // Deduplicate
  const deduped = []; const seen60 = new Set();
  for (const l of lines) { const k = l.slice(0, 60); if (!seen60.has(k)) { seen60.add(k); deduped.push(l); } }

  const bpTerms = ["录用定稿", "排版定稿", "整期汇编", "网络首发", "出版确认",
    "出版管理条例", "期刊出版管理规定", "首发视为正式出版", "编辑部工作流程",
    "中国学术期刊", "ISSN", "CN 11-", "文献标志码", "中图分类号"];
  let current = "body";
  const buckets = { abstract: [], keywords: [], body: [] };

  for (const line of deduped) {
    if (/^摘要[：:]/.test(line)) { current = "abstract"; buckets.abstract.push(line.replace(/^摘要[：:]\s*/, "")); continue; }
    if (/^关键词[：:]/.test(line)) { current = "keywords"; buckets.keywords.push(line.replace(/^关键词[：:]\s*/, "")); continue; }
    if (/^作者简介|^基金项目/.test(line)) { continue; }
    if (/^引用格式[：:]/.test(line)) { meta.citation = line.replace(/^引用格式[：:]\s*/, ""); continue; }
    if (bpTerms.some(t => line.includes(t))) { meta.boilerplate.push(line); continue; }
    const doiM = line.match(/(?:doi|DOI)[：:]\s*(\S+)/); if (doiM) { meta.doi = doiM[1]; continue; }
    const urlM = line.match(/(https?:\/\/\S+)/); if (urlM) { meta.url = urlM[1]; continue; }
    buckets[current].push(line);
  }

  meta.abstract = buckets.abstract.join(" ").trim();
  meta.keywords = buckets.keywords.join(" ").split(/[；;，,]/).map(k => k.trim()).filter(k => k.length > 1 && k.length < 30);
  meta.bodyLines = buckets.body.filter(l => !bpTerms.some(t => l.includes(t)));
  return meta;
}

function renderKbCitations(items) {
  const box = $("kbCitations");
  if (!box) return;
  const cits = Array.isArray(items) ? items : [];
  if (!cits.length) { box.innerHTML = "<div class='citation-empty'>（无引用）</div>"; return; }

  // Group by document (all chunks, not just adjacent)
  const docMap = new Map();
  for (const c of cits) {
    if (!c) continue;
    const key = c.doc_id || c.doc_name || "";
    const entry = docMap.get(key);
    if (entry) { entry.snippets.push(c.snippet || ""); }
    else { docMap.set(key, { key, docName: c.doc_name || c.doc_id || "?", snippets: [c.snippet || ""] }); }
  }
  const groups = [...docMap.values()];

  // Merge + parse each group
  const cards = [];
  for (const g of groups) {
    const lines = g.snippets.join("\n").split("\n");
    const seen = new Set(); const uniq = [];
    for (const l of lines) { const t = l.trim(); if (!t) continue; const k = t.slice(0, 60); if (!seen.has(k)) { seen.add(k); uniq.push(t); } }
    const merged = uniq.join("\n");
    const meta = _parseCitationMeta(merged);
    const docLabel = esc(g.docName.replace(/\.(pdf|docx?|md|txt)$/i, ""));
    // Best title: use docLabel, stripped clean
    let title = docLabel
      .replace(/\.(pdf|docx?|md|txt)$/i, "")
      .replace(/\s*\/\s*\S.*$/, "");  // strip trailing path/date fragments

    let html = `<div class="cit-card">
      <div class="cit-title">${esc(title)}</div>`;

    // Abstract — main content
    const ab = meta.abstract || meta.bodyLines.slice(0, 8).join(" ");
    if (ab) {
      const maxLen = 300;
      const show = ab.length > maxLen ? ab.slice(0, maxLen) + "…" : ab;
      html += `<div class="cit-abstract">${renderMD(show)}</div>`;
    }

    // Keywords as tags
    if (meta.keywords.length) {
      html += `<div class="cit-keywords">${meta.keywords.map(k => `<span class="cit-tag">${esc(k)}</span>`).join("")}</div>`;
    }

    html += `</div>`;
    cards.push(html);
  }

  box.innerHTML = cards.join("\n");
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
    // Template dir may not exist yet on first run — not an error
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
    toast("加载失败: " + e.message, "error");
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
    toast("保存失败: " + e.message, "error");
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
    toast(LANG.toast_delete_failed + ': ' + e.message, "error");
  }
}

async function useTemplateOnTask() {
  const name = S._tplEditor.currentName || $("tplEditorSelect").value;
  if (name) useTemplateForTask(name);
  else toast(LANG.toast_select_file, "info");
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
  $("tplEditorName").value = "";
  $("tplEditorTextarea").value = "";
  $("tplEditorSelect").value = "";
  S._tplEditor.currentName = "";
  S._tplEditor.content = "";
  $("tplPreviewBox").innerHTML = "";
  refreshTplPreview();
  S._tplEditor.dirty = false;
}

async function loadVariables() {
  try {
    const data = await apiGet("/api/template/variables");
    S._tplEditor.variables = data.system_variables || [];
    renderVarList(data.system_variables || [], data.section_variables_note || "");
  } catch (e) {
    $("tplVarList").innerHTML = '<span class="text-hint-sm">加载失败</span>';
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
    toast("使用失败: " + e.message, "error");
  }
}

/* ── Task list ──────────────────────────────────────────────────── */

async function loadTaskList() {
  try {
    const data = await apiGet("/api/tasks");
    const tasks = Array.isArray(data.tasks) ? data.tasks : [];
    const sel = $("taskListSelect");
    if (!sel) return;
    sel.innerHTML = '<option value="">— 历史任务 —</option>';
    for (const t of tasks) {
      const opt = document.createElement("option");
      opt.value = t.task_id;
      const label = t.task_id + (t.status ? " (" + t.status + ")" : "");
      opt.textContent = label;
      sel.appendChild(opt);
    }
    sel.style.display = tasks.length ? "" : "none";
    $("btnRefreshTasks").style.display = tasks.length ? "" : "none";
  } catch (e) {
    // task list is non-critical
  }
}

async function loadTaskById() {
  const tid = $("taskIdInput").value.trim();
  if (!tid) { toast(LANG.toast_need_task, "error"); return; }
  setTaskBadge(tid);
  $("previewBox").textContent = "";
  setDownloads({ files: [] });
  setStatus(LANG.status_loading, "busy");
  stopPolling();
  $("evalCard").style.display = "none";
  $("warningsCard").style.display = "none";
  S._evalData = null;
  try {
    const data = await apiGet("/api/tasks?task_id=" + encodeURIComponent(tid));
    if (data.status === "finished" && data.outline) S._outline = data.outline;
    if (data.status === "finished" && data.content) S._content = data.content;
    setDownloads(data.downloads || {});
    showPreview();
    await loadVersions("outline");
    await loadVersions("content");
    const ov = S._versions.outline || [];
    const cv = S._versions.content || [];
    if (ov.length > 1 || cv.length > 1) $("tabCompare").style.display = "";
    S.generating = false;
    S._satisfactionKey = "";
    S._contentStreaming = false;
    setStatus(LANG.status_done, "ok");
  } catch (e) {
    setStatus(LANG.status_failed, "error");
    toast("加载任务失败: " + e.message, "error");
    S.generating = false;
  }
}

function onTaskListSelect() {
  const sel = $("taskListSelect");
  if (!sel) return;
  const tid = sel.value;
  if (tid) {
    $("taskIdInput").value = tid;
    loadTaskById();
  }
}

async function quickUseTemplate() {
  const name = $("tplQuickSelect").value;
  if (!name) { toast("请选择一个模板", "info"); return; }
  // Remember the selection — will be applied when task starts
  S._pendingTemplate = name;
  toast(`已选择模板「${name}」，将在生成报告时自动应用。`, "ok", 3000);
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
  if (r.status === 204) return {}; // No content on successful delete
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

  // Eval tabs removed — unified card replaces old eval system

  // Upload
  $("btnUpload").addEventListener("click", uploadAndGenerate);
  $("btnAppend").addEventListener("click", appendToTask);

  // Chat
  $("btnSend").addEventListener("click", sendChat);
  bindChatHints();
  $("chatInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      if (slashPanelVisible()) { e.preventDefault(); selectActiveSlashItem(); return; }
      sendChat();
      return;
    }
    if (e.key === "ArrowDown") { e.preventDefault(); moveSlashSelection(1); return; }
    if (e.key === "ArrowUp") { e.preventDefault(); moveSlashSelection(-1); return; }
    if (e.key === "Escape") { hideSlashPanel(); return; }
  });
  $("chatInput").addEventListener("input", onChatInput);

  // Chat-bar KB controls
  loadChatKbList().catch(() => {});
  checkEmbedHealth().catch(() => {});
  $("chatKbSelect").addEventListener("change", onChatKbChange);
  $("btnChatKbCreate").addEventListener("click", onCreateKb);
  $("btnChatKbUpload").addEventListener("click", () => {
    const kb = $("chatKbSelect").value;
    if (!kb) { toast("请先选择或创建一个知识库", "warn"); return; }
    $("chatKbFiles").click();
  });
  $("chatKbFiles").addEventListener("change", onChatKbUpload);

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
  $("btnKbDelete").addEventListener("click", deleteKb);
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

  // Retrieval KB selector
  const loadRetrievalKbList = async () => {
    try {
      const data = await apiGet("/api/kb/list");
      const items = (data && Array.isArray(data.kbs)) ? data.kbs : ["default"];
      const sel = $("retrievalKbSelect");
      sel.innerHTML = "";
      const emptyOpt = document.createElement("option");
      emptyOpt.value = "";
      emptyOpt.textContent = "— 不使用知识库检索 —";
      sel.appendChild(emptyOpt);
      for (const name of items) {
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name;
        sel.appendChild(opt);
      }
    } catch {}
  };
  S._refreshRetrievalKbList = loadRetrievalKbList;
  $("btnRefreshRetrievalKb").addEventListener("click", loadRetrievalKbList);
  $("retrievalKbSelect").addEventListener("change", () => {
    const v = $("retrievalKbSelect").value;
    if (v) toast("已选择检索知识库：" + v, "ok", 2500);
  });
  loadRetrievalKbList();

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
  // Publish initial uiState + taskId to the reactive store so any future
  // module can subscribe via store.on('uiState', ...) / store.on('taskId', ...).
  syncUIState();
  try {
    if (window.store && typeof window.store.set === "function") {
      window.store.set({ taskId: S.taskId });
    }
  } catch (e) { /* store optional */ }
}

document.addEventListener("DOMContentLoaded", init);
