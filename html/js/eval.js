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
  if (!scores) return '<div class="eval-empty">' + LANG.eval_no_data + '</div>';
  const dims = [
    [LANG.eval_dim_relevance, scores.relevance || 0],
    [LANG.eval_dim_faithfulness, scores.faithfulness || 0],
    [LANG.eval_dim_coherence, scores.coherence || 0],
    [LANG.eval_dim_completeness, scores.completeness || 0],
  ];
  const avg = (dims.reduce((s, d) => s + d[1], 0) / 4);
  let html = dims.map(d => renderScoreBar(d[0], d[1])).join("");
  html += '<div class="eval-avg">' +
    '<span>' + LANG.eval_avg_score + '</span>' +
    '<span class="eval-avg-badge" style="background:' + avgBadgeColor(avg) + '">' + (avg * 100).toFixed(0) + '</span>' +
  '</div>';
  return html;
}

/* ── Render Warnings Card ─────────────────────────────────────────── */

function renderWarningsList(warnings) {
  const card = $("warningsCard");
  const list = $("warningsList");
  const count = $("warningsCount");

  if (!warnings || !warnings.length) {
    card.style.display = "none";
    return;
  }

  card.style.display = "";
  count.textContent = warnings.length;

  list.innerHTML = warnings.map((w, i) => {
    const type = w.type || "warning";
    const severity = w.severity || "medium";
    const icon = type === "citation" ? "📖" : type === "contrastive" ? "⚖️" : "⚠️";
    const title = w.title || "待核实问题";
    const desc = w.description || "";
    const details = w.details || [];
    const context = w.context || "";
    const reason = w.reason || "";
    const suggestion = w.suggestion || "";

    let detailsHtml = "";
    if (details.length) {
      detailsHtml = `<div class="warning-detail">
        <div class="warning-detail-label">详情</div>
        ${details.map(d => `<div>• ${esc(d)}</div>`).join("")}
      </div>`;
    } else if (context) {
      detailsHtml = `<div class="warning-detail">
        <div class="warning-detail-label">引用上下文</div>
        <div>「${esc(context)}...」</div>
      </div>`;
    }

    if (reason) {
      detailsHtml += `<div class="warning-detail">
        <div class="warning-detail-label">问题原因</div>
        <div>${esc(reason)}</div>
      </div>`;
    }

    let suggestionHtml = "";
    if (suggestion) {
      suggestionHtml = `<div class="warning-suggestion">${esc(suggestion)}</div>`;
    }

    return `<div class="warning-item" id="warn-${i}">
      <div class="warning-header" onclick="toggleWarning(${i})">
        <div class="warning-icon ${type} ${severity}">${icon}</div>
        <div class="warning-title">${esc(title)}</div>
        <div class="warning-desc">${esc(desc)}</div>
        <div class="warning-toggle" id="warn-toggle-${i}">▶</div>
      </div>
      <div class="warning-body" id="warn-body-${i}" style="display:none">
        ${detailsHtml}
        ${suggestionHtml}
      </div>
    </div>`;
  }).join("");
}

function toggleWarning(i) {
  const body = $("warn-body-" + i);
  const toggle = $("warn-toggle-" + i);
  if (!body) return;

  if (body.style.display === "none") {
    body.style.display = "";
    if (toggle) { toggle.textContent = "▼"; toggle.classList.add("expanded"); }
  } else {
    body.style.display = "none";
    if (toggle) { toggle.textContent = "▶"; toggle.classList.remove("expanded"); }
  }
}

/* ── Eval metrics bar + badges ────────────────────────────────────── */

function renderEvalMetricsBar(metrics) {
  if (!metrics) return;
  $("evalCard").style.display = "";

  const realWarns = ((S._content || "").match(/> ⚠/g) || []).length;
  const wb = $("evalWarns");
  if (realWarns > 0) {
    wb.style.background = "#fef2f2"; wb.style.color = "#dc2626";
    wb.textContent = '\u26a0\ufe0f ' + LANG.eval_warn_count.replace('%d', realWarns);
  } else {
    wb.style.background = "#f0fdf4"; wb.style.color = "#16a34a";
    wb.textContent = "\u2705 " + LANG.eval_no_warn;
  }

  const fs = metrics.factscore;
  const fb = $("evalFActScore");
  if (fs != null && fs > 0) {
    const pct = Math.round(fs * 100);
    fb.style.display = ""; fb.style.background = pct >= 70 ? "#f0fdf4" : pct >= 40 ? "#fffbeb" : "#fef2f2";
    fb.style.color = pct >= 70 ? "#16a34a" : pct >= 40 ? "#b45309" : "#dc2626";
    fb.textContent = LANG.eval_dim_factscore + ': ' + pct + '%';
  } else { fb.style.display = "none"; }

  const cov = metrics.coverage;
  const cb = $("evalCoverage");
  if (cov != null && cov > 0) {
    const pct = Math.round(cov * 100);
    cb.style.display = ""; cb.style.background = pct >= 70 ? "#f0fdf4" : pct >= 40 ? "#fffbeb" : "#fef2f2";
    cb.style.color = pct >= 70 ? "#16a34a" : pct >= 40 ? "#b45309" : "#dc2626";
    cb.textContent = LANG.eval_dim_coverage + ': ' + pct + '%';
  } else { cb.style.display = "none"; }

  const cs = metrics.consistency_score;
  const csb = $("evalConsistency");
  if (cs != null) {
    const pct = Math.round(cs * 100);
    csb.style.display = ""; csb.style.background = pct >= 90 ? "#f0fdf4" : "#fffbeb";
    csb.style.color = pct >= 90 ? "#16a34a" : "#b45309";
    csb.textContent = LANG.eval_dim_consistency + ': ' + pct + '%';
  } else { csb.style.display = "none"; }

  const totalWarns = ((S._content || "").match(/> ⚠/g) || []).length;
  const citeWarns = ((S._content || "").match(/引用溯源提醒/g) || []).length;
  const cntrWarns = ((S._content || "").match(/对比论断待核实/g) || []).length;
  const title = $("evalCard").querySelector(".card-title");
  if (title) {
    let extra = [];
    if (totalWarns) extra.push(`${totalWarns}条事实`);
    if (citeWarns) extra.push(`${citeWarns}条引用`);
    if (cntrWarns) extra.push(`${cntrWarns}条对比`);
    title.textContent = extra.length ? '📊 ' + LANG.label_quality_report + '（' + extra.join('\uff0c') + '）' : '📊 ' + LANG.label_quality_report;
  }
}

function renderEvalBars(scores) {
  const dims = [];
  if (scores && (scores.relevance != null || scores.faithfulness != null)) {
    dims.push([LANG.eval_dim_relevance, scores.relevance || 0]);
    dims.push([LANG.eval_dim_faithfulness, scores.faithfulness || 0]);
    dims.push([LANG.eval_dim_coherence, scores.coherence || 0]);
    dims.push([LANG.eval_dim_completeness, scores.completeness || 0]);
  }
  dims.push([LANG.eval_dim_factscore, scores.factscore || 0]);
  dims.push([LANG.eval_dim_coverage, scores.coverage || 0]);

  if (!dims.length) { $("evalBars").innerHTML = ""; return; }
  let html = dims.map(d => renderScoreBar(d[0], d[1])).join("");
  const avg = dims.reduce((s, d) => s + d[1], 0) / dims.length;
  html += '<div class="eval-avg">' +
    '<span>' + LANG.eval_avg_score + '</span>' +
    '<span class="eval-avg-badge" style="background:' + avgBadgeColor(avg) + '">' + Math.round(avg * 100) + '</span>' +
  '</div>';
  $("evalBars").innerHTML = html;
}

function renderEvalContent(evalData) {
  $("evalContent").innerHTML = "";
  if (!evalData) return;
  renderEvalBars(evalData);
  const content = S._content || "";
  const warnBlocks = (content || "").split("\n").filter(line =>
    line.trim().startsWith(">") && /警告|提醒|核实|⚠|核查/.test(line)
  );
  if (warnBlocks && warnBlocks.length) {
    let h = '<div class="eval-warn-box">';
    for (const w of warnBlocks.slice(0, 8)) {
      h += `<div class="text-warn-item">${esc(w)}</div>`;
    }
    if (warnBlocks.length > 8) h += `<div class="text-warn-more">...还有 ${warnBlocks.length - 8} 条</div>`;
    h += '</div>';
    $("evalContent").innerHTML += h;
  }
}

function renderEval() {
  const data = S._evalData;
  if (!data) {
    $("evalCard").style.display = "none";
    $("warningsCard").style.display = "none";
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
      : '<div class="eval-empty">' + LANG.eval_llm_empty + '</div>';
  }
  content.innerHTML = html || '<div class="eval-empty">' + LANG.eval_no_data + '</div>';
}
