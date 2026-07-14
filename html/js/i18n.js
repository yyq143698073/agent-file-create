/* ── i18n preparation: centralized UI strings ────────────────────────
 *   Extract commonly used Chinese strings here so future translation
 *   only needs to edit this file. Loaded as plain <script> BEFORE app.
 *   Usage:  LANG.status_ready    or  LANG["大纲"]
 *   HTML-mixed strings (@see chat.js LANG.satisfiedLabel etc.) are
 *   left inlined for now — too entangled with markup.
 *   ═══════════════════════════════════════════════════════════════════ */

const LANG = {
  /* ── Role labels ────────────────────────────────────────────── */
  role_user: "用户",
  role_assistant: "助手",

  /* ── Doc type labels ────────────────────────────────────────── */
  label_outline: "大纲",
  label_content: "正文",
  label_report: "报告正文",
  label_quality_report: "报告质量评估",
  label_feedback: "反馈",
  label_selected: "已选中",
  label_version: "版本",
  label_lines: "行",

  /* ── Status ─────────────────────────────────────────────────── */
  status_ready: "就绪",
  status_done: "完成",
  status_success: "成功",
  status_failed: "失败",
  status_canceled: "已取消",
  status_generating: "生成中…",
  status_regenerating: "重新生成中…",
  status_uploading: "上传中…",
  status_evaluating: "评估中…",
  status_loading: "加载中…",
  status_queued: "排队中…",

  /* ── Common button / action labels ──────────────────────────── */
  btn_delete: "删除",
  btn_cancel: "取消",
  btn_save: "保存",
  btn_skip: "跳过",
  btn_send: "发送",
  btn_regen: "重生成",
  btn_stop: "停止",
  btn_upload: "上传",
  btn_append: "追加",

  /* ── Feedback ───────────────────────────────────────────────── */
  feedback_satisfied: "满意，继续",
  feedback_satisfied_progress: "满意，继续生成…",
  feedback_unsatisfied: "不满意，重新生成",
  feedback_submit: "提交反馈，重新生成",
  feedback_skip: "跳过，直接重做",
  feedback_back: "返回",

  /* ── Toast / notifications ──────────────────────────────────── */
  toast_saved: "已保存",
  toast_save_failed: "保存失败",
  toast_deleted: "已删除",
  toast_delete_failed: "删除失败",
  toast_upload_failed: "上传失败",
  toast_append_failed: "追加失败",
  toast_gen_failed: "生成失败",
  toast_regen_failed: "重新生成失败",
  toast_network_error: "网络连接失败，请检查服务是否运行",
  toast_need_task: "请先生成或加载一个任务",
  toast_task_busy: "任务正在进行中，请等待完成",
  toast_select_files: "请选择至少一个文件",
  toast_select_kb: "请先选择知识库",
  toast_select_file: "请选择文件",

  /* ── Chat ───────────────────────────────────────────────────── */
  chat_need_info: "需要补充信息",
  chat_report_done: "报告已完成",
  chat_report_generated: "报告已生成完成，请查看下方预览。",
  chat_no_task: "无任务",
  chat_general_mode: "通用助手模式（无任务）",

  /* ── KB ─────────────────────────────────────────────────────── */
  kb_loading_docs: "加载中…",
  kb_no_docs: "该知识库暂无文档",
  kb_loading_failed: "加载失败",
  kb_select: "— 选择一个知识库 —",

  /* ── Eval ───────────────────────────────────────────────────── */
  eval_no_data: "暂无数据",
  eval_llm_empty: "LLM 评判未启用或暂无评语",
  eval_no_content: "（无内容）",
  eval_dim_relevance: "相关性",
  eval_dim_faithfulness: "忠实度",
  eval_dim_coherence: "连贯性",
  eval_dim_completeness: "完整性",
  eval_dim_factscore: "原子事实",
  eval_dim_coverage: "主题覆盖",
  eval_dim_consistency: "一致性",
  eval_avg_score: "综合均分",
  eval_warn_count: "%d 条警告",
  eval_no_warn: "无警告",

  /* ── Versions ───────────────────────────────────────────────── */
  ver_no_content: "（无内容）",
  ver_same_version: "（同一版本）",
  ver_no_compare: "暂无版本可对比",
  ver_confirm_delete: "确定要删除 V{ver} 吗？此操作不可撤销。",
  ver_toast_deleted: "已删除 V{ver}",
  ver_toast_selected: "已选择 V{ver} 作为最终版本",
  ver_toast_select_failed: "选择版本失败: ",
  ver_only_one: "（该类型仅有 1 个版本，无法对比差异）",
  ver_no_versions: "（该类型暂无版本）",

  /* ── Quality gate / Final confirm ────────────────────────────── */
  qg_title: "报告已完成，是否进行质量评估？",
  qg_accepted: "要，开启评估",
  qg_skipped: "不了，跳过评估",
  fc_title: "请进行最终确认后渲染报告",
  fc_confirm: "确认渲染",
};
