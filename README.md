# agent-file-create

多模态文档智能生成系统 — 上传 PDF/图片/Office 文件，自动提取信息、生成大纲、撰写正文，最终输出结构化文档（Markdown / DOCX / PDF）。

## 处理流程

```
上传文件 → 多模态抽取 → 大纲生成 → 内容生成 → 质量检查 → 模板渲染 → 输出文档
  │            │            │           │           │            │
  │        OCR + 视觉模型   结构化      逐节撰写     忠实度/      填充用户
  │        提取文字+数据    Markdown    长文段落     引用检查     模板样式
  │                       大纲
  │            │            │           │           │            │
  └── PDF/图片/Word/PPT/Excel             └── 失败自动重试 ──────┘
```

1. **多模态抽取** — 根据文件类型（图片/PDF/DOCX/PPTX/Excel）选择最优抽取策略，结合 OCR 和视觉语言模型提取结构化信息
2. **大纲生成** — 基于材料内容和用户需求，输出层级完整的 Markdown 大纲（含结构校验、命名质量检查、主题覆盖度检查）
3. **内容生成** — 按大纲章节逐节撰写正文，每节生成后进行忠实度审查，发现问题自动修正
4. **质量检查** — 多维度质量评分（忠实度、完整性、引用合规），不达标自动重试
5. **模板渲染** — 将生成内容填充到用户选择的输出模板，导出为最终文档

## 模型选型说明

本项目**默认使用云端大模型**（DeepSeek-v4-flash）进行大纲生成和内容撰写。原因：

- **指令遵循能力**：大纲生成涉及十余条结构约束（层级规范、命名原则、批判章节等），本地 4B/9B 模型难以稳定遵守，频繁出现格式错误需要反复重试
- **长文质量**：内容生成要求基于材料撰写数千字流畅中文段落，小模型容易出现逻辑断裂、重复啰嗦、偏离素材等问题
- **推理速度**：本地 14B 模型在消费级显卡上生成速度约 5-12 tok/s，一次大纲生成需 1.5-3 分钟，算上重试可能超过 5 分钟；云端模型仅需 3-5 秒

架构上完整支持 OpenAI 兼容 API 和 Ollama 本地模型，只需修改环境变量即可切换。如果本地部署 14B+ 模型且可接受较慢速度，可通过配置切换到本地推理。

## 项目结构

```
agent_file_create/
├── document/               # 核心：文档生成 Pipeline
│   ├── extractor.py        #   多模态信息抽取
│   ├── outline_generator.py # 大纲生成 + 结构/命名/覆盖度校验
│   ├── content_generator.py # 逐节正文撰写 + 忠实度审查
│   ├── template_renderer.py # 模板渲染（md/docx/pdf）
│   ├── _critic.py          #   批判性审查节点
│   ├── _reviewer.py        #   事实核查 + 连贯性审查
│   └── _quality.py         #   多维度质量评分
├── agent/                  # LangGraph 文档生成 Agent
├── rag/                    # RAG 知识库（chunk/embed/检索/重排序）
├── chat/                   # 对话系统（意图识别/历史管理）
├── prompts/                # Prompt 模板管理
├── quality/                # 质量评估 Pipeline（忠实度/引用/对比）
├── evaluation/             # 离线评测框架
├── skills/                 # 技能系统（文件分析/图表生成/网页搜索）
├── web/                    # FastAPI Web 服务
├── config.py               # 全局配置（环境变量驱动）
├── llm_client.py           # LLM 调用封装（支持 OpenAI 兼容 API + Ollama）
└── preprocessor.py         # 文件预处理
html/                       # Web 前端
scripts/                    # 评测 & 测试脚本
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example`（如有）或直接创建 `.env` 文件。**云端模型（推荐）** 最少配置：

```env
# 大纲 & 正文生成（DeepSeek 兼容 API）
OUTLINE_MODEL_NAME=deepseek-v4-flash
OUTLINE_API_KEY=sk-your-key
CONTENT_MODEL_NAME=deepseek-v4-flash
CONTENT_API_KEY=sk-your-key

# 本地模型（用于信息抽取和 Embedding，需先安装 Ollama）
MODEL_NAME=qwen3:4b
EMBED_MODEL_NAME=bge-m3:latest
VISION_MODEL_NAME=minicpm-v:8b
```

如果想切回本地模型进行大纲/内容生成：

```env
OUTLINE_API_STYLE=ollama
OUTLINE_MODEL_NAME=qwen3:14b
OUTLINE_API_ENDPOINT=http://localhost:11434/api/generate
# 同上修改 CONTENT_* 变量
```

全部配置项见 `config.py`。

### 3. 启动

**命令行：**
```bash
python -m agent_file_create
```
将材料放入 `resource/` 目录后按提示操作。

**Web 界面：**
```bash
python agent_web.py
```
访问 `http://127.0.0.1:8000/`，支持文件上传、模板选择、在线生成。

### 4. 知识库（可选）

Web 端内置轻量 RAG 知识库：
- 上传文件 → 自动分块 + 向量化存储
- 对话中通过 `/kb use <name>` 切换知识库，问答时自动检索

## License

MIT
