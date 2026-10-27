# 问学知库（WenXue ZhiKu）

一个基于 **RAG（检索增强生成）** 的本地知识库学习助手：上传自己的笔记/题库 → 自动向量化入库 →
既能**多轮对话问答**（自动决定查本地知识库还是联网搜索），也能**自动出题**、把错题整理成笔记回存知识库。

前端用 Gradio，后端用 FastAPI，检索采用 **FAISS 向量检索 + BM25 关键词检索的混合召回**，
再经 **bge-reranker-v2-m3** 精排，兼顾语义与关键词命中。

---

## ✨ 功能特性

| 模块 | 说明 |
| --- | --- |
| 知识库管理 | 创建/删除知识库，上传 `.txt/.md/.csv/.pdf` 文档，可配置切分粒度与重叠长度，自动向量化 |
| 知识库文件管理 | 查看/编辑/删除库内笔记文件，编辑后自动整库重建向量索引，避免检索到过期内容 |
| 智能问答 | ReAct Agent + 工具调用：`knowledge_search`（查本地库）、`web_search`（联网）、`weather_check`（天气）；支持流式逐字输出 |
| 指定知识库提问 | 可手动指定检索某个库；不指定时按「库名/描述与问题的关键词重叠度」自动挑选最相关的库注入 |
| 对话问答整理 | 按「回溯分类 → 提炼最优提问模板 → 过滤保留 → 按序输出」把多轮对话整理成结构化笔记 |
| 笔记自动归档 | 笔记导出时用 embedding 余弦相似度自动匹配已有知识库，匹配不到则按主题自动新建 |
| 自动出题 / 错题本 | 基于库内容生成选择题/填空题/判断题/简答题，对完答案后把错题勾选保存为「错题」笔记，同名自动追加 |

---

## 🏗 技术栈

- **Web 框架**：FastAPI 0.141 + Uvicorn；前端 Gradio 6.26
- **大模型 / Agent**：LangChain 1.4（`langchain-classic` 的 ReAct Agent）+ OpenAI 兼容接口（SiliconFlow）
- **向量检索**：FAISS（`faiss-cpu`）+ BM25（`rank-bm25`，中文用 jieba 分词）→ `EnsembleRetriever` 混合召回
- **重排序**：`sentence-transformers` + `bge-reranker-v2-m3`
- **Embedding**：`BAAI/bge-large-zh-v1.5`（走 API）
- **元数据存储**：SQLAlchemy + SQLite
- **文档解析**：LangChain Community Loaders（TextLoader / CSVLoader / PyPDFLoader）

---

## 📁 目录结构

```
.
├── FastAPIRouter.py                 # FastAPI 聚合入口（CORS、可选鉴权、挂载全部路由）
├── knowledge_gradio.py              # Gradio 前端入口与页面布局
├── configs/
│   ├── setting.py                   # 全局配置（全部从环境变量/.env 读取，不含任何密钥）
│   └── prompt.py                    # Agent 提示词、对话总结提示词
├── chat/
│   └── ChatFastAPI.py               # /chat/chatbot（流式问答）、/chat/summarize（对话整理）
├── knowledgebase_server/loader/
│   ├── KnowledgeFastAPI.py          # 知识库 CRUD、上传、检索、笔记保存、出题
│   └── loader.py                    # 按扩展名分派文档加载器
├── db_server/data/
│   ├── KnowledgeBase.py             # SQLAlchemy engine / session
│   └── KnowledgeDB.py               # 知识库元数据 CRUD
├── tools/                           # Agent 可调用的工具
│   ├── tools_select.py              # 工具注册
│   ├── knowledge_search.py          # 查本地知识库
│   ├── web_search.py                # SerpAPI 主通道 + 必应抓取回退通道
│   └── weather_check.py             # wttr.in 天气（无需 Key）
├── utils/
│   ├── callback.py                  # 流式 token 回调 → 异步队列
│   ├── load_docs.py                 # 读取上传文件正文
│   └── paths.py                     # 路径安全工具（防路径穿越）
├── webui/                           # 前端各页面的请求逻辑（不含 UI 组件）
│   ├── Chat_UI.py  ├── Knowledge_UI.py  ├── Summary_UI.py  └── Quiz_UI.py
├── eval/                            # 评测：指标纯函数、配置、数据集、消融脚本
│   ├── configs/*.toml               # 三份检索配置（纯向量 / 混合 / 混合+重排）
│   ├── dataset/                     # 冻结数据集（specs 里是逐字证据片段）+ CHANGELOG
│   ├── corpus/唐诗三百首.md          # 公版语料，供 CI 复现
│   └── run_ablation.py              # 三配置 × 40 题，出 results/YYYY-MM-DD.md
├── results/                         # 实跑生成的评测报告（Markdown + 原始 JSON）
├── data/                            # 知识库文档与笔记（运行时生成，已 gitignore）
├── knowledgebases/                  # FAISS 向量索引（运行时生成，已 gitignore）
└── temp/                            # 对话上传的临时文件（已 gitignore）
```

---

## 🚀 快速开始

### 1. 准备环境

Python 3.10（开发环境为 3.10.x）：

```bash
conda create -n ai_rag python=3.10 -y
conda activate ai_rag
pip install -r requirements.txt
```

### 2. 配置密钥（重要）

**本项目不在源码里保存任何密钥**，全部通过 `.env` 读取：

```bash
cp .env.example .env
```

编辑 `.env`，至少填入大模型 Key：

```ini
SILICONFLOW_API_KEY=sk-你的密钥        # 必填，https://cloud.siliconflow.cn/account/ak
SERPAPI_KEY=                          # 选填；不填则联网搜索自动走必应抓取回退通道
```

> `.env` 已在 `.gitignore` 中，不会被提交到 GitHub。

**哪些功能依赖 API Key？**（容易搞混，列清楚）

| 功能 | 是否需要 API Key | 是否需要重排序模型 |
| --- | --- | --- |
| 启动服务、打开界面、设置页 | ❌ 不需要 | ❌ 不需要 |
| **新建知识库** | ✅ **需要**（建空索引时会做一次向量化） | ❌ 不需要 |
| **上传文档 / 编辑笔记重建索引** | ✅ **需要**（把文本切块向量化） | ❌ 不需要 |
| **知识库检索** | ✅ **需要**（把问题向量化才能比对） | ⭕ 可选（缺了只是不精排） |
| 对话问答 / 总结 / 出题 | ✅ 需要 | ❌ 不需要 |

也就是说：**重排序模型是"可选增强"，API Key 是"必需项"**。
没配 Key 时相关接口会返回 **503** 并说明原因（不会再是看不懂的 500）。

### 3. 重排序模型（**可选** —— 不下载也能先跑起来）

`bge-reranker-v2-m3` 用于对混合检索的候选做精排，建议放在 `models/models/bge-reranker-v2-m3`
（可通过环境变量 `RERANK_MODEL_PATH` 改路径）。

> **它是懒加载的**：服务启动、知识库管理、上传、对话、设置页都**不需要**它。
> 只有真正调用检索（`/knowledge/search_kb`）时才第一次加载。
> 如果没下载，检索**不会报错**，而是降级为"仅混合检索、不重排"，并在返回里带
> `"reranked": false` 和一条 `warning` 提示——所以你可以先跑通全流程，之后再补模型。

从 ModelScope 下载（`--repo-type model` 与 `--local-dir` 为实测可用的参数写法）：

```bash
pip install modelscope
modelscope download --repo-type model \
  --local-dir ./models/models/bge-reranker-v2-m3 \
  BAAI/bge-reranker-v2-m3
```

> 若该路径已存在（例如指向别处的软链接），请先删除再下载，否则会写入失败。

**怎么确认它就绪了**：访问 `GET /knowledge/status`，看 `rerank.rerank_loaded` 是否为 `true`。

### 4. 启动

需要**两个进程**（后端 API + 前端界面）：

```bash
# 终端 1：后端（默认 16055）
uvicorn FastAPIRouter:app --host 0.0.0.0 --port 16055

# 终端 2：前端（默认 7860）
python knowledge_gradio.py
```

打开 <http://127.0.0.1:7860> 即可使用；接口文档在 <http://127.0.0.1:16055/docs>。

---

## 🎛 模型与密钥设置（页面内即可修改）

不用改代码、也不用重启，直接在页面里切换模型和填 Key。

**方式一：侧边栏（最顺手）**
「聊天机器人」页左侧有一个 **🔑 API Key** 折叠面板，填入 Key 后点「保存 API Key」即可。

**方式二：「⚙️ 设置」页（完整）**
页面上方新增 **⚙️ 设置** 标签页：

| 项 | 说明 |
| --- | --- |
| **模型来源** | **硅基流动（云端）** / **本地模型**（Ollama、LM Studio、vLLM）/ **自定义**。切换来源会**自动填好默认接口地址**并给出提示 |
| **接口地址（base_url）** | 自动填好后仍可手动改，任何 OpenAI 兼容地址都能直接粘贴 |
| **对话模型** | 可搜索的下拉框，点「🔄 刷新对话模型」从该来源**实时拉取**；也可直接粘贴模型名（`Qwen/Qwen3-8B`、`qwen2.5:7b`…） |
| **向量化（Embedding）模型** | 同上，用于知识库向量化 |
| **API Key** | 密码框，留空表示**不修改**；本地模型通常不需要填 |
| **向量化使用独立配置**（可选） | 默认**跟随对话的地址与 Key**；可勾选后单独填，用于「对话跑本地、向量化用云端」 |
| **🔍 测试连接** | 真实调用一次对话模型与向量化模型，验证「地址 + Key + 模型名」这套组合是否可用 |

### 用本地模型（Ollama 为例）

```bash
# 1) 宿主机上启动 Ollama 并拉模型
ollama serve
ollama pull qwen2.5:7b
ollama pull bge-m3          # 向量化模型（知识库需要）
```
2) 页面「设置」→ **模型来源** 选「本地模型」，地址会自动填成 `http://host.docker.internal:11434/v1`
3) 点「🔄 刷新对话模型」拉到本机已下载的模型，选中 → **保存设置** → **🔍 测试连接**

> ⚠️ **地址的坑**：应用跑在 Docker 里时，本地服务必须用 `host.docker.internal`，
> 用 `127.0.0.1` 指的是**容器自己**，必然连不上。端口对应关系：Ollama `11434`、LM Studio `1234`、vLLM `8000`。

三个关键行为：

1. **保存后立即生效**，不需要重启服务——模型与 Key 是在**每次调用时**从运行时可改配置（`configs/runtime_config.py`）里读的；
2. **持久化到 `.env`**，重启后依然有效（`.env` 已被 gitignore，不会上传）；
3. **接口不回传明文 Key**，页面只显示打码值（如 `sk-ydsw****dvzk`），避免 Key 出现在页面或日志里。

> 换向量化模型后，旧索引的向量维度可能与新模型不兼容，建议把知识库文档重新上传一次以重建索引。

---

## ⚙️ 配置项（环境变量 / `.env`）

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `PROFILE` | `siliconflow` | 模型来源：`siliconflow`（云端）/ `local`（本地 Ollama、LM Studio、vLLM）/ `custom` |
| `SILICONFLOW_API_KEY` | 无（云端必填） | 大模型与 embedding 的 API Key（也可在页面「设置」里填；本地模型可不填） |
| `SILICONFLOW_BASE_URL` | `https://api.siliconflow.cn/v1` | OpenAI 兼容接口地址（页面可改；本地模型填 `http://host.docker.internal:11434/v1`） |
| `CHAT_MODEL_NAME` | `Qwen/Qwen3-8B` | 对话/出题所用模型（页面可改） |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-large-zh-v1.5` | 向量化模型（页面可改） |
| `EMBEDDING_BASE_URL` | 空 | 向量化专用地址；留空 = 跟随上面的地址 |
| `EMBEDDING_API_KEY` | 空 | 向量化专用 Key；留空 = 跟随上面的 Key |
| `SERPAPI_KEY` | 空 | 联网搜索主通道 Key，留空则直接用必应回退 |
| `BACKEND_URL` | `http://127.0.0.1:16055` | 前端与 Agent 工具访问后端的地址 |
| `TIME_OUT` | `300` | Agent 任务/工具调用最长等待秒数 |
| `CORS_ORIGINS` | `http://127.0.0.1:7860,http://localhost:7860` | 允许跨域访问后端的来源（逗号分隔） |
| `MAX_UPLOAD_MB` | `50` | 单次上传文件大小上限 |
| `API_TOKEN` | 空 | 可选接口令牌；设置后所有 `/chat`、`/knowledge` 接口需带 `X-API-Token` 头 |
| `KB_DIR` / `FILE_STORAGE_DIR` / `TEMP_FILE_STORAGE_DIR` | `./knowledgebases` / `./data` / `./temp/data` | 数据目录 |
| `RERANK_MODEL_PATH` | `./models/models/bge-reranker-v2-m3` | 重排序模型路径 |

---

## 🔌 主要接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/chat/chatbot` | 流式问答（SSE，逐 token 返回 `{"answer": ...}`） |
| `POST` | `/chat/summarize` | 把多轮对话整理为结构化问答笔记 |
| `POST` | `/knowledge/add_knowledge` | 创建知识库 |
| `DELETE` | `/knowledge/del_knowledge` | 删除知识库（含索引与文档目录） |
| `GET` | `/knowledge/queryall_knowledge` | 列出全部知识库名称 |
| `POST` | `/knowledge/upload_docs` | 上传文档并向量化 |
| `POST` | `/knowledge/search_kb` | 混合检索 + 重排序，返回带分数的片段 |
| `POST` | `/knowledge/list_files` | 列出库内文件 |
| `POST` | `/knowledge/read_file` | 读取库内笔记正文 |
| `POST` | `/knowledge/update_file` | 保存笔记修改并重建索引 |
| `POST` | `/knowledge/delete_file` | 删除库内文件 |
| `POST` | `/knowledge/match_kb` | 按主题语义匹配最合适的知识库 |
| `POST` | `/knowledge/save_note` | 保存笔记到知识库（自动匹配/新建） |
| `POST` | `/knowledge/quiz` | 基于库内容生成练习题 |
| `GET` | `/settings` | 读取当前设置（Key 只回传打码值） |
| `POST` | `/settings` | 保存来源/模型/Key/地址并**立即生效**（字段留空 = 不修改，`clear` 可显式清空） |
| `GET` | `/settings/profiles` | 返回模型来源预设（默认地址、默认模型名、提示文案） |
| `GET` | `/settings/models?kind=chat\|embedding` | 从对应来源实时拉取可用模型列表 |
| `POST` | `/settings/test` | 真实调用一次模型，验证「地址 + Key + 模型名」是否可用 |

---

## 🧪 评测与检索消融

这个项目原本只能靠「感觉答得还行」判断效果——改切块大小、关重排，谁也说不清是变好还是变坏。
`eval/` 补上的就是这件事：**同一份代码 + 三份配置 + 一份冻结数据集**，跑出可复现的数字。

### 三份配置（只换参数，不改代码）

| 配置文件 | 纯向量 | BM25 | 重排 | top_k_vector | top_k_bm25 | top_k_final |
| --- | --- | --- | --- | --- | --- | --- |
| `eval/configs/vector_only.toml` | ✅ | ❌ | ❌ | 5 | 4 | 5 |
| `eval/configs/hybrid.toml` | ✅ | ✅ | ❌ | 4 | 4 | 5 |
| `eval/configs/hybrid_rerank.toml`（线上默认） | ✅ | ✅ | ✅ | 4 | 4 | 5 |

三份配置由 `knowledgebase_server/loader/KnowledgeFastAPI.py` 里的 `RetrievalConfig` 承载，
`search_kb` 的对外行为与改造前等价（同一问题返回同样的 5 条、同样的重排分数）。

### 数据集 v1.0（冻结）

- 40 条：`normal` 24 / `boundary` 8 / `no_answer` 8，逐条人工核过 gold。
- md5 `559d14690bb7338048021235f28019f7`，变更记录见 [`eval/dataset/CHANGELOG.md`](eval/dataset/CHANGELOG.md)。
- gold 不是手抄 UUID：`eval/dataset/specs/v1.0.json` 里逐字摘答案片段，由构建脚本归一化后反查
  **内容寻址**的 chunk id（`sha1(文件名 + 正文)[:16]`）。语料一改，构建直接失败，标注不会悄悄漂移。
- 语料共 874 个 chunk：Agent 面试问答（私有笔记，不入库）+ 唐诗三百首（公版，提交在 `eval/corpus/`）。

### 指标口径

零 LLM 裁判，全是纯函数（`eval/metrics.py`，63 个单测）：

- **证据覆盖 Recall@5 / MRR@5 / nDCG@5**（主指标）：答案片段是否出现在 top-5 的某段正文里。
  128 字的切块会把一段答案切开，只按 chunk id 判分会被切块边界噪声吃掉，所以主指标看「上下文里有没有这句话」。
- **块级 Recall@5**（对照）：整块 chunk id 精确匹配，用来观察索引本身有没有取到同一块。
- **误拒率 / 拒答正确率**：字符串规则判定——拒答语只在回答**开头**才算，末尾的免责声明不算拒答
  （这个坑实测把误拒率虚高到 100%）。

### 实测结果（2026-10-27，Qwen3-8B + bge-large-zh-v1.5，874 chunk）

| 配置 | 证据 Recall@5 | 证据 MRR@5 | 证据 nDCG@5 | 块级 Recall@5 | 误拒率 | 拒答正确率 | 检索 p50 | 检索 p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `vector_only` | 0.719 | 0.526 | 0.575 | 0.703 | 0% | 75% | 98 ms | 108 ms |
| `hybrid` | 0.656 | 0.490 | 0.532 | 0.641 | 3% | 75% | 104 ms | 120 ms |
| `hybrid_rerank` | 0.719 | 0.594 | 0.625 | 0.719 | 0% | 75% | 1544 ms | 2165 ms |

读数（逐题命中名次见 [`results/2026-10-27.md`](results/2026-10-27.md)）：

1. **重排买的是顺序，不是覆盖率**：`hybrid_rerank` 的 Recall@5 与纯向量持平（0.719），
   但 MRR@5 从 0.526 升到 0.594、nDCG@5 从 0.575 升到 0.625，代价是检索 p50 从 98 ms 涨到 1544 ms（约 15 倍）。
2. **这个语料上 BM25 反而拖后腿**：`hybrid` 覆盖率比纯向量低 6.3 个百分点。128 字的小块里，
   BM25 按字面命中「注意力」「位置编码」这类高频词，把同词不同义的块挤进 top-5。
   这是真实结果，不做选择性汇报——它说明混合召回的权重和切块大小值得单独调，而不是「加了就一定好」。
3. **生成层没有乱拒，但会硬答**：检索已覆盖答案片段时误拒率 0%；8 条 `no_answer` 上拒答正确率 75%，
   即 2 条被硬答。当前更值得修的是这个短板。

### 怎么跑

```bash
# 1) 先把两篇语料上传成知识库「评测基准」（走真实 /knowledge/upload_docs，chunk_size=128 / overlap=20）
# 2) 跑三种配置，写出 results/YYYY-MM-DD.md 与同名 JSON
python -m eval.run_ablation --kb 评测基准 --dataset eval/dataset/v1.0.jsonl \
    --out results/2026-10-27.md --json-out results/2026-10-27.json
# 只测检索（跳过生成层，秒级出结果）
python -m eval.run_ablation --kb 评测基准 --no-generate --out /tmp/retrieval.md
# 只想改报告排版、不想重跑实验：用上次的 JSON 重渲染
python -m eval.run_ablation --from-json results/2026-10-27.json --out results/2026-10-27.md
# 单测（不联网、不加载模型）
python -m unittest discover -s eval -t .
```

生成层固定 `temperature=0`、`max_tokens=512`、单轮固定 prompt，并显式关闭 Qwen3 的思考模式
（实测同一问题 107.7s → 1.0s；评测要看的是答案层，不是思维链）。

### CI

`.github/workflows/eval.yml`：

- **unit**（push / PR 自动跑）：只跑指标与拒答规则的单测，不装任何第三方包——指标算错不会被环境问题掩盖。
- **retrieval**（仅手动 `workflow_dispatch`）：装 CPU 版 torch + `eval/requirements-eval.txt`（**不含 2.2GB 重排权重**），
  用提交在仓库里的公版唐诗语料重建知识库，跑 `vector_only` + `hybrid`，报告写进 Job Summary。
  需要在仓库 Secrets 配 `SILICONFLOW_API_KEY`。`hybrid_rerank` 要下 2.2GB 本地权重，CI 不跑，也不假装跑过。

### 局限

- 40 条是「够用但不显著」，趋势比绝对值可靠；生成层单次运行有抖动。
- 私有笔记语料不入库，克隆者只能复现唐诗那一半（CI 就是这么做的）。
- 语料里的唐诗是繁体/异体字版本，标准写法（「滚滚来」「猿啸哀」）匹配不到，出题按语料原文写。

---

## 🔐 安全设计

1. **密钥零硬编码**：所有 Key 只从环境变量 / `.env` 读取；`.env` 与 `secrets.json` 已 gitignore；
   仓库只提供 `.env.example` 模板。缺失 Key 时启动会打印醒目告警，调用模型时抛出可操作的错误信息。
2. **路径穿越防护**（`utils/paths.py`）：知识库名称走白名单校验，文件名统一清洗为纯文件名，
   所有文件系统访问都经 `resolve_within()` 校验是否逃出允许的根目录。
   注意：仅用 `str.startswith()` 判断前缀是不够的（`/app/data-evil` 也以 `/app/data` 开头）。
3. **上传限制**：文件名清洗 + 单文件大小上限（`MAX_UPLOAD_MB`）。
4. **CORS 收紧**：只放行 `CORS_ORIGINS` 中配置的来源（默认仅本机前端）。
5. **可选接口鉴权**：设置 `API_TOKEN` 后，所有业务接口都要求 `X-API-Token` 请求头，
   并使用 `secrets.compare_digest` 比较以避免时序侧信道。
6. **FAISS 反序列化**：加载索引时使用了 `allow_dangerous_deserialization=True`（FAISS 的
   `index.pkl` 是 pickle 格式）。索引文件在本地生成，请勿把不可信来源的 `knowledgebases/`
   目录拷贝进来，否则存在反序列化执行代码的风险。
7. **设置接口不回传明文 Key**：`GET /settings` 只返回 `has_key` 与打码值（`sk-ydsw****dvzk`），
   前端密码框永远留空、留空即不修改，避免 Key 出现在页面、浏览器历史或日志里。

> ⚠️ 本项目默认面向**本机/内网单机使用**，没有用户体系与权限隔离。若部署到公网，请务必：
> 设置 `API_TOKEN`、把 `CORS_ORIGINS` 收敛到真实前端域名、并在前面加一层反向代理做 HTTPS 与访问控制。

---

## 📝 说明

- `data/`、`knowledgebases/`、`temp/`、`*.db`、`models/` 均为运行时生成或本地资源，已加入 `.gitignore`。
- 删除知识库会同时删除 `./knowledgebases/<库名>.faiss` 与 `./data/<库名>/`，操作不可撤销。
- 修改库内笔记后会自动整库重建索引，库较大时该操作会明显变慢。
