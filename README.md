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

### 3. 重排序模型

`bge-reranker-v2-m3` 需放在 `models/models/bge-reranker-v2-m3`（可通过 `RERANK_MODEL_PATH` 改路径）。
从 ModelScope 下载（`--repo-type model` 与 `--local-dir` 为实测可用的参数写法）：

```bash
pip install modelscope
modelscope download --repo-type model \
  --local-dir ./models/models/bge-reranker-v2-m3 \
  BAAI/bge-reranker-v2-m3
```

> 若该路径已存在（例如指向别处的软链接），请先删除再下载，否则会写入失败。

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

## ⚙️ 配置项（环境变量 / `.env`）

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `SILICONFLOW_API_KEY` | 无（必填） | 大模型与 embedding 的 API Key |
| `SILICONFLOW_BASE_URL` | `https://api.siliconflow.cn/v1` | OpenAI 兼容接口地址 |
| `CHAT_MODEL_NAME` | `Qwen/Qwen3-8B` | 对话/出题所用模型 |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-large-zh-v1.5` | 向量化模型 |
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

> ⚠️ 本项目默认面向**本机/内网单机使用**，没有用户体系与权限隔离。若部署到公网，请务必：
> 设置 `API_TOKEN`、把 `CORS_ORIGINS` 收敛到真实前端域名、并在前面加一层反向代理做 HTTPS 与访问控制。

---

## 📝 说明

- `data/`、`knowledgebases/`、`temp/`、`*.db`、`models/` 均为运行时生成或本地资源，已加入 `.gitignore`。
- 删除知识库会同时删除 `./knowledgebases/<库名>.faiss` 与 `./data/<库名>/`，操作不可撤销。
- 修改库内笔记后会自动整库重建索引，库较大时该操作会明显变慢。
