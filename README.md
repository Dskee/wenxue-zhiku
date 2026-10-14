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
