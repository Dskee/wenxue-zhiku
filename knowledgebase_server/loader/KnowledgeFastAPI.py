from typing import List
from fastapi import Body, APIRouter, Request, HTTPException, UploadFile, File, Form, Query
from db_server.data.KnowledgeBase import session
from db_server.data.KnowledgeDB import add, delete, queryall
import jieba
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from langchain_community.retrievers import BM25Retriever
# ⚠️ EnsembleRetriever不在community，在 langchain.retrievers
from langchain_classic.retrievers import EnsembleRetriever

from knowledgebase_server.loader.loader import data_loader

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from configs.setting import (
    KB_DIR,
    FILE_STORAGE_DIR,
    RERANK_MODEL_PATH,
    base_url,
    MAX_UPLOAD_MB,
)
from configs import runtime_config
from utils.paths import (
    UnsafePathError,
    resolve_within,
    safe_filename,
    safe_kb_name,
)
import os
import shutil
import math
import re
import threading
import time
from pydantic import BaseModel
from sentence_transformers import CrossEncoder



# 确保知识库目录存在（KB_DIR / FILE_STORAGE_DIR 统一来自 configs.setting，
# 此前在本文件内二次写死为 './knowledgebases'、'./data'，会覆盖配置，已移除）
if not os.path.exists(KB_DIR):
    os.makedirs(KB_DIR)
knowledgeapp = APIRouter(prefix='/knowledge', tags=['knowledge management'])
kb_router = APIRouter(prefix='/knowledge', tags=["users management"])




class CustomEmbeddingModel(Embeddings):
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name

    def embed_documents(self, texts):
        return [self.model.client.create(model=self.model_name, input=text).data[0].embedding for text in texts]

    def embed_query(self, text):
        return self.model.client.create(model=self.model_name, input=text).data[0].embedding


# 密钥 / 地址 / 向量模型都来自运行时可改配置（「设置」页可改，改完立刻生效并写回 .env）。
# 这里用一个**可重建的代理对象**代替原来一次性构造的客户端：
#   - 对外仍然是 Embeddings 实例（FAISS.load_local 的 isinstance 校验能通过），
#     所以所有调用点（embed_query / embed_documents）写法完全不用改；
#   - 内部按 runtime_config.version() 判断配置是否变过，变了就重建底层客户端，
#     这样在页面上换了 API Key 之后，向量化立刻用新 Key，无需重启服务。
_embedding_lock = threading.RLock()
_embedding_cache: dict = {"version": -1, "model": None}


def _build_embedding_model() -> "CustomEmbeddingModel":
    """按当前设置构造一个向量化客户端（可使用向量化专属的地址与 Key）。"""
    client = OpenAIEmbeddings(
        api_key=runtime_config.api_key_for("embedding") or runtime_config.PLACEHOLDER_KEY,
        base_url=runtime_config.base_url_for("embedding") or base_url,
    )
    name = runtime_config.get("embedding_model") or "BAAI/bge-large-zh-v1.5"
    return CustomEmbeddingModel(client, name)


def get_embedding_model() -> "CustomEmbeddingModel":
    """取当前生效的向量化模型；配置变过则自动重建。"""
    with _embedding_lock:
        current = runtime_config.version()
        if _embedding_cache["model"] is None or _embedding_cache["version"] != current:
            _embedding_cache["model"] = _build_embedding_model()
            _embedding_cache["version"] = current
            print(f"[embedding] 已按最新设置初始化向量化模型 (version={current})")
        return _embedding_cache["model"]


class _EmbeddingProxy(Embeddings):
    """把调用转发给"当前生效"的向量化模型，使配置改动无需重启即可生效。"""

    def embed_documents(self, texts):
        return get_embedding_model().embed_documents(texts)

    def embed_query(self, text):
        return get_embedding_model().embed_query(text)


embedding_model = _EmbeddingProxy()


# ==================== 路径安全收敛点 ====================
# 说明：知识库名称、文件名均来自 HTTP 请求，可被伪造。下面三个函数是所有文件系统访问的
# 唯一入口，统一完成「名称白名单校验 + 路径穿越校验」，非法输入返回 400 而不是 500。
def _check_kb_name(kb_name: str) -> str:
    """校验知识库名称，非法则抛出 HTTP 400。"""
    try:
        return safe_kb_name(kb_name)
    except UnsafePathError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _kb_index_path(kb_name: str) -> str:
    """./knowledgebases/<kb_name>.faiss 的绝对路径（已校验，无法逃出 KB_DIR）。"""
    name = _check_kb_name(kb_name)
    try:
        return resolve_within(KB_DIR, f"{name}.faiss")
    except UnsafePathError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _kb_storage_dir(kb_name: str) -> str:
    """./data/<kb_name> 知识库文档目录的绝对路径（已校验，无法逃出 FILE_STORAGE_DIR）。"""
    name = _check_kb_name(kb_name)
    try:
        return resolve_within(FILE_STORAGE_DIR, name)
    except UnsafePathError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _kb_file_path(kb_name: str, file_name: str) -> str:
    """知识库内某个文档的绝对路径。

    文件名必须是「纯文件名」：一旦清洗结果与原值不一致（说明含 `../`、`/`、`\\`
    等目录成分），直接判为非法，避免静默改写成另一个文件。
    """
    target_dir = _kb_storage_dir(kb_name)
    cleaned = safe_filename(file_name, fallback="")
    if not cleaned or cleaned != str(file_name or "").strip():
        raise HTTPException(status_code=400, detail="非法的文件名")
    try:
        return resolve_within(target_dir, cleaned)
    except UnsafePathError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@knowledgeapp.post('/add_knowledge')
async def add_knowledge(name: str = Body(..., description='知识库名称'),
                        description: str = Body(..., description='知识库描述')):
    name = _check_kb_name(name)
    kb_path = _kb_index_path(name)
    if os.path.exists(kb_path):
        raise HTTPException(status_code=400, detail='知识库已经存在')

    doc = Document(page_content="init", metadata={})
    vector_store = FAISS.from_documents([doc], embedding_model, normalize_L2=True)
    ids = list(vector_store.docstore._dict.keys())
    vector_store.delete(ids)
    vector_store.save_local(kb_path)

    msg = add(session, name, description)
    return {"msg": msg}


@knowledgeapp.delete('/del_knowledge')
async def del_knowledge(name: str = Query(..., description='要删除的知识库名称')):
    # 名称校验在前：此前 name 被直接拼进路径，`../` 可让 shutil.rmtree 删除项目外的目录
    name = _check_kb_name(name)
    kb_path = _kb_index_path(name)
    kb_files_path = _kb_storage_dir(name)
    print(kb_files_path)
    if not os.path.exists(kb_path):
        raise HTTPException(status_code=404, detail="Database not found")
    shutil.rmtree(kb_path)
    if os.path.exists(kb_files_path):
        shutil.rmtree(kb_files_path)

    msg = delete(session, name)
    return {"msg": msg}


@knowledgeapp.get('/queryall_knowledge')
async def knowledge_list():
    knowledge_list = queryall(session)
    result = [knowledge['name'] for knowledge in knowledge_list]
    return result


@kb_router.post('/upload_docs')
async def upload_docs(
    files: List[UploadFile] = File(...),
    kb_name: str = Form(..., description='知识库名称'),
    chunk_size: int = Form(128, description='知识库中单段文本最大长度'),
    chunk_overlap: int = Form(20, description='知识库中相邻文本重合长度')
):
    kb_name = _check_kb_name(kb_name)
    kb_path = _kb_index_path(kb_name)
    kb_file_storage_path = _kb_storage_dir(kb_name)

    if not os.path.exists(kb_path):
        raise HTTPException(status_code=404, detail="Knowledge base not found")


    os.makedirs(kb_file_storage_path, exist_ok=True)
    updated_file = []
    limit = MAX_UPLOAD_MB * 1024 * 1024
    for file in files:
        # 清洗上传文件名（原实现直接用 file.filename 拼路径，`../../x` 可写到项目外）
        safe_name = safe_filename(file.filename)
        if not safe_name:
            raise HTTPException(status_code=400, detail="非法的文件名")
        file_path = resolve_within(kb_file_storage_path, safe_name)
        file_content = file.file.read()
        # 上传大小限制：避免超大文件一次性读入内存并打满磁盘
        if len(file_content) > limit:
            raise HTTPException(
                status_code=413,
                detail=f"文件 {safe_name} 超过 {MAX_UPLOAD_MB}MB 上传上限",
            )


        with open(file_path, 'wb') as f:
            f.write(file_content)


        loader = data_loader().get_loader(file_path)
        docs = loader.load()


        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=[
            "\n\n", 
            "\n", 
            " ", 
            ".",
            ",",
            "，",
            "。"
            ""],
        )

        chunks = text_splitter.split_documents(docs)

        vectorstore = FAISS.load_local(kb_path, embedding_model, allow_dangerous_deserialization=True)

        vectorstore.add_documents(chunks)

        vectorstore.save_local(kb_path)

        updated_file.append(safe_name)

    return {"message": f"File '{updated_file}' added to database '{kb_name}' successfully."}


# ---------------- 重排序模型：按需懒加载 ----------------
# ⚠️ 不要把 CrossEncoder(...) 放在模块导入时执行：
# 初始化要读取几千 MB 的权重，一旦写成模块级调用，**整个服务在模型就绪前都起不来**
# （连设置页、知识库管理页都打不开），对刚 clone 下来的人是硬门槛。
# 改为首次真正检索时才加载；加载失败也不让服务崩，而是降级为"只用混合检索结果"。
_rerank_lock = threading.RLock()
_rerank_state: dict = {"model": None, "error": None, "loaded": False}


def get_rerank_model():
    """获取重排序模型；不可用时返回 None（调用方据此降级）。

    返回 None 的两种情况：
    ① 模型文件不存在 / 加载报错（例如还没按 README 下载）；
    ② 加载成功过一次之后被缓存。
    """
    with _rerank_lock:
        if _rerank_state["loaded"]:
            return _rerank_state["model"]
        _rerank_state["loaded"] = True
        try:
            print(f"[rerank] 首次加载重排序模型: {RERANK_MODEL_PATH}")
            _rerank_state["model"] = CrossEncoder(RERANK_MODEL_PATH)
            _rerank_state["error"] = None
            print("[rerank] 加载完成")
        except Exception as e:
            _rerank_state["model"] = None
            _rerank_state["error"] = f"{type(e).__name__}: {e}"
            print(
                f"[rerank] ⚠️ 重排序模型加载失败，本次降级为仅用混合检索结果（不重排）: "
                f"{_rerank_state['error']}\n"
                f"         如需启用重排，请按 README「3. 重排序模型」下载到 {RERANK_MODEL_PATH}，"
                f"或用环境变量 RERANK_MODEL_PATH 指向已有目录。"
            )
        return _rerank_state["model"]


def rerank_status() -> dict:
    """当前重排序模型的状态（供 /knowledge/status 与排障使用）。"""
    with _rerank_lock:
        return {
            "rerank_model_path": RERANK_MODEL_PATH,
            "rerank_loaded": _rerank_state["model"] is not None,
            "rerank_error": _rerank_state["error"],
            "note": "首次检索时才会加载模型；加载失败不影响服务启动，检索会降级为不重排",
        }

# ---------------- search_kb 检索器缓存 ----------------
# 每次查询都重新 FAISS.load_local + BM25Retriever.from_documents 是大库慢/超时的根因。
# 改为按知识库缓存混合检索器，仅在 .faiss 索引文件的 mtime 变化时重建（上传/删除/更新文档后 mtime 自动变化）。
_search_cache: dict = {}


def _get_ensemble_retriever(kb_name, kb_path):
    """获取该知识库的混合检索器（FAISS + BM25），带 mtime 缓存。"""
    mtime = os.path.getmtime(kb_path)
    cached = _search_cache.get(kb_name)
    if cached and cached["mtime"] == mtime:
        return cached["ensemble"]

    vector_store = FAISS.load_local(kb_path, embedding_model,
                                    allow_dangerous_deserialization=True)
    faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    docs = list(vector_store.docstore._dict.values())
    bm25_retriever = BM25Retriever.from_documents(
        docs,
        preprocess_func=jieba.lcut_for_search,
    )
    bm25_retriever.k = 4
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
    )
    _search_cache[kb_name] = {"mtime": mtime, "ensemble": ensemble_retriever}
    print(f"[CACHE] 构建知识库「{kb_name}」检索器 (文档数={len(docs)})")
    return ensemble_retriever


@kb_router.post("/search_kb")
def search_kb(
    kb_name: str = Body(..., description="知识库名称"),
    query: str = Body(..., description="问题")
):
    # 注意：这里是同步 def（非 async）。FAISS 加载 / BM25 构建 / bge 重排都是 CPU 密集同步操作，
    # 若用 async def 会阻塞 uvicorn 事件循环，导致所有请求（包括 LLM 流式）一起变慢/超时。
    # FastAPI 对同步 def 会自动放入线程池执行，不阻塞其他请求。
    print("="*20)
    kb_name = _check_kb_name(kb_name)
    kb_path = _kb_index_path(kb_name)

    if not os.path.exists(kb_path):
        raise HTTPException(status_code=404, detail="Database not found")

    ensemble_retriever = _get_ensemble_retriever(kb_name, kb_path)
    print('============ensemble_retriever=======', ensemble_retriever)
    raw_docs = ensemble_retriever.invoke(query)

    # ----------------Rerank重排序（模型缺失时降级为不重排）----------------
    # 懒加载：模型在这里才第一次被读取；拿不到就只用检索结果，保证"能查得到东西"
    # 而不是整个接口报错——刚 clone 下来还没下载模型的人也能先把功能跑起来。
    reranker = get_rerank_model()
    reranked = reranker is not None
    warning = None

    if reranked:
        pairs = [[query, doc.page_content] for doc in raw_docs]
        scores = reranker.predict(pairs)
        doc_score_pairs = list(zip(raw_docs, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
    else:
        # 未重排：保持混合检索的原始顺序，分数置空
        doc_score_pairs = [(doc, None) for doc in raw_docs]
        warning = (
            "重排序模型未就绪，本次结果未经过 rerank（仅混合检索）。"
            f"原因：{_rerank_state['error']}。"
            f"请按 README 下载模型到 {RERANK_MODEL_PATH}，或设置环境变量 RERANK_MODEL_PATH。"
        )
        print(f"[search_kb] ⚠️ 降级返回（未重排）: {_rerank_state['error']}")

    contexts = {
        "results": [
            {
                "sources": doc.metadata.get("source", ""),
                "page_contents": doc.page_content,
                "rerank_score": (float(score) if score is not None else None),
            }
            for doc, score in doc_score_pairs
        ],
        "reranked": reranked,
    }
    if warning:
        contexts["warning"] = warning
    print('=========contexts======', contexts)
    return contexts


@kb_router.get("/status")
def knowledge_status():
    """服务与依赖的就绪状态，便于部署后自检（也可用于 CI 冒烟测试）。"""
    return {
        "ok": True,
        "routes": "see /docs",
        "rerank": rerank_status(),
        "kb_dir": os.path.abspath(KB_DIR),
        "file_storage_dir": os.path.abspath(FILE_STORAGE_DIR),
    }


class DeleteFileReq(BaseModel):
    kb_name: str
    file_name: str

@kb_router.post("/delete_file")
async def delete_file(req: DeleteFileReq):
    kb_name = req.kb_name
    file_name = req.file_name
    print(f"[BACKEND DEBUG] kb_name={repr(kb_name)}, file_name={repr(file_name)}")

    # 统一走安全路径解析：
    # 原实现用 abs_target.startswith(abs_storage) 判越界，存在「同前缀兄弟目录」绕过
    # （如 /app/data-evil 也以 /app/data 开头），且 file_name 未做 basename 处理。
    kb_path = _kb_index_path(kb_name)
    kb_file_storage_path = _kb_storage_dir(kb_name)
    target_file_path = _kb_file_path(kb_name, file_name)

    # 校验知识库faiss向量文件是否存在
    if not os.path.exists(kb_path):
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    if not os.path.exists(kb_file_storage_path):
        raise HTTPException(status_code=404, detail="Knowledge base file directory not found")

    if not os.path.isfile(target_file_path):
        raise HTTPException(status_code=404, detail=f"File {file_name} not found")

    try:
        os.remove(target_file_path)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")

    return {"msg": "file delete success"}


@kb_router.post("/list_files")
async def list_files(
    kb_name: str = Query(..., description="知识库名称")
):
    """获取指定知识库下全部磁盘文件列表"""
    target_dir = _kb_storage_dir(kb_name)
    if not os.path.exists(target_dir):
        return {"files": []}

    file_list = []
    for entry in os.listdir(target_dir):
        full_path = os.path.join(target_dir, entry)
        if os.path.isfile(full_path):
            file_list.append(entry)
    return {"files": file_list}


class ReadFileReq(BaseModel):
    kb_name: str
    file_name: str


@kb_router.post("/read_file")
async def read_file(req: ReadFileReq):
    """读取指定知识库内文件（笔记）的正文内容，用于复习梳理知识点"""
    file_path = _kb_file_path(req.kb_name, req.file_name)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail=f"File {req.file_name} not found")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Read file failed: {e}")
    return {"kb_name": req.kb_name, "file_name": req.file_name, "content": content}


class UpdateFileReq(BaseModel):
    kb_name: str
    file_name: str
    content: str
    chunk_size: int = 128
    chunk_overlap: int = 20


@kb_router.post("/update_file")
async def update_file(req: UpdateFileReq):
    """
    修改知识库内笔记文件：校验路径合法性 → 写回文件内容 → 整库重建向量索引。
    文件内容变化后必须重建 FAISS，避免旧向量残留导致检索到过期内容。
    """
    file_path = _kb_file_path(req.kb_name, req.file_name)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail=f"File {req.file_name} not found")

    content = req.content or ""
    if not content.strip():
        raise HTTPException(status_code=400, detail="笔记内容为空，无法保存")

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Write file failed: {e}")

    # 内容已变化：整库重建向量索引，保证检索到的是修改后的内容
    kb_path = _kb_index_path(req.kb_name)
    chunks = 0
    if os.path.exists(kb_path):
        try:
            chunks = _rebuild_kb(req.kb_name, kb_path, req.chunk_size, req.chunk_overlap)
        except Exception as e:
            print(f"[update_file] 重建知识库失败(不影响文件已保存): {e}")

    return {"kb_name": req.kb_name, "file_name": req.file_name, "rebuilt_chunks": chunks}

# ================= 对话问答笔记导出：知识库匹配与保存 =================
MATCH_THRESHOLD = 0.35  # 语义匹配阈值：高于该值判定该问题系列属于已有知识库


def _cosine_sim(vec_a, vec_b):
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _extract_topic_from_note(note: str) -> str:
    """从笔记中提取主题：优先取第一个问题，其次取最优提问模板，最后取开头片段"""
    if not note:
        return ""
    m = re.search(r"### Q：(.+)", note)
    if m:
        return m.group(1).strip()
    m = re.search(r"模板：(.+)", note)
    if m:
        return m.group(1).strip()
    return note.strip()[:60]


def _sanitize_kb_name(name: str, max_len: int = 30) -> str:
    """清洗知识库名称：去掉文件系统非法字符，保证可作为目录名。

    注意：除了替换非法字符，还必须剔除 `.`（避免生成 `.` / `..` 这类会被
    os.path.join 解释为「上级目录」的名字），最后再统一走 safe_kb_name 白名单校验，
    保证自动新建知识库时不会写出 ./data 之外。
    """
    name = re.sub(r'[\\/:*?"<>|\s]+', '', (name or "")).strip().strip('.')
    name = name[:max_len].strip().strip('.')
    try:
        return safe_kb_name(name)
    except UnsafePathError:
        return "对话问答笔记"


class MatchKbReq(BaseModel):
    topic: str


@kb_router.post("/match_kb")
def match_kb(req: MatchKbReq):
    """
    判断该问题系列是否属于已有知识库：
    对主题与各知识库（名称+描述）做 embedding 余弦相似度匹配，返回最高分知识库。
    同步 def：embedding 为同步 API 调用，放线程池避免阻塞事件循环。
    """
    topic = (req.topic or "").strip()
    if not topic:
        return {"kb_name": None, "score": 0.0}
    knowledge_list = queryall(session)
    if not knowledge_list:
        return {"kb_name": None, "score": 0.0}

    query_vec = embedding_model.embed_query(topic)
    best_name, best_score = None, -1.0
    for kb in knowledge_list:
        name = kb["name"] or ""
        desc = kb["description"] or ""
        # 匹配文本 = 名称 + 描述 + 库内文件名 + 少量内容片段（判断“问题系列属于哪个知识库”）
        # 注意控制长度：bge 等嵌入模型对输入 token 有上限
        kb_text = f"{name} {desc}".strip()
        # 名称来自数据库，仍做一次安全解析：即使历史数据里存了带 ../ 的名字也不会越界读取
        try:
            storage = _kb_storage_dir(name)
        except HTTPException:
            continue
        if os.path.isdir(storage):
            file_names = [f for f in os.listdir(storage) if os.path.isfile(os.path.join(storage, f))]
            kb_text += " " + " ".join(file_names[:5])
            for fn in file_names[:2]:
                try:
                    with open(os.path.join(storage, fn), "r", encoding="utf-8", errors="ignore") as f:
                        kb_text += " " + f.read(200)
                except Exception:
                    continue
        kb_text = kb_text[:500]
        kb_vec = embedding_model.embed_query(kb_text)
        score = _cosine_sim(query_vec, kb_vec)
        if score > best_score:
            best_name, best_score = name, score

    if best_score < MATCH_THRESHOLD:
        return {"kb_name": None, "score": best_score}
    return {"kb_name": best_name, "score": best_score}


class SaveNoteReq(BaseModel):
    file_name: str = "对话问答笔记.md"
    content: str
    kb_name: str = ""
    description: str = ""
    chunk_size: int = 128
    chunk_overlap: int = 20


def _rebuild_kb(kb_name, kb_path, chunk_size=128, chunk_overlap=20):
    """
    整库重建：遍历 ./data/<kb>/ 下全部文件重新切分、向量化，覆盖保存 FAISS 索引。
    用于笔记同名追加后内容变化的场景，避免旧向量与新向量重复召回。
    """
    storage = _kb_storage_dir(kb_name)
    files = sorted(f for f in os.listdir(storage) if not f.startswith('.'))
    docs = []
    for fn in files:
        fp = os.path.join(storage, fn)
        docs += data_loader().get_loader(fp).load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ".", ",", "，", "。", ""],
    )
    chunks = text_splitter.split_documents(docs)
    vs = FAISS.from_documents(chunks, embedding_model, normalize_L2=True)
    vs.save_local(kb_path)
    print(f"[_rebuild_kb] 重建知识库 {kb_name}: {len(chunks)} chunks, {len(files)} files")
    return len(chunks)


@kb_router.post("/save_note")
def save_note(req: SaveNoteReq):
    """
    将对话问答笔记保存到知识库：
    1) kb_name 为空时自动匹配已有知识库（属于已有知识库则直接保存），匹配不到则新建知识库；
    2) 知识库不存在则自动创建（FAISS 向量库 + sqlite 记录）；
    3) 笔记以 .md 文件形式保存到 ./data/<kb>/ 目录并向量化入库。
    同步 def：整库重建（_rebuild_kb）是重活，放线程池避免阻塞事件循环。
    """
    content = req.content or ""
    if not content.strip():
        raise HTTPException(status_code=400, detail="笔记内容为空，无法保存")

    # ---------- 1. 确定目标知识库 ----------
    target_kb = (req.kb_name or "").strip()
    created = False
    if not target_kb:
        # 自动判断：该问题系列是否属于已有知识库
        topic = _extract_topic_from_note(content)
        matched = match_kb(MatchKbReq(topic=topic))
        if matched.get("kb_name"):
            target_kb = matched["kb_name"]
        else:
            # 新建知识库：名称由主题清洗生成，避免与已有重名
            target_kb = _sanitize_kb_name(topic)
            existing_names = {kb["name"] for kb in queryall(session)}
            if target_kb in existing_names:
                target_kb = f"{target_kb}{int(time.time()) % 10000}"
            created = True
    # 无论来自用户指定还是自动匹配，统一做白名单校验：名称会作为目录名使用
    target_kb = _check_kb_name(target_kb)

    # ---------- 2. 知识库不存在则自动创建 ----------
    kb_path = _kb_index_path(target_kb)
    if not os.path.exists(kb_path):
        doc = Document(page_content="init", metadata={})
        vector_store = FAISS.from_documents([doc], embedding_model, normalize_L2=True)
        ids = list(vector_store.docstore._dict.keys())
        vector_store.delete(ids)
        vector_store.save_local(kb_path)
        msg = add(session, target_kb, req.description or f"对话问答笔记自动创建于 {time.strftime('%Y-%m-%d %H:%M')}")
        print(f"[save_note] 创建知识库: {target_kb} -> {msg}")
        created = True

    # ---------- 3. 保存笔记文件并向量化 ----------
    storage_path = _kb_storage_dir(target_kb)
    os.makedirs(storage_path, exist_ok=True)

    # 文件命名：{用户输入笔记名}_{知识库名}_{日期}.md（日期取保存当天 YYYYMMDD）
    # 用户传入的笔记名先清洗为纯文件名，避免其中的 `/`、`..` 影响后续拼接
    raw_name = os.path.splitext(safe_filename(req.file_name or "", fallback=""))[0].strip().strip('.')
    if not raw_name:
        raw_name = "对话问答笔记"

    # 同名笔记检测：按基础名 {笔记名}_{库名} 匹配（文件名含日期后缀，需剥离日期匹配）
    # 匹配规则：笔记名_库名_YYYYMMDD[_HHMMSS].md
    import re as _re
    base_pattern = _re.compile(rf"^{_re.escape(raw_name)}_{_re.escape(target_kb)}_\d{{8}}(?:_\d{{6}})?\.md$")
    matched_files = [f for f in os.listdir(storage_path) if base_pattern.match(f)]

    appended = False
    if matched_files:
        # 同名 → 合并：读最近一次的同名笔记，追加新内容并自动标注追加日期，覆盖写回原文件
        # （保留原文件名与首次创建日期，便于复习对照时间线）
        latest = max(matched_files, key=lambda f: os.path.getmtime(os.path.join(storage_path, f)))
        with open(os.path.join(storage_path, latest), "r", encoding="utf-8") as f:
            old = f.read()
        content = old + f"\n\n---\n## 📅 追加于 {time.strftime('%Y-%m-%d %H:%M')}\n\n" + content
        safe_file = latest
        file_path = resolve_within(storage_path, latest)
        appended = True
    else:
        # 无同名 → 新建：笔记名_库名_日期.md（同日再同名时追加时分秒，防止覆盖）
        safe_file = f"{raw_name}_{target_kb}_{time.strftime('%Y%m%d')}.md"
        file_path = resolve_within(storage_path, safe_file)
        if os.path.exists(file_path):
            stem, ext = os.path.splitext(safe_file)
            safe_file = f"{stem}_{time.strftime('%H%M%S')}{ext}"
            file_path = resolve_within(storage_path, safe_file)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    # 与 upload_docs 相同的向量化逻辑
    if appended:
        # 同名追加后文件内容已变化：整库重建，避免新旧内容在 FAISS 中重复召回
        _rebuild_kb(target_kb, kb_path, req.chunk_size, req.chunk_overlap)
    else:
        loader = data_loader().get_loader(file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
            separators=["\n\n", "\n", " ", ".", ",", "，", "。", ""],
        )
        chunks = text_splitter.split_documents(docs)
        vectorstore = FAISS.load_local(kb_path, embedding_model, allow_dangerous_deserialization=True)
        vectorstore.add_documents(chunks)
        vectorstore.save_local(kb_path)

    print(f"[save_note] 笔记已保存: kb={target_kb}, file={safe_file}, created={created}, appended={appended}")
    return {"kb_name": target_kb, "file_name": safe_file, "created": created, "appended": appended}


class QuizReq(BaseModel):
    kb_name: str = Body(..., description="出题的知识库名称")
    num_questions: int = Body(5, description="题目数量")
    question_type: str = Body("混合", description="题型：混合/选择题/填空题/判断题/简答题")
    topic: str = Body("", description="出题方向/主题，可留空则覆盖全部内容")


@knowledgeapp.post('/quiz')
async def quiz(req: QuizReq):
    """根据知识库内容生成练习题（出题模式）：
    1) 校验知识库存在；2) 读取知识库全部文档文本；3) LLM 基于内容出题（Markdown）。
    """
    kb_name = (req.kb_name or "").strip()
    if not kb_name:
        raise HTTPException(status_code=400, detail="请先选择知识库")
    kb_name = _check_kb_name(kb_name)
    kb_path = _kb_index_path(kb_name)
    if not os.path.exists(kb_path):
        raise HTTPException(status_code=404, detail="Database not found")

    # ---------- 1. 收集知识库文档全文 ----------
    target_dir = _kb_storage_dir(kb_name)
    chunks = []
    if os.path.isdir(target_dir):
        for entry in os.listdir(target_dir):
            fp = os.path.join(target_dir, entry)
            if os.path.isfile(fp):
                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        chunks.append(f"【{entry}】\n{f.read()}")
                except Exception:
                    continue
    if not chunks:
        raise HTTPException(status_code=400, detail="该知识库暂无文档内容，无法出题")

    content = "\n\n".join(chunks)
    if len(content) > 8000:
        content = content[:8000] + "\n…（内容过长，已截断前 8000 字符）"

    # ---------- 2. LLM 出题 ----------
    # 模型与 Key 在调用时从 runtime_config 读，页面改完立刻生效
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model=runtime_config.get("chat_model"),
        api_key=runtime_config.require_api_key("chat"),
        base_url=runtime_config.base_url_for("chat") or base_url,
        temperature=0.7,
        max_tokens=700,
        streaming=False,
        request_timeout=300,
    )
    topic_line = f"，重点围绕「{req.topic}」" if (req.topic or "").strip() else ""
    prompt = (
        f"你是出题老师。请根据以下「{kb_name}」知识库的内容，出 {req.num_questions} 道{req.question_type}题{topic_line}。\n\n"
        "要求：\n"
        "1. 题目必须基于给定内容，不得编造内容中不存在的事实；\n"
        "2. 选择题需要给出 A/B/C/D 选项；判断题需说明正确或错误及依据；填空题留空用____标注；简答题给出要点；\n"
        "3. 每道题包含：题目、答案、简要解析（解析说明依据了哪部分内容）；\n"
        "4. 使用 Markdown 格式，按 1. 2. 3. 编号输出，不要输出多余的开场白。\n\n"
        f"知识库「{kb_name}」内容：\n{content}"
    )
    try:
        import asyncio
        result = await asyncio.to_thread(llm.invoke, prompt)
        quiz_text = result.content if hasattr(result, "content") else str(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"出题生成失败: {type(e).__name__}: {e}")

    if not quiz_text or not quiz_text.strip():
        raise HTTPException(status_code=500, detail="出题生成结果为空，请重试")

    # 出题只负责生成题目；错题保存已由前端 save_quiz_notes 复用 /knowledge/save_note 完成（见 webui/Quiz_UI.py）
    return {"kb_name": kb_name, "quiz": quiz_text.strip()}
