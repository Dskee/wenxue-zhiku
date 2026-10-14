"""设置接口：读取 / 保存模型与 API Key、选择模型来源、拉取可用模型列表。

挂载前缀 `/settings`。设计要点：
- **不回传明文 Key**：只回传是否已配置 + 打码值；
- 保存后**立即生效**（写入 runtime_config，无需重启），同时写回 `.env` 持久化；
- **模型来源**支持硅基流动（云端）/ 本地（Ollama、LM Studio、vLLM）/ 自定义，
  本地服务通常不需要 Key；
- 模型列表从对应来源的 OpenAI 兼容 `/models` 接口实时拉取；本地与云端用的是同一套逻辑，
  所以"自己把地址和模型名粘进来"也能用。
"""
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from configs import runtime_config

settingsapp = APIRouter(prefix="/settings", tags=["settings"])

# 拉不到列表时的兜底候选，保证界面不至于空白
FALLBACK_CHAT_MODELS = [
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/DeepSeek-R1",
    # 本地常见（Ollama 命名）
    "qwen2.5:7b",
    "qwen2.5:14b",
    "llama3.1:8b",
    "gemma2:9b",
]
FALLBACK_EMBEDDING_MODELS = [
    "BAAI/bge-large-zh-v1.5",
    "BAAI/bge-m3",
    # 本地常见
    "bge-m3",
    "nomic-embed-text",
    "quentinz/bge-large-zh-v1.5",
]

# 用于把模型按用途粗分：命中这些关键词的视为向量化/重排模型，不算对话模型
_NON_CHAT_HINTS = ("bge-", "embedding", "embed", "reranker", "rerank", "gte-", "bce-", "nomic-embed")


class SettingsReq(BaseModel):
    """保存设置的请求体。

    字段留空 / 不传 = 该项不修改；要显式清空某项（如取消向量化独立配置）请用 `clear`。
    """
    profile: Optional[str] = None
    chat_model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    embedding_model: Optional[str] = None
    embedding_base_url: Optional[str] = None
    embedding_api_key: Optional[str] = None
    clear: Optional[List[str]] = None


_CHANGE_NAMES = {
    "profile": "模型来源",
    "chat_model": "对话模型",
    "api_key": "API Key",
    "base_url": "接口地址",
    "embedding_model": "向量化模型",
    "embedding_base_url": "向量化接口地址",
    "embedding_api_key": "向量化 API Key",
}


@settingsapp.get("")
def read_settings():
    """读取当前设置（Key 已打码）。"""
    return runtime_config.snapshot()


@settingsapp.get("/profiles")
def list_profiles():
    """返回可选的模型来源预设（地址、默认模型名、提示文案）。"""
    return runtime_config.profiles()


@settingsapp.post("")
def write_settings(req: SettingsReq):
    """保存设置：校验 → 立即生效 → 写回 .env。"""
    try:
        result = runtime_config.update(
            clear=req.clear or (),
            profile=req.profile,
            chat_model=req.chat_model,
            api_key=req.api_key,
            base_url=req.base_url,
            embedding_model=req.embedding_model,
            embedding_base_url=req.embedding_base_url,
            embedding_api_key=req.embedding_api_key,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    changed = result.get("changed") or []
    snapshot = runtime_config.snapshot()

    if changed:
        msg = "已保存并立即生效：" + "、".join(_CHANGE_NAMES.get(c, c) for c in changed)
    else:
        msg = "没有检测到改动（留空的项表示不修改）"

    return {"msg": msg, "changed": changed, "settings": snapshot, "env_path": result.get("env_path")}


@settingsapp.get("/models")
def list_models(kind: str = "chat"):
    """从当前来源拉取可用模型列表（OpenAI 兼容 `GET /models`）。

    `kind=chat|embedding`：向量化可以配独立的地址与 Key，所以要区分。
    失败时返回兜底候选 + `error`，前端仍可手动输入模型名。
    """
    kind = "embedding" if kind == "embedding" else "chat"
    base_url = runtime_config.base_url_for(kind)
    key = runtime_config.api_key_for(kind)

    fallback = FALLBACK_EMBEDDING_MODELS if kind == "embedding" else FALLBACK_CHAT_MODELS
    empty_result = {
        "chat_models": [] if kind == "embedding" else fallback,
        "embedding_models": fallback if kind == "embedding" else [],
        "kind": kind,
        "source": "fallback",
        "error": None,
    }

    if not base_url:
        empty_result["error"] = "尚未配置接口地址，请先在「设置」里填写"
        return empty_result

    url = base_url.rstrip("/") + "/models"
    headers = {"Accept": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    try:
        import requests
        resp = requests.get(url, headers=headers, timeout=30)
    except Exception as e:
        empty_result["error"] = f"请求 {url} 失败：{type(e).__name__}: {e}"
        return empty_result

    if resp.status_code != 200:
        empty_result["error"] = (
            f"{url} 返回 HTTP {resp.status_code}：{resp.text[:200]}"
        )
        return empty_result

    try:
        payload = resp.json()
        ids = [str(item.get("id", "")).strip() for item in (payload.get("data") or [])]
    except Exception as e:
        empty_result["error"] = f"解析模型列表失败：{type(e).__name__}: {e}"
        return empty_result

    ids = sorted({i for i in ids if i})
    embedding_models = [i for i in ids if any(h in i.lower() for h in _NON_CHAT_HINTS)]
    chat_models = [i for i in ids if i not in set(embedding_models)]

    return {
        "chat_models": chat_models,
        "embedding_models": embedding_models,
        "kind": kind,
        "total": len(ids),
        "source": "provider",
        "base_url": base_url,
        "error": None,
    }


class TestReq(BaseModel):
    """测试连接：用当前设置真实发一次最小请求。"""
    kind: str = "chat"


@settingsapp.post("/test")
def test_connection(req: TestReq):
    """真实调用一次模型，验证「地址 + Key + 模型名」这套组合是否可用。

    这是排查本地模型最有效的手段（最常见的问题就是端口填错、模型没 pull）。
    """
    kind = "embedding" if req.kind == "embedding" else "chat"
    base_url = runtime_config.base_url_for(kind)
    key = runtime_config.api_key_for(kind)
    model = runtime_config.get("embedding_model" if kind == "embedding" else "chat_model")

    if not base_url:
        return {"ok": False, "detail": "尚未配置接口地址"}
    if not model:
        return {"ok": False, "detail": "尚未配置模型名"}

    try:
        if kind == "embedding":
            # 与项目实际的向量化路径保持一致（CustomEmbeddingModel 内部就是
            # client.create(model=..., input=...)），避免用 OpenAIEmbeddings 的默认模型
            # 导致误报 "Model does not exist"。
            from openai import OpenAI
            client = OpenAI(
                api_key=key or runtime_config.PLACEHOLDER_KEY,
                base_url=base_url,
                timeout=60,
            )
            resp = client.embeddings.create(model=model, input="连接测试")
            vec = resp.data[0].embedding
            return {
                "ok": True,
                "detail": f"向量化成功：模型 {model}，维度 {len(vec)}",
                "base_url": base_url,
                "model": model,
            }
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=model,
            api_key=key or runtime_config.PLACEHOLDER_KEY,
            base_url=base_url,
            temperature=0,
            max_tokens=16,
            request_timeout=60,
        )
        out = llm.invoke("只回复两个字：正常")
        text = getattr(out, "content", str(out))
        return {
            "ok": True,
            "detail": f"对话成功：模型 {model}，返回 {text[:60]!r}",
            "base_url": base_url,
            "model": model,
        }
    except Exception as e:
        status = getattr(e, "status_code", None)
        msg = str(e)
        lower = msg.lower()

        # 按错误类型给对应的排查提示，别一律甩"本地模型排查"（Key 无效时说这个会误导）
        if status in (401, 403):
            hint = (
                "API Key 无效或已过期。请到「⚙️ 设置」页重新填写有效的 Key 并保存；"
                "云端 Key 可在 https://cloud.siliconflow.cn/account/ak 重新生成。"
                "若用的是本地模型，本地服务通常不需要 Key，请把「模型来源」切换为本地。"
            )
        elif status == 404 or "does not exist" in lower or "model not found" in lower:
            hint = (
                f"模型「{model}」在该服务上不存在。请点「🔄 刷新对话模型 / 刷新向量模型」"
                "从当前来源拉取真实可用的模型名，而不是手输猜测的名字。"
            )
        elif status == 429:
            hint = "请求过于频繁或账户额度不足，稍后重试或检查账户余额。"
        elif "connection" in type(e).__name__.lower() or "connect" in lower:
            hint = (
                "连不上这个地址。本地模型排查：① 服务是否已启动（Ollama: ollama serve）"
                "② 模型是否已下载（ollama pull 模型名）"
                "③ 容器内地址要用 host.docker.internal 而不是 127.0.0.1"
                "④ 端口是否对应（Ollama 11434 / LM Studio 1234 / vLLM 8000）"
            )
        else:
            hint = "请核对「模型来源 / 接口地址 / 模型名 / API Key」四项是否匹配。"

        return {
            "ok": False,
            "detail": f"{type(e).__name__}: {msg[:300]}",
            "base_url": base_url,
            "model": model,
            "hint": hint,
        }
