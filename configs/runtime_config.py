"""运行时可修改的配置：模型来源 / 对话模型 / API Key / API 地址 / 向量化模型。

与 `configs/setting.py` 的分工：
- `setting.py` 是**启动时的静态配置**：进程启动时从环境变量 / `.env` 读一次；
- 本模块是**页面上随时可改的活配置**：
   改动立刻在内存生效，**不需要重启服务**（正在跑的请求也会用到新值）；
   同时写回项目根目录的 `.env`，**重启后依然是新值**。

关于「模型来源（profile）」：
- `siliconflow`：硅基流动云端，必须配 API Key；
- `local`：本地 OpenAI 兼容服务（Ollama / LM Studio / vLLM），通常**不需要 Key**；
- `custom`：自定义，地址与模型名全靠手填。

`version()` 是单调递增的版本号：Key / 地址 / 模型一旦变化，依赖它们的对象
（如 embedding 客户端）就据此判断需要重建缓存。
"""
import os
import re
import threading
from pathlib import Path

from configs import setting

_ENV_PATH: Path = Path(setting.BASE_DIR) / ".env"

# 本地服务不校验 Key，但 OpenAI 客户端要求 api_key 非空，所以给一个占位串
PLACEHOLDER_KEY = "local-no-key-needed"

# 模型来源预设：选中后前端用它自动填好地址与默认模型名，用户仍可自由覆盖
PROFILES = {
    "siliconflow": {
        "label": "硅基流动（云端）",
        "base_url": "https://api.siliconflow.cn/v1",
        "default_chat": "Qwen/Qwen3-8B",
        "default_embedding": "BAAI/bge-large-zh-v1.5",
        "needs_key": True,
        "hint": "云端服务，需要 API Key；点「刷新对话模型 / 刷新向量模型」会从你的硅基流动账号实时拉取可用模型。",
    },
    "local": {
        "label": "本地模型（Ollama / LM Studio / vLLM）",
        "base_url": "http://host.docker.internal:11434/v1",
        "default_chat": "qwen2.5:7b",
        "default_embedding": "bge-m3",
        "needs_key": False,
        "hint": (
            "本地 OpenAI 兼容服务，通常不需要 Key。地址注意：应用跑在 Docker 里时必须用 "
            "host.docker.internal（127.0.0.1 指的是容器自己）；Ollama=11434、LM Studio=1234、vLLM=8000。"
        ),
    },
    "custom": {
        "label": "自定义（手动填地址与模型名）",
        "base_url": "",
        "default_chat": "",
        "default_embedding": "",
        "needs_key": True,
        "hint": "任何 OpenAI 兼容服务都可以：把接口地址与模型名直接粘贴到下面的输入框。",
    },
}
DEFAULT_PROFILE = "siliconflow"

# 内存中的活配置（初值来自 .env / 环境变量）
_state = {
    "profile": setting.PROFILE or DEFAULT_PROFILE,
    "chat_model": setting.chat_model_name,
    "api_key": setting.api_key,
    "base_url": setting.base_url,
    "embedding_model": setting.embedding_model_name,
    # 向量化可选独立配置：留空表示"跟随对话的地址/Key"
    "embedding_base_url": setting.EMBEDDING_BASE_URL,
    "embedding_api_key": setting.EMBEDDING_API_KEY,
}

# 状态字段 .env 中的变量名
_ENV_KEYS = {
    "profile": "PROFILE",
    "chat_model": "CHAT_MODEL_NAME",
    "api_key": "SILICONFLOW_API_KEY",
    "base_url": "SILICONFLOW_BASE_URL",
    "embedding_model": "EMBEDDING_MODEL_NAME",
    "embedding_base_url": "EMBEDDING_BASE_URL",
    "embedding_api_key": "EMBEDDING_API_KEY",
}

_MODEL_NAME_RE = re.compile(r"^[\w.\-:]+(/[\w.\-:]+)?$")

_lock = threading.RLock()
_version = 0


# ==================== 读取 ====================

def get(name: str, default: str = "") -> str:
    """读取当前生效的值。"""
    with _lock:
        return _state.get(name) or default


def version() -> int:
    """当前配置版本号；变化即代表 Key / 地址 / 模型被改过。"""
    with _lock:
        return _version


def profiles() -> dict:
    """给前端的来源预设（含默认地址、默认模型名、提示文案）。"""
    return {"default": DEFAULT_PROFILE, "profiles": PROFILES}


class MissingApiKeyError(RuntimeError):
    """未配置必需的 API Key（由 FastAPIRouter 统一转成 503 + 中文说明）。"""


class ProviderUnavailableError(RuntimeError):
    """模型服务不可用：Key 无效、模型不存在、网络不通等（同样统一转成 503）。"""


def needs_key() -> bool:
    """当前来源是否必须配 Key（本地服务通常不需要）。"""
    return bool(PROFILES.get(get("profile") or DEFAULT_PROFILE, {}).get("needs_key", True))


def base_url_for(kind: str = "chat") -> str:
    """取实际使用的接口地址。kind='embedding' 时可被独立配置覆盖。"""
    if kind == "embedding":
        return get("embedding_base_url") or get("base_url")
    return get("base_url")


def api_key_for(kind: str = "chat") -> str:
    """取实际使用的 Key；本地服务且未配 Key 时返回占位串（OpenAI 客户端要求非空）。"""
    if kind == "embedding":
        key = get("embedding_api_key") or get("api_key")
    else:
        key = get("api_key")
    if key:
        return key
    return "" if needs_key() else PLACEHOLDER_KEY


def mask_key(key: str) -> str:
    """把 Key 打码后用于展示：sk-abcd****wxyz。"""
    key = (key or "").strip()
    if not key:
        return ""
    if len(key) <= 12:
        return key[:3] + "****"
    return f"{key[:7]}****{key[-4:]}"


def snapshot() -> dict:
    """给前端的设置快照：**不回传真实 Key**，只回传是否已配置 + 打码值。"""
    with _lock:
        key = _state.get("api_key") or ""
        emb_key = _state.get("embedding_api_key") or ""
        profile = _state.get("profile") or DEFAULT_PROFILE
        return {
            "profile": profile,
            "profile_label": PROFILES.get(profile, {}).get("label", profile),
            "chat_model": _state.get("chat_model") or "",
            "embedding_model": _state.get("embedding_model") or "",
            "base_url": _state.get("base_url") or "",
            "embedding_base_url": _state.get("embedding_base_url") or "",
            "embedding_uses_own_config": bool(_state.get("embedding_base_url") or emb_key),
            "needs_key": needs_key(),
            "has_key": bool(key),
            "api_key_masked": mask_key(key),
            "has_embedding_key": bool(emb_key),
            "embedding_api_key_masked": mask_key(emb_key),
        }


def require_api_key(kind: str = "chat") -> str:
    """在真正调用模型前校验 Key，缺失时抛出可读错误（由 FastAPIRouter 转成 503）。"""
    key = api_key_for(kind)
    if not key:
        what = "向量化（embedding）" if kind == "embedding" else "大模型"
        raise MissingApiKeyError(
            f"未配置 API Key，无法调用{what}服务。"
            "请在页面左侧「🔑 API Key」输入框或「⚙️ 设置」页填写后保存"
            "（会写入项目根目录的 .env，重启后依然有效）。"
            "如果你用的是本地模型，请把「模型来源」切换为本地 —— 本地服务通常不需要 Key。"
        )
    return key


# ==================== 写入 ====================

def _validate(name: str, value: str) -> str:
    value = (value or "").strip()
    if name in ("base_url", "embedding_base_url"):
        if not re.match(r"^https?://", value):
            raise ValueError("接口地址必须以 http:// 或 https:// 开头")
        return value.rstrip("/")
    if name == "profile":
        if value not in PROFILES:
            raise ValueError(f"未知的模型来源：{value}")
        return value
    if name in ("chat_model", "embedding_model"):
        if not value:
            raise ValueError("模型名不能为空")
        if not _MODEL_NAME_RE.match(value):
            raise ValueError(f"模型名格式不合法：{value}（示例：Qwen/Qwen3-8B、qwen2.5:7b、gpt-4o-mini）")
        return value
    if name in ("api_key", "embedding_api_key"):
        if len(value) < 8:
            raise ValueError("API Key 看起来太短，请检查是否复制完整")
        if any(ch.isspace() for ch in value):
            raise ValueError("API Key 中不能包含空格或换行")
        return value
    return value


def update(clear=(), **changes) -> dict:
    """更新设置：校验  内存生效  写回 .env。

    - `changes` 中值为 None 或空字符串表示"该项不修改"（前端密码框留空即不覆盖旧 Key）；
    - `clear` 里列出的字段会被显式清空（例如取消"向量化独立配置"）。
    返回 {"changed": [...], "env_path": "..."}。
    """
    global _version
    applied = {}
    for name, raw in changes.items():
        if name not in _state or raw is None:
            continue
        value = str(raw).strip()
        if value == "":
            continue  # 留空 = 不修改
        applied[name] = _validate(name, value)

    for name in clear or ():
        if name in _state:
            applied[name] = ""

    if not applied:
        return {"changed": [], "env_path": str(_ENV_PATH)}

    with _lock:
        changed = []
        for name, value in applied.items():
            if (_state.get(name) or "") != value:
                _state[name] = value
                changed.append(name)
        if changed:
            _version += 1

    if changed:
        _write_env({_ENV_KEYS[n]: _state[n] for n in changed})

    return {"changed": changed, "env_path": str(_ENV_PATH)}


def _write_env(updates: dict) -> None:
    """把键值写回 .env：已存在的行就地替换，不存在的追加；**保留注释与其他配置**。"""
    try:
        lines = _ENV_PATH.read_text(encoding="utf-8").splitlines() if _ENV_PATH.is_file() else []
    except OSError:
        lines = []

    pending = dict(updates)
    out = []
    for line in lines:
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            out.append(line)
            continue
        key = line.split("=", 1)[0].strip()
        if key in pending:
            out.append(f"{key}={pending.pop(key)}")
        else:
            out.append(line)

    if pending:
        if out and out[-1].strip():
            out.append("")
        out.append("# 由「设置」页面自动写入")
        for key, value in pending.items():
            out.append(f"{key}={value}")

    _ENV_PATH.write_text("\n".join(out) + "\n", encoding="utf-8")
    # 同步到本进程的环境变量，方便其他按 os.environ 读取的代码也拿到新值
    for key, value in updates.items():
        os.environ[key] = value
