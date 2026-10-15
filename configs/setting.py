"""全局配置。

安全约定：**任何密钥（API Key）都不得写死在源码里**，一律从环境变量或项目根目录的
`.env` 文件读取。这样源码可以安全地提交到 GitHub，而密钥只留在本机。

使用方式（二选一）：
1) 复制 `.env.example` 为 `.env`，填入自己的 Key（推荐本地开发）；
2) 直接用环境变量注入（Docker / CI 推荐）：
       export SILICONFLOW_API_KEY=sk-xxxx
       export SERPAPI_KEY=xxxx
"""
import os
from pathlib import Path

# 项目根目录（configs/ 的上一级），用于定位 .env
BASE_DIR = Path(__file__).resolve().parent.parent
_ENV_FILE = BASE_DIR / ".env"


def _load_dotenv(path: Path) -> None:
    """把 .env 中的键值对载入 os.environ。

    优先使用 python-dotenv；若环境里没有该库，则退回内置的极简解析器，
    保证「零额外依赖也能启动」。已存在的真实环境变量优先，不被 .env 覆盖。
    """
    if not path.is_file():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(path, override=False)
        return
    except ImportError:
        pass
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(_ENV_FILE)


def _get(name: str, default: str = "") -> str:
    """读取环境变量并去掉首尾空白（空串视为未设置，返回默认值）。"""
    return (os.environ.get(name) or "").strip() or default


def _get_int(name: str, default: int) -> int:
    try:
        return int(_get(name, str(default)))
    except ValueError:
        return default


def _get_list(name: str, default):
    """读取逗号分隔的列表型配置。"""
    raw = _get(name)
    if not raw:
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


# ==================== 数据库 / 目录 ====================
# 知识库信息数据库路径
SQLALCHEMY_DATABASE = './db_server/data/'

# 知识库信息数据库的uri（历史遗留配置，实际元数据库见 db_server/data/KnowledgeBase.py）
SQLALCHEMY_DATABASE_URI = 'sqlite:///./db_server/data/info.db'

# 向量数据库的路径
KB_DIR = _get('KB_DIR', './knowledgebases')

# 存储上传知识库的文档
FILE_STORAGE_DIR = _get('FILE_STORAGE_DIR', './data')

# 本地 embedding 模型目录（当前走远程 API 嵌入，此路径为备用）
embedding_model_path = _get('EMBEDDING_MODEL_PATH', './models/AI-ModelScope/bge-large-zh-v1.5')

# Rerank 重排序模型目录
RERANK_MODEL_PATH = _get('RERANK_MODEL_PATH', './models/models/bge-reranker-v2-m3')

# 大模型聊天中上传文件存储路径
TEMP_FILE_STORAGE_DIR = _get('TEMP_FILE_STORAGE_DIR', './temp/data')

# 媒体文件保存路径
MEDIA_DIR = _get('MEDIA_DIR', './temp/medias')

# ==================== 大模型（云端或本地 OpenAI 兼容服务） ====================
# 密钥只能来自环境变量 / .env，源码中不出现
# 模型来源：siliconflow（硅基流动云端）/ local（本地 Ollama、LM Studio、vLLM）/ custom（自定义）
PROFILE = _get('PROFILE', 'siliconflow')
api_key = _get('SILICONFLOW_API_KEY')
base_url = _get('SILICONFLOW_BASE_URL', 'https://api.siliconflow.cn/v1')
chat_model_name = _get('CHAT_MODEL_NAME', 'Qwen/Qwen3-8B')
embedding_model_name = _get('EMBEDDING_MODEL_NAME', 'BAAI/bge-large-zh-v1.5')

# 向量化可选独立配置：留空表示"跟随对话的地址与 Key"
# （常见需求：对话用本地模型、向量化仍用云端，避免本地缺少 embedding 模型）
EMBEDDING_BASE_URL = _get('EMBEDDING_BASE_URL')
EMBEDDING_API_KEY = _get('EMBEDDING_API_KEY')

# ==================== 联网搜索（SerpAPI） ====================
# 未配置时 web_search 工具会自动跳过 SerpAPI，直接走必应抓取回退通道
serpapi_key = _get('SERPAPI_KEY')

# ==================== 服务与网络 ====================
# 本地 FastAPI 后端地址（前端 Gradio 与 Agent 工具统一从这里取，变更部署时只需改这一处）
BACKEND_URL = _get('BACKEND_URL', 'http://127.0.0.1:16055')

# 设置超时（Agent 任务/工具调用最长等待秒数；LLM 生成较慢时需放宽，否则会强杀导致无输出）
TIME_OUT = _get_int('TIME_OUT', 300)

# CORS 白名单：只允许前端所在来源，避免任意站点跨域调用后端
CORS_ORIGINS = _get_list('CORS_ORIGINS', [
    'http://127.0.0.1:7860',
    'http://localhost:7860',
])

# 单次上传文件大小上限（MB）
MAX_UPLOAD_MB = _get_int('MAX_UPLOAD_MB', 50)

# 可选的接口访问令牌：留空 = 不校验（本地单机默认）。
# 一旦设置，后端所有 /chat、/knowledge 接口都必须带 X-API-Token 请求头。
API_TOKEN = _get('API_TOKEN')

# 前端请求统一携带的鉴权头（未配置 API_TOKEN 时为空字典，不影响现有行为）
AUTH_HEADERS = {'X-API-Token': API_TOKEN} if API_TOKEN else {}


def require_api_key() -> str:
    """在真正调用大模型前校验密钥，缺失时给出可操作的报错而不是静默失败。"""
    if not api_key:
        raise RuntimeError(
            "未配置 SILICONFLOW_API_KEY。请在项目根目录创建 .env 文件"
            "（可直接复制 .env.example 并填写），或设置同名环境变量后重启服务。"
        )
    return api_key


if not api_key:
    import sys
    print(
        "[configs.setting] ⚠️ 未检测到 SILICONFLOW_API_KEY：请复制 .env.example 为 .env 并填写密钥，"
        "否则所有大模型调用都会失败。",
        file=sys.stderr,
    )
if not serpapi_key:
    import sys
    print(
        "[configs.setting] ℹ️ 未检测到 SERPAPI_KEY：web_search 工具将直接使用必应抓取回退通道。",
        file=sys.stderr,
    )
