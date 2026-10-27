"""检索配置：eval/configs/*.toml → 项目里的 RetrievalConfig。

三份配置对应三个消融点（BM25 / 重排 / 都不开），同一份代码只换参数，不做代码分支。
读 TOML 用 tomllib（3.11+）或 tomli（3.10 需安装），所以本模块不依赖项目重型依赖；
真正构造 RetrievalConfig 时才 import 项目模块。
"""

import importlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
PROJECT_CONFIG_MODULE = "knowledgebase_server.loader.KnowledgeFastAPI"

RETRIEVAL_KEYS = (
    "use_bm25",
    "use_reranker",
    "top_k_vector",
    "top_k_bm25",
    "top_k_final",
)


class ConfigError(ValueError):
    """配置文件不合法（缺字段、多字段、类型不对）。"""


def load_config_file(path) -> Dict:
    path = Path(path)
    with open(path, "rb") as handle:
        raw = tomllib.load(handle)

    unknown = set(raw) - {"name", "description", "retrieval"}
    if unknown:
        raise ConfigError(f"{path.name}: 未知字段 {sorted(unknown)}")

    name = raw.get("name")
    if not name:
        raise ConfigError(f"{path.name}: 缺少 name")
    if name != path.stem:
        raise ConfigError(f"{path.name}: name={name!r} 与文件名不一致")

    retrieval = raw.get("retrieval")
    if not isinstance(retrieval, dict):
        raise ConfigError(f"{path.name}: 缺少 [retrieval] 段")

    missing = [key for key in RETRIEVAL_KEYS if key not in retrieval]
    if missing:
        raise ConfigError(f"{path.name}: [retrieval] 缺少 {missing}")
    unknown = set(retrieval) - set(RETRIEVAL_KEYS)
    if unknown:
        raise ConfigError(f"{path.name}: [retrieval] 未知字段 {sorted(unknown)}")

    return {
        "name": name,
        "description": raw.get("description", ""),
        "retrieval": {key: retrieval[key] for key in RETRIEVAL_KEYS},
    }


def load_configs(paths: Optional[Sequence] = None) -> List[Dict]:
    """不传 paths 时按文件名顺序加载 configs/ 下全部 .toml。"""
    files = [Path(item) for item in paths] if paths else sorted(CONFIG_DIR.glob("*.toml"))
    if not files:
        raise ConfigError(f"{CONFIG_DIR} 下没有找到任何 *.toml")
    return [load_config_file(path) for path in files]


def to_retrieval_config(spec: Dict):
    """配置字典 → 项目 RetrievalConfig（数据类本身在业务模块里，这里不做镜像）。"""
    module = importlib.import_module(PROJECT_CONFIG_MODULE)
    constructor = getattr(module, "RetrievalConfig", None)
    if constructor is None:
        raise ConfigError(f"{PROJECT_CONFIG_MODULE} 里没有 RetrievalConfig，P1 改造未生效？")
    return constructor(**spec["retrieval"])