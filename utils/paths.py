"""路径安全工具：统一收敛「用户输入  文件系统路径」的校验，防止路径穿越。

威胁模型：知识库名称、文件名、上传文件名、session_id 全部来自 HTTP 请求（前端或直接
调用 API 的人都可以伪造）。若直接拼进 os.path.join，攻击者可用 `../` 读写甚至删除项目
目录之外的文件。本模块提供三个收敛点：

- safe_filename：把单个文件名清洗成不含路径分隔符的安全名字；
- safe_kb_name：校验知识库名称（同时拒绝 `..`，因为它会被拼成目录名）；
- resolve_within：把目标路径解析为绝对真实路径，并确认它确实在允许的根目录内。
"""
import os
import re
import unicodedata

# 允许的知识库名称：中英文、数字、下划线、连字符、点、空格，长度 1~50
_KB_NAME_RE = re.compile(r"^[\w\u4e00-\u9fff][\w\u4e00-\u9fff.\- ]{0,49}$", re.UNICODE)

# 文件名中不允许出现的字符（含 Windows 保留字符与路径分隔符）
_UNSAFE_CHARS_RE = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


class UnsafePathError(ValueError):
    """路径校验失败（调用方应转换为 HTTP 400）。"""


def safe_filename(name: str, fallback: str = "unnamed", max_len: int = 120) -> str:
    """把任意字符串清洗为「单个文件名」，绝不包含目录成分。

    - 去掉任何目录部分（同时处理 Windows 反斜杠，os.path.basename 在 Linux 下不认 `\\`）；
    - 替换文件系统非法字符；
    - 去掉首尾的点和空白，避免生成 `.`、`..`、`.hidden` 之类；
    - 结果为空时使用 fallback。
    """
    raw = str(name or "")
    # 统一分隔符后再取末段，防止在 Linux 下被 `.` / `\` 绕过
    raw = raw.replace("\\", "/")
    raw = unicodedata.normalize("NFC", raw)
    base = raw.split("/")[-1].split("\\")[-1]
    base = _UNSAFE_CHARS_RE.sub("_", base).strip().strip(".")
    # 再次确认不含 `..`（清洗后仍可能出现形如 a..b 的名字，作为普通文件名是安全的）
    if base in ("", ".", ".."):
        return fallback
    return base[:max_len]


def safe_kb_name(name: str) -> str:
    """校验知识库名称，非法则抛 UnsafePathError。

    知识库名会被用作目录名（./data/<kb>）与索引文件名（./knowledgebases/<kb>.faiss），
    所以必须严格白名单：拒绝路径分隔符、`..`、以及任何控制字符。
    """
    value = str(name or "").strip()
    if not value:
        raise UnsafePathError("知识库名称不能为空")
    if ".." in value or "/" in value or "\\" in value:
        raise UnsafePathError("知识库名称包含非法字符")
    if not _KB_NAME_RE.match(value):
        raise UnsafePathError("知识库名称仅支持中英文、数字、下划线、连字符、点和空格，且不超过 50 个字符")
    return value


def resolve_within(root: str, *parts: str) -> str:
    """把 parts 拼到 root 下，返回绝对真实路径，并确保结果未逃出 root。

    使用 realpath + 前缀比较（带上 os.sep），避免 `startswith("/a/data")` 这种
    被同前缀兄弟目录（`/a/data-evil`）绕过的经典缺陷。
    """
    root_abs = os.path.realpath(root)
    target = os.path.realpath(os.path.join(root_abs, *parts))
    if target != root_abs and not target.startswith(root_abs + os.sep):
        raise UnsafePathError("非法的路径访问")
    return target
