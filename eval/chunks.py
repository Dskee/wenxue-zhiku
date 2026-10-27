"""稳定的 chunk id。

FAISS 里每个 chunk 的 key 是 LangChain 建的 UUID，重建知识库就变；metadata 只有 source，
没有行号。评测集如果直接记 UUID，换一次切分参数整份 gold 就作废。所以这里定义一个
内容寻址的 id：

    sha1(文件名 + "\\x00" + 归一化正文)[:16]

同一份文件、同样的切分参数重建，id 不变；文件改名才会变（这是有意的，见下）。
"""

import hashlib
import re
from typing import Iterable, List

WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """压掉所有空白差异（换行/缩进/CRLF），只保留可见字符。"""
    return WHITESPACE_RE.sub("", text or "")


def source_name(source: str) -> str:
    """只取文件名，避免绝对路径 / 相对路径写法不同导致 id 不稳定。"""
    return (source or "").replace("\\", "/").rsplit("/", 1)[-1]


def chunk_id(source: str, text: str) -> str:
    payload = f"{source_name(source)}\x00{normalize_text(text)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def chunk_id_of(document) -> str:
    """Document（langchain）→ chunk id。"""
    metadata = getattr(document, "metadata", None) or {}
    return chunk_id(str(metadata.get("source", "")), document.page_content)


def chunk_ids_of(documents: Iterable) -> List[str]:
    return [chunk_id_of(document) for document in documents]


def dedupe_keep_order(ids: Iterable[str]) -> List[str]:
    """去重但保留首次出现的顺序；排序指标里重复条目只会白占名次。"""
    seen = set()
    ordered = []
    for item in ids:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered