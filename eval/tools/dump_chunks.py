"""导出知识库全部 chunk（id / 来源 / 正文）为 JSONL，用于人工标注 gold。

chunk_id 由 `eval.chunks` 按「文件名 + 正文」内容寻址计算，与 FAISS 内部 UUID 无关，
所以知识库重建后 id 依然对得上——数据集里的 gold_chunk_ids 才能长期有效。

用法（容器内、仓库根目录）：
    /root/miniconda3/envs/ai_rag/bin/python -m eval.tools.dump_chunks \
        --kb 评测基准 --out /tmp/chunks.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.chunks import chunk_id_of, source_name  # noqa: E402

DEFAULT_KB_DIR = Path("./knowledgebases")


class _NoEmbeddings:
    """读已有索引不需要向量化，这里给一个会报错的占位，避免误触发 embedding 调用。"""

    def embed_documents(self, texts):
        raise RuntimeError("dump_chunks 只读索引，不重新向量化")

    def embed_query(self, text):
        raise RuntimeError("dump_chunks 只读索引，不重新向量化")


def resolve_kb_path(kb_dir: Path, kb_name: str):
    """索引目录是 `<知识库名>.faiss`，和线上的 `_kb_index_path()` 保持一致。"""
    from utils.paths import safe_kb_name

    name = safe_kb_name(kb_name)  # 拒绝 .. / 路径分隔符
    candidate = kb_dir / f"{name}.faiss"
    return candidate if candidate.is_dir() else None


def load_documents(kb_path: Path):
    from langchain_community.vectorstores import FAISS

    store = FAISS.load_local(
        str(kb_path),
        _NoEmbeddings(),
        allow_dangerous_deserialization=True,  # 本地自建索引，非外部输入
    )
    return list(store.docstore._dict.values())


def main() -> int:
    parser = argparse.ArgumentParser(description="导出知识库 chunk 清单")
    parser.add_argument("--kb", required=True, help="知识库名，如 评测基准")
    parser.add_argument("--kb-dir", default=str(DEFAULT_KB_DIR), help="知识库目录")
    parser.add_argument("--out", required=True, help="输出 JSONL 路径")
    args = parser.parse_args()

    kb_path = resolve_kb_path(Path(args.kb_dir), args.kb)
    if kb_path is None:
        print(f"知识库不存在: {Path(args.kb_dir) / args.kb}.faiss", file=sys.stderr)
        return 2

    documents = load_documents(kb_path)
    rows = []
    for index, document in enumerate(documents):
        metadata = getattr(document, "metadata", None) or {}
        rows.append(
            {
                "chunk_id": chunk_id_of(document),
                "source": source_name(str(metadata.get("source", ""))),
                "chars": len(document.page_content or ""),
                "text": document.page_content or "",
                "order": index,
            }
        )
    rows.sort(key=lambda row: (row["source"], row["order"]))

    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    per_source = {}
    for row in rows:
        per_source[row["source"]] = per_source.get(row["source"], 0) + 1
    unique = len({row["chunk_id"] for row in rows})
    print(f"chunks={len(rows)} unique_ids={unique} -> {out_path}")
    for source, count in sorted(per_source.items()):
        print(f"  {source}: {count}")
    if unique != len(rows):
        print("警告：存在内容重复的 chunk，chunk_id 冲突", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())