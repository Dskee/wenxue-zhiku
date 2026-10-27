"""CI 用：用仓库里提交的公版语料建一个知识库，并生成可复现的题子集。

为什么单独写这个而不是直接跑全量：
- 40 题里的「Agent 面试问答」部分来自私有笔记，不入库也不提交，克隆者拿不到；
- 但 `eval/corpus/唐诗三百首.md` 是公版语料，已提交，所以 CI 能重建这一半，
  再用数据集里 gold_source 指向它的题组成子集，跑出一份别人也能复现的数字。

用法（仓库根目录，需 SILICONFLOW_API_KEY）：
    python -m eval.ci.build_ci_kb --kb ci-tangshi-300 --out /tmp/subset.jsonl
然后：
    python -m eval.run_ablation --kb ci-tangshi-300 --dataset /tmp/subset.jsonl \
        --configs eval/configs/vector_only.toml eval/configs/hybrid.toml \
        --out /tmp/ci-report.md --no-generate
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def reset_kb(kb_name: str) -> None:
    """同名库先整个清掉再重建，保证反复跑和全新跑的结果一致。

    不能用 `/knowledge/delete_file`：那个接口只删磁盘上的文件、不重建 FAISS 索引
    （实测删掉再传，同一个库的 chunk 数从 385 变 770），而 `upload_docs` 是追加语义。
    所以这里清三样：索引目录、语料目录、元数据行。
    """
    import shutil
    import sqlite3

    from configs.setting import FILE_STORAGE_DIR, KB_DIR
    from utils.paths import resolve_within, safe_kb_name

    name = safe_kb_name(kb_name)
    index_dir = Path(resolve_within(KB_DIR, f"{name}.faiss"))
    storage_dir = Path(resolve_within(FILE_STORAGE_DIR, name))
    db_path = Path("knowledge_base.db")

    existed = index_dir.exists() or storage_dir.exists()
    shutil.rmtree(index_dir, ignore_errors=True)
    shutil.rmtree(storage_dir, ignore_errors=True)
    if db_path.exists():
        con = sqlite3.connect(db_path)
        try:
            cur = con.execute("delete from knowledge where name = ?", (name,))
            con.commit()
            existed = existed or cur.rowcount > 0
        finally:
            con.close()
    if existed:
        print(f"已清理旧库：{name}")


def build_kb(kb_name: str, corpus_path: Path, chunk_size: int, chunk_overlap: int) -> None:
    """走真实的 HTTP 路由建库，不绕过业务代码另写一套入库逻辑。"""
    from fastapi.testclient import TestClient

    import FastAPIRouter

    reset_kb(kb_name)
    client = TestClient(FastAPIRouter.app)

    response = client.post(
        "/knowledge/add_knowledge",
        json={"name": kb_name, "description": "CI 评测库：唐诗三百首（公版语料）"},
    )
    if response.status_code != 200:
        raise SystemExit(f"建库失败 {response.status_code}: {response.text}")

    content = corpus_path.read_bytes()
    response = client.post(
        "/knowledge/upload_docs",
        files=[("files", (corpus_path.name, content, "text/markdown"))],
        data={
            "kb_name": kb_name,
            "chunk_size": str(chunk_size),
            "chunk_overlap": str(chunk_overlap),
        },
    )
    if response.status_code != 200:
        raise SystemExit(f"上传失败 {response.status_code}: {response.text}")
    print(f"已入库：{corpus_path.name} -> {kb_name}（chunk {chunk_size}/{chunk_overlap}）")


def write_subset(dataset_path: Path, out_path: Path, corpus_name: str) -> int:
    """只保留 gold 落在公版语料里的题；其它题的 gold 在私有笔记里，CI 复现不了。"""
    items = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    subset = [item for item in items if item.get("gold_source") == corpus_name]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in subset), encoding="utf-8"
    )
    print(f"子集 {len(subset)} / {len(items)} 条（gold_source == {corpus_name}）")
    if not subset:
        raise SystemExit("子集为空：数据集的 gold_source 和语料文件名对不上")
    return len(subset)


def main() -> int:
    parser = argparse.ArgumentParser(description="CI：建公版语料知识库 + 生成题子集")
    parser.add_argument("--corpus", default="eval/corpus/唐诗三百首.md")
    parser.add_argument("--kb", default="ci-tangshi-300")
    parser.add_argument("--dataset", default="eval/dataset/v1.0.jsonl")
    parser.add_argument("--out", default="ci-out/subset.jsonl")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--chunk-overlap", type=int, default=20)
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        raise SystemExit(f"语料不存在: {corpus_path}")

    build_kb(args.kb, corpus_path, args.chunk_size, args.chunk_overlap)
    write_subset(Path(args.dataset), Path(args.out), corpus_path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())