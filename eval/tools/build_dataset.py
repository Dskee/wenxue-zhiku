"""用「证据片段」反查 chunk id，把 specs/*.json 编译成可冻结的数据集 JSONL。

为什么要多这一步：gold 如果手抄 chunk id，语料一改就全错，而且没人能核对标注是否正确。
specs 里写的是「答案在语料里逐字长什么样」，构建脚本负责把它翻译成内容寻址的 chunk id，
于是每条标注都能被机械校验：片段找不到 = 语料变了或标注写错了，构建直接失败。

用法（容器内、仓库根目录）：
    python -m eval.tools.build_dataset --spec v1.0 --out eval/dataset/v1.0.jsonl

需要能读到知识库索引（默认 ./knowledgebases/<kb_name>.faiss）。
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.chunks import chunk_id_of, normalize_text, source_name  # noqa: E402
from eval.tools.dump_chunks import resolve_kb_path  # noqa: E402

SPEC_DIR = Path(__file__).resolve().parents[1] / "dataset" / "specs"
LONG_OVERLAP_WARN = 20
LONG_OVERLAP_FAIL = 40


def load_index(kb_dir: Path, kb_name: str):
    from eval.tools.dump_chunks import load_documents

    kb_path = resolve_kb_path(kb_dir, kb_name)
    if kb_path is None:
        raise SystemExit(f"知识库不存在: {kb_dir / (kb_name + '.faiss')}")
    records = []
    for document in load_documents(kb_path):
        metadata = getattr(document, "metadata", None) or {}
        records.append(
            {
                "chunk_id": chunk_id_of(document),
                "source": source_name(str(metadata.get("source", ""))),
                "normalized": normalize_text(document.page_content),
            }
        )
    return records


def longest_common_substring(left: str, right: str) -> str:
    """只用来提示「问题是不是把答案原句抄进去了」，不需要最优算法。"""
    best = ""
    for start in range(len(left)):
        if len(left) - start <= len(best):
            break
        end = start + len(best) + 1
        while end <= len(left) and left[start:end] in right:
            best = left[start:end]
            end += 1
    return best


def build(spec: dict, records, strict: bool = True):
    by_normalized = [(record["normalized"], record) for record in records]
    items = []
    problems = []
    warnings = []

    seen_ids = set()
    for raw in spec["items"]:
        item_id = raw["id"]
        item_type = raw["type"]
        if item_id in seen_ids:
            problems.append(f"{item_id}: id 重复")
        seen_ids.add(item_id)

        expected_prefix = {"normal": "n", "boundary": "b", "no_answer": "x"}[item_type]
        if not item_id.startswith(expected_prefix):
            problems.append(f"{item_id}: type={item_type} 的 id 前缀应为 {expected_prefix}")

        evidence = raw.get("evidence") or []
        gold_ids = []
        gold_sources = set()
        matched_evidence = []

        for snippet in evidence:
            needle = normalize_text(snippet)
            if not needle:
                problems.append(f"{item_id}: 证据片段为空")
                continue
            hits = [record for normalized, record in by_normalized if needle in normalized]
            if not hits:
                problems.append(f"{item_id}: 证据片段在语料里找不到 → {snippet!r}")
                continue
            matched_evidence.append(snippet)
            for hit in hits:
                gold_ids.append(hit["chunk_id"])
                gold_sources.add(hit["source"])

        gold_ids = list(dict.fromkeys(gold_ids))

        if item_type == "no_answer":
            if gold_ids or evidence:
                problems.append(f"{item_id}: no_answer 不应带 gold")
            declared = raw.get("source")
            if declared:
                problems.append(f"{item_id}: no_answer 不应声明 source")
        else:
            if not gold_ids:
                problems.append(f"{item_id}: 没有解析出任何 gold chunk")
            declared = raw.get("source")
            if declared and gold_sources and declared not in gold_sources:
                problems.append(
                    f"{item_id}: 声明的 source={declared} 与实际 {sorted(gold_sources)} 不一致"
                )
            if len(gold_ids) > 1:
                warnings.append(f"{item_id}: 多 gold（{len(gold_ids)} 个 chunk 含该答案）")

        # 问题与 gold 正文的最长公共子串：太长说明在抄原文，检索会虚高
        if gold_ids:
            gold_text = "".join(
                record["normalized"] for record in records if record["chunk_id"] in set(gold_ids)
            )
            common = longest_common_substring(normalize_text(raw["question"]), gold_text)
            if len(common) >= LONG_OVERLAP_FAIL:
                problems.append(f"{item_id}: 问题与 gold 正文重复 {len(common)} 字 → {common!r}")
            elif len(common) >= LONG_OVERLAP_WARN:
                warnings.append(f"{item_id}: 问题与 gold 正文重复 {len(common)} 字 → {common!r}")

        items.append(
            {
                "id": item_id,
                "type": item_type,
                "question": raw["question"],
                "gold_chunk_ids": gold_ids,
                "gold_source": sorted(gold_sources)[0] if gold_sources else None,
                "gold_evidence": matched_evidence,
                "notes": raw.get("notes", ""),
            }
        )

    if problems and strict:
        raise SystemExit("数据集校验失败：\n" + "\n".join(problems))
    return items, problems, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="specs → dataset JSONL")
    parser.add_argument("--spec", default="v1.0", help="specs 文件名（不含 .json）")
    parser.add_argument("--kb-dir", default="./knowledgebases")
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", default="", help="校验报告输出路径")
    parser.add_argument("--allow-missing", action="store_true", help="有校验错误也写出文件")
    args = parser.parse_args()

    spec_path = SPEC_DIR / f"{args.spec}.json"
    if not spec_path.exists():
        raise SystemExit(f"spec 不存在: {spec_path}")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))

    records = load_index(Path(args.kb_dir), spec["kb_name"])
    items, problems, warnings = build(spec, records, strict=not args.allow_missing)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(item, ensure_ascii=False) for item in items) + "\n"
    out_path.write_text(payload, encoding="utf-8")
    digest = hashlib.md5(payload.encode("utf-8")).hexdigest()

    per_type = {}
    for item in items:
        per_type[item["type"]] = per_type.get(item["type"], 0) + 1
    gold_chunks = len({cid for item in items for cid in item["gold_chunk_ids"]})

    lines = [
        f"spec={args.spec} version={spec['version']} kb={spec['kb_name']}",
        f"indexed_chunks={len(records)}",
        f"items={len(items)} {per_type}",
        f"distinct_gold_chunks={gold_chunks}",
        f"md5={digest}",
        f"out={out_path}",
        "",
        "warnings:",
    ] + [f"  - {line}" for line in warnings]
    lines += ["", "problems:"] + [f"  - {line}" for line in problems]
    report = "\n".join(lines) + "\n"
    if args.report:
        Path(args.report).write_text(report, encoding="utf-8")
    print(report, end="")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())