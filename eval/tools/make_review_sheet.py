"""从 v1.0.jsonl 生成人工核验清单（Markdown）。

为什么需要：gold 是评测的地基，不能只由模型或脚本认定。清单把「题目 / 答案片段 / 出处」
摊平成一页，方便逐条打勾，不用去翻 JSONL。

用法：python eval/tools/make_review_sheet.py [--dataset eval/dataset/v1.0.jsonl] [--out eval/dataset/v1.0.review.md]
"""

import argparse
import hashlib
import json
from pathlib import Path

TYPE_LABEL = {"normal": "常规", "boundary": "边界", "no_answer": "拒答"}


def md5_of(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def render(items, dataset_name: str, digest: str) -> str:
    counts = {}
    for item in items:
        counts[item["type"]] = counts.get(item["type"], 0) + 1

    lines = [
        "# 评测集 v1.0 人工核验清单",
        "",
        f"- 数据集：`{dataset_name}`（md5 `{digest}`，{len(items)} 条："
        + " / ".join(f"{k} {v}" for k, v in counts.items())
        + "）",
        "- 核验要点：① 题目是否能从语料里答出来；② 「答案片段」是否**逐字**出现在语料里；"
        "③ 出处文件是否正确；④ 拒答题是否真的在语料里查不到。",
        "- 答案片段是**子串匹配**用的锚点，不需要覆盖完整答案，但必须能在原文里找到（标点空白不计）。",
        "- 只有一处 gold 的题，命中该片段就算检索成功；多处 gold 的题按 gold 集合算 Recall。",
        "",
    ]

    for kind in ("normal", "boundary", "no_answer"):
        group = [item for item in items if item["type"] == kind]
        if not group:
            continue
        lines += [f"## {TYPE_LABEL[kind]}题（{len(group)} 条）", ""]
        lines += ["| 编号 | 题目 | 答案片段（应在语料中逐字出现） | 出处 | 核验 |", "| --- | --- | --- | --- | --- |"]
        for item in group:
            evidences = "<br>".join(item.get("gold_evidence") or []) or "（应无答案）"
            source = item.get("gold_source") or "—"
            lines.append(f"| {item['id']} | {item['question']} | {evidences} | {source} | ☐ |")
        lines.append("")

    lines += [
        "## 怎么核验最快",
        "",
        "1. 逐条读「答案片段」，在心里或编辑器里对语料搜一下（`eval/corpus/唐诗三百首.md` 是公版那半，"
        "另一半是私有笔记，在 `data/评测基准/agent面试问答.md`）。",
        "2. 觉得题目和片段对不上，就改 `eval/dataset/specs/v1.0.json` 里对应条目的 `evidence` 再重建：",
        "",
        "```bash",
        "python -m eval.tools.build_dataset --spec v1.0 --out eval/dataset/v1.0.jsonl",
        "```",
        "",
        "3. 重建后 md5 会变，记得更新 `eval/dataset/CHANGELOG.md` 和报告里的数据集指纹。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="生成评测集人工核验清单")
    parser.add_argument("--dataset", default="eval/dataset/v1.0.jsonl")
    parser.add_argument("--out", default="eval/dataset/v1.0.review.md")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    items = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    out_path = Path(args.out)
    out_path.write_text(render(items, dataset_path.as_posix(), md5_of(dataset_path)), encoding="utf-8")
    print(f"核验清单已写入 {out_path}（{len(items)} 条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())