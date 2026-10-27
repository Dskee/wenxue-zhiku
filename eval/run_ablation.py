"""检索消融实验：同一份代码 + 同一份数据集，只换配置跑三种召回方案。

跑法（容器内、仓库根目录，需先起好知识库）：
    python -m eval.run_ablation --kb 评测基准 --dataset eval/dataset/v1.0.jsonl \
        --out results/2026-10-27.md

指标全部是规则计算（字符串/名次），不调用大模型当裁判；生成层只用来测拒答和证据命中，
温度固定 0，prompt 固定，见 eval/prompts.py。
"""

import argparse
import datetime as dt
import gc
import json
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval import configs as eval_configs  # noqa: E402
from eval.chunks import chunk_ids_of, dedupe_keep_order, normalize_text  # noqa: E402
from eval.metrics import (  # noqa: E402
    evaluate_evidence_query,
    evaluate_query,
    mean,
    percentile,
)
from eval.prompts import is_refusal  # noqa: E402

ANSWERABLE = ("normal", "boundary")
# 主指标：证据片段是否出现在 top-k 上下文里（不受 128 字切块边界影响）
EVIDENCE_METRICS = ("evidence_recall@5", "evidence_mrr@5", "evidence_ndcg@5")
# 对照指标：整块命中（chunk id 精确匹配），受切块边界影响，数值会偏低
CHUNK_METRICS = ("recall@5", "mrr@5", "ndcg@5")


def git_revision() -> str:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return head + ("（工作区有未提交改动）" if dirty else "")
    except Exception:
        return "unknown"


def load_dataset(path: Path):
    items = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if not items:
        raise SystemExit(f"数据集为空: {path}")
    return items


def dataset_md5(path: Path) -> str:
    import hashlib

    return hashlib.md5(path.read_bytes()).hexdigest()


def run_retrieval(module, kb_name: str, items, retrieval_config, k: int, limit: int = 0):
    rows = []
    selected = items[:limit] if limit else items
    for index, item in enumerate(selected, start=1):
        started = time.perf_counter()
        error = ""
        documents = []
        try:
            result = module.retrieve(
                kb_name,
                item["question"],
                config=retrieval_config,
                strict_reranker=retrieval_config.use_reranker,
            )
            documents = list(result.docs)
        except Exception as exc:  # 单题失败不能拖垮整轮，但要记下来
            error = f"{type(exc).__name__}: {exc}"
        elapsed_ms = (time.perf_counter() - started) * 1000

        ids = dedupe_keep_order(chunk_ids_of(documents))
        texts = [document.page_content for document in documents]
        gold = item["gold_chunk_ids"]
        row = {
            "id": item["id"],
            "type": item["type"],
            "question": item["question"],
            "retrieved_ids": ids[:k],
            "gold_ids": gold,
            "retrieved_count": len(ids),
            "latency_ms": round(elapsed_ms, 1),
            "error": error,
        }
        if item["type"] in ANSWERABLE:
            row.update(evaluate_query(ids, gold, k))
            row.update(evaluate_evidence_query(texts, item["gold_evidence"], k))
        rows.append(row)
        print(
            f"    [{index}/{len(selected)}] {item['id']} "
            f"ev_rank={row.get('evidence_first_rank') or '-'} "
            f"chunk_rank={row.get('first_hit_rank') or '-'} {elapsed_ms:.0f}ms"
            + (f" ERROR {error}" if error else ""),
            flush=True,
        )
    return rows


def run_generation(module, items, rows_by_id, retrieval_config, k: int):
    from eval.generation import build_chat_model, generate_answer
    from eval.prompts import format_contexts

    model = build_chat_model()
    for index, item in enumerate(items, start=1):
        row = rows_by_id[item["id"]]
        documents = row.pop("_documents", [])
        answer = ""
        error = ""
        elapsed = 0.0
        try:
            answer, elapsed = generate_answer(model, item["question"], documents)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        refused = is_refusal(answer)
        gold_text = "".join(normalize_text(snippet) for snippet in item["gold_evidence"])
        row.update(
            {
                "answer": answer,
                "answer_chars": len(answer),
                "refused": refused,
                "generation_ms": round(elapsed * 1000, 1),
                "generation_error": error,
                "evidence_in_answer": bool(gold_text) and any(
                    normalize_text(snippet) in normalize_text(answer)
                    for snippet in item["gold_evidence"]
                ),
            }
        )
        print(
            f"    [{index}/{len(items)}] {item['id']} refused={refused} "
            f"{len(answer)}字 {elapsed:.1f}s" + (f" ERROR {error}" if error else ""),
            flush=True,
        )
    return rows_by_id


def summarize(rows, generate: bool):
    answerable = [row for row in rows if row["type"] in ANSWERABLE]
    no_answer = [row for row in rows if row["type"] == "no_answer"]

    summary = {
        "items": len(rows),
        "answerable": len(answerable),
        "no_answer": len(no_answer),
        "retrieval_errors": sum(1 for row in rows if row["error"]),
        "generation_errors": sum(1 for row in rows if row.get("generation_error")),
        "latency_ms_p50": round(percentile([row["latency_ms"] for row in rows], 50), 1),
        "latency_ms_p95": round(percentile([row["latency_ms"] for row in rows], 95), 1),
        "retrieved_mean": round(mean([row["retrieved_count"] for row in rows]), 2),
    }
    for metric in EVIDENCE_METRICS + CHUNK_METRICS:
        summary[metric] = round(mean([row[metric] for row in answerable]), 4)
    summary["by_type"] = {
        item_type: {
            "count": len([row for row in rows if row["type"] == item_type]),
            **{
                metric: round(mean([r[metric] for r in rows if r["type"] == item_type]), 4)
                for metric in EVIDENCE_METRICS
            },
        }
        for item_type in ANSWERABLE
    }

    if generate:
        answered = [row for row in answerable if row.get("answer") is not None]
        summary["refusal_rate_answerable"] = round(
            mean([1.0 if row["refused"] else 0.0 for row in answered]), 4
        ) if answered else None
        summary["refusal_accuracy_no_answer"] = round(
            mean([1.0 if row["refused"] else 0.0 for row in no_answer]), 4
        ) if no_answer else None
        non_refused = [row for row in answered if not row["refused"]]
        summary["evidence_hit_rate"] = round(
            mean([1.0 if row["evidence_in_answer"] else 0.0 for row in non_refused]), 4
        ) if non_refused else None
        # 检索已经把答案片段带进上下文、模型却仍然拒答 —— 这一格才是生成层自己的问题，
        # 和「检索没找到所以只能拒答」分开看，不然检索差会全部记到生成层头上。
        covered = [row for row in answered if row.get("evidence_first_rank")]
        summary["refusal_rate_answerable_covered"] = round(
            mean([1.0 if row["refused"] else 0.0 for row in covered]), 4
        ) if covered else None
        summary["covered_count"] = len(covered)
        summary["generation_ms_p50"] = round(
            percentile([row["generation_ms"] for row in answered], 50), 1
        ) if answered else None
    return summary


def render_report(meta, results) -> str:
    lines = [
        f"# 检索消融实验报告（{meta['date']}）",
        "",
        "> 本文件由 `python -m eval.run_ablation` 生成，数字全部来自本次实跑，未做任何手工修饰。",
        "",
        "## 一、实验设置",
        "",
        f"- 代码版本：`{meta['git']}`",
        f"- 数据集：`{meta['dataset']}`（md5 `{meta['dataset_md5']}`，{meta['items']} 条："
        f"normal {meta['counts']['normal']} / boundary {meta['counts']['boundary']} / "
        f"no_answer {meta['counts']['no_answer']}）",
        f"- 知识库：`{meta['kb']}`，{meta['chunks'] if meta['chunks'] else '未知'} 个 chunk，来源 {', '.join(meta['corpus'])}",
        f"- 生成模型：`{meta['chat_model']}`，temperature=0，max_tokens=512，固定单轮 prompt",
        f"- 主指标：证据覆盖 Recall@{meta['k']} / MRR@{meta['k']} / nDCG@{meta['k']}"
        "（答案片段是否出现在 top-k 上下文的某段正文里，与切块边界无关，只看 normal+boundary）",
        f"- 对照指标：块级 Recall@{meta['k']}（整块 chunk id 精确匹配，切块边界会拉低它）",
        "- 生成层：误拒率（该答却拒答）、拒答正确率（该拒就拒）、gold 证据命中率；全部字符串规则，零 LLM 裁判。"
        "误拒率另给一列「检索已覆盖答案片段时」的条件值，用来把检索的锅和生成的锅分开",
        f"- 运行环境：{meta['python']}",
        "",
        "## 二、配置对比",
        "",
        "| 配置 | 说明 | 纯向量 | BM25 | 重排 | top_k_vector | top_k_bm25 | top_k_final |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for spec in meta["config_specs"]:
        retrieval = spec["retrieval"]
        lines.append(
            f"| `{spec['name']}` | {spec['description']} | "
            f"{'否' if not retrieval['use_bm25'] else '是'} | "
            f"{'是' if retrieval['use_bm25'] else '否'} | "
            f"{'是' if retrieval['use_reranker'] else '否'} | "
            f"{retrieval['top_k_vector']} | {retrieval['top_k_bm25']} | {retrieval['top_k_final']} |"
        )

    lines += [
        "",
        "## 三、总表",
        "",
        "| 配置 | 证据 Recall@5 | 证据 MRR@5 | 证据 nDCG@5 | 块级 Recall@5 | 误拒率 | 检索已覆盖时误拒率 | 证据命中率 | 检索 p50 | 检索 p95 | 生成 p50 | 错误 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, summary in results:
        lines.append(
            f"| `{name}` | {summary['evidence_recall@5']:.3f} | {summary['evidence_mrr@5']:.3f} | "
            f"{summary['evidence_ndcg@5']:.3f} | {summary['recall@5']:.3f} | "
            f"{fmt_pct(summary.get('refusal_rate_answerable'))} | "
            f"{fmt_pct(summary.get('refusal_rate_answerable_covered'))} | "
            f"{fmt_pct(summary.get('evidence_hit_rate'))} | "
            f"{summary['latency_ms_p50']:.0f} ms | {summary['latency_ms_p95']:.0f} ms | "
            f"{fmt_ms(summary.get('generation_ms_p50'))} | "
            f"{summary['retrieval_errors'] + summary['generation_errors']} |"
        )

    lines += [
        "",
        "两套口径一起列，是因为 128 字的切块把同一段答案切进了相邻两块：只按 chunk id 判分时，"
        "检索到「答案的前半段」也算没命中，配置之间的差异会被切块边界噪声吃掉。"
        "证据口径问的是「上下文里有没有这句话」，更接近 RAG 真正关心的事；块级口径保留，"
        "用来观察索引本身有没有取到同一块。",
        "",
        "## 四、分类型",
        "",
        "| 配置 | 类型 | 条数 | 证据 Recall@5 | 证据 MRR@5 | 证据 nDCG@5 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, summary in results:
        for item_type in ANSWERABLE:
            block = summary["by_type"][item_type]
            lines.append(
                f"| `{name}` | {item_type} | {block['count']} | {block['evidence_recall@5']:.3f} | "
                f"{block['evidence_mrr@5']:.3f} | {block['evidence_ndcg@5']:.3f} |"
            )

    lines += [
        "",
        "## 五、逐题首命中名次（证据口径，- 表示 top-5 内没覆盖到答案片段）",
        "",
        "| 题目 | 类型 | " + " | ".join(f"`{name}`" for name, _ in results) + " |",
        "| --- | --- | " + " | ".join("---" for _ in results) + " |",
    ]
    by_config = meta["per_item"]
    item_meta = meta["item_meta"]
    for item_id in item_meta["order"]:
        info = item_meta["items"][item_id]
        cells = []
        for name, _ in results:
            row = by_config[name][item_id]
            rank = row.get("evidence_first_rank")
            cells.append(str(rank) if rank else "-")
        lines.append(f"| {item_id} | {info['type']} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## 六、拒答明细（no_answer 8 条）",
        "",
        "| 题目 | " + " | ".join(f"`{name}` 是否拒答" for name, _ in results) + " |",
        "| --- | " + " | ".join("---" for _ in results) + " |",
    ]
    for item_id in item_meta["order"]:
        if item_meta["items"][item_id]["type"] != "no_answer":
            continue
        cells = []
        for name, _ in results:
            row = by_config[name][item_id]
            if row.get("generation_error"):
                cells.append("报错")
            else:
                cells.append("是" if row.get("refused") else "否")
        lines.append(f"| {item_id} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## 七、结论",
        "",
        meta["conclusion"],
        "",
        "## 八、复现方式",
        "",
        "```bash",
        "# 1. 建评测知识库（需 SILICONFLOW_API_KEY）",
        "python -m eval.tools.dump_chunks --kb 评测基准 --out /tmp/chunks.jsonl   # 仅供人工核对",
        "python -m eval.tools.build_dataset --spec v1.0 --out eval/dataset/v1.0.jsonl",
        "# 2. 跑三种配置（含生成层）",
        "python -m eval.run_ablation --kb 评测基准 --dataset eval/dataset/v1.0.jsonl \\",
        "    --out results/2026-10-27.md",
        "```",
        "",
    ]
    return "\n".join(lines)


def fmt_pct(value):
    return "-" if value is None else f"{value * 100:.0f}%"


def fmt_ms(value):
    return "-" if value is None else f"{value:.0f} ms"


def main() -> int:
    parser = argparse.ArgumentParser(description="检索消融实验")
    parser.add_argument("--kb", default="评测基准")
    parser.add_argument("--dataset", default="eval/dataset/v1.0.jsonl")
    parser.add_argument("--configs", nargs="+", default=None)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out", required=True)
    parser.add_argument("--json-out", default="")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条，调试用")
    parser.add_argument("--no-generate", action="store_true", help="跳过生成层（只测检索）")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise SystemExit(f"数据集不存在: {dataset_path}")
    items = load_dataset(dataset_path)

    specs = eval_configs.load_configs(args.configs)
    module = __import__(
        "knowledgebase_server.loader.KnowledgeFastAPI", fromlist=["retrieve"]
    )
    from configs import runtime_config

    counts = {}
    for item in items:
        counts[item["type"]] = counts.get(item["type"], 0) + 1

    results = []
    per_item = {}
    for spec in specs:
        name = spec["name"]
        retrieval_config = eval_configs.to_retrieval_config(spec)
        print(f"== 配置 {name} ==", flush=True)
        # 预热：首个请求要加载 embedding / 重排模型，不计入延迟统计
        started = time.perf_counter()
        try:
            module.retrieve(args.kb, "预热", config=retrieval_config, strict_reranker=False)
        except Exception as exc:
            print(f"   预热失败：{type(exc).__name__}: {exc}", flush=True)
        print(f"   预热 {time.perf_counter() - started:.1f}s", flush=True)

        rows = run_retrieval(module, args.kb, items, retrieval_config, args.k, args.limit)
        per_item[name] = {row["id"]: row for row in rows}

        if not args.no_generate:
            print(f"   生成层…", flush=True)
            # 生成需要原始 Document，重跑一次检索（保持文档对象，避免 pickle 到磁盘）
            from configs import runtime_config as _rc  # noqa: F401

            docs_by_id = {}
            for item in items[: args.limit] if args.limit else items:
                result = module.retrieve(
                    args.kb,
                    item["question"],
                    config=retrieval_config,
                    strict_reranker=retrieval_config.use_reranker,
                )
                docs_by_id[item["id"]] = list(result.docs)
            for row in rows:
                row["_documents"] = docs_by_id.get(row["id"], [])
            run_generation(
                module,
                items[: args.limit] if args.limit else items,
                per_item[name],
                retrieval_config,
                args.k,
            )
            for row in rows:
                row.pop("_documents", None)

        summary = summarize(rows, generate=not args.no_generate)
        results.append((name, summary))
        print(f"   -> {json.dumps(summary, ensure_ascii=False)}", flush=True)
        del_module_retrieval_cache()
        gc.collect()

    selected_items = items[: args.limit] if args.limit else items
    item_meta = {
        "order": [item["id"] for item in selected_items],
        "items": {
            item["id"]: {"type": item["type"], "question": item["question"]}
            for item in selected_items
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.json_out:
        payload = {
            "summary": {name: summary for name, summary in results},
            "per_item": {
                name: {item_id: row for item_id, row in rows.items()}
                for name, rows in per_item.items()
            },
        }
        Path(args.json_out).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    conclusions = build_conclusion(results, args.k)
    meta = {
        "date": dt.date.today().isoformat(),
        "git": git_revision(),
        "dataset": str(dataset_path),
        "dataset_md5": dataset_md5(dataset_path),
        "items": len(items),
        "counts": counts,
        "kb": args.kb,
        "chunks": count_chunks(args.kb),
        "corpus": ["agent面试问答.md", "唐诗三百首.md"],
        "chat_model": runtime_config.get("chat_model"),
        "k": args.k,
        "python": f"{sys.version.split()[0]} / {sys.platform}",
        "config_specs": specs,
        "per_item": per_item,
        "item_meta": item_meta,
        "conclusion": conclusions,
    }
    report = render_report(meta, results)
    out_path.write_text(report, encoding="utf-8")
    print(f"\n报告已写入 {out_path}", flush=True)
    return 0


def count_chunks(kb_name: str):
    """报告里写清语料规模；读不出来就返回 None，不让统计影响实验本身。"""
    try:
        from eval.tools.dump_chunks import load_documents, resolve_kb_path

        kb_path = resolve_kb_path(Path("./knowledgebases"), kb_name)
        if not kb_path:
            return None
        return len(load_documents(kb_path))
    except Exception:
        return None


def build_conclusion(results, k: int) -> str:
    if len(results) < 2:
        return "只有一份配置，无法对比。"
    baseline_name, baseline = results[0]
    lines = []
    for name, summary in results[1:]:
        delta_recall = summary["evidence_recall@5"] - baseline["evidence_recall@5"]
        delta_mrr = summary["evidence_mrr@5"] - baseline["evidence_mrr@5"]
        delta_p50 = summary["latency_ms_p50"] - baseline["latency_ms_p50"]
        lines.append(
            f"- 相对 `{baseline_name}`：`{name}` 的证据 Recall@{k} "
            f"{'+' if delta_recall >= 0 else ''}{delta_recall * 100:.1f} 个百分点，"
            f"证据 MRR@{k} {'+' if delta_mrr >= 0 else ''}{delta_mrr * 100:.1f}，"
            f"检索 p50 {'+' if delta_p50 >= 0 else ''}{delta_p50:.0f} ms。"
        )
    return "\n".join(lines)


def del_module_retrieval_cache():
    """配置之间换召回器时释放缓存，避免误用上一份配置的 retriever。"""
    try:
        from knowledgebase_server.loader import KnowledgeFastAPI as module

        cache = getattr(module, "_search_cache", None)
        if isinstance(cache, dict):
            cache.clear()
    except Exception:
        pass


if __name__ == "__main__":
    raise SystemExit(main())