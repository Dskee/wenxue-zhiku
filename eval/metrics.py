"""检索指标：纯函数，不调模型、不读文件、不写文件。

约定：
- ranked 是检索返回的 chunk id 列表，按相关度从高到低；函数内部会先按首次出现去重。
- gold 是正确答案所在 chunk 的 id 集合，可以为多个（同一个问题被切进两段）。
- k 为截断位置（1-based），所有指标只看 ranked[:k]。
- gold 为空视为数据集错误，直接抛 ValueError，不允许静默返回 0。

指标口径（报告里会原样写明，避免歧义）：
- Recall@k = 命中的 gold 数 / gold 总数。只有一个 gold 时等价于 Hit@k。
- MRR@k   = 1 / 第一个命中 gold 的名次；k 内无命中记 0。
- nDCG@k  = DCG@k / IDCG@k，增益二值（命中记 1），DCG = Σ gain_i / log2(i + 1)。

块级指标（上面三个）用 chunk id 判分，精确但受切块边界影响；证据覆盖指标
（evidence_recall/mrr/ndcg）以「答案片段是否出现在某段检索结果正文里」判分，
不受切块大小影响，报告的主指标用它，块级指标保留作对照。
"""

import math
from typing import Dict, Iterable, List, Sequence

from eval.chunks import dedupe_keep_order, normalize_text

DEFAULT_K = 5


def _validate(ranked: Sequence[str], gold: Iterable[str], k: int) -> List[str]:
    if not isinstance(k, int) or k <= 0:
        raise ValueError(f"k 必须是正整数，收到 {k!r}")
    gold_set = set(gold or [])
    if not gold_set:
        raise ValueError("gold 不能为空；no_answer 类样本不应计算检索指标")
    return dedupe_keep_order(list(ranked or []))


def recall_at_k(ranked: Sequence[str], gold: Iterable[str], k: int = DEFAULT_K) -> float:
    """命中的 gold 占全部 gold 的比例，取值 [0, 1]。"""
    gold_set = set(gold)
    ordered = _validate(ranked, gold_set, k)
    if not gold_set:
        raise ValueError("gold 不能为空")
    hits = sum(1 for chunk in ordered[:k] if chunk in gold_set)
    return hits / len(gold_set)


def mrr_at_k(ranked: Sequence[str], gold: Iterable[str], k: int = DEFAULT_K) -> float:
    """第一个命中 gold 的名次的倒数，取值 [0, 1]。"""
    gold_set = set(gold)
    ordered = _validate(ranked, gold_set, k)
    for position, chunk in enumerate(ordered[:k], start=1):
        if chunk in gold_set:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked: Sequence[str], gold: Iterable[str], k: int = DEFAULT_K) -> float:
    """二值增益的归一化折损累计增益，取值 [0, 1]。"""
    gold_set = set(gold)
    ordered = _validate(ranked, gold_set, k)
    dcg = 0.0
    for position, chunk in enumerate(ordered[:k], start=1):
        if chunk in gold_set:
            dcg += 1.0 / math.log2(position + 1)
    ideal_hits = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(position + 1) for position in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def hit_at_k(ranked: Sequence[str], gold: Iterable[str], k: int = DEFAULT_K) -> float:
    """top-k 里是否至少命中一个 gold，返回 1.0 / 0.0。"""
    return 1.0 if recall_at_k(ranked, gold, k) > 0 else 0.0


def evaluate_query(
    ranked: Sequence[str],
    gold: Iterable[str],
    k: int = DEFAULT_K,
) -> Dict[str, float]:
    """单条样本的三个检索指标。"""
    gold_set = set(gold)
    return {
        f"recall@{k}": recall_at_k(ranked, gold_set, k),
        f"mrr@{k}": mrr_at_k(ranked, gold_set, k),
        f"ndcg@{k}": ndcg_at_k(ranked, gold_set, k),
    }


def first_hit_rank(ranked: Sequence[str], gold: Iterable[str], k: int = DEFAULT_K) -> int:
    """第一个命中的名次，未命中返回 0。写失败样例时用。"""
    gold_set = set(gold)
    for position, chunk in enumerate(dedupe_keep_order(list(ranked or []))[:k], start=1):
        if chunk in gold_set:
            return position
    return 0


def evidence_ranks(
    retrieved_texts: Sequence[str],
    evidences: Sequence[str],
    k: int = DEFAULT_K,
) -> List[int]:
    """每条证据片段第一次被 top-k 中某段正文覆盖的名次；没被覆盖记 0。

    为什么不用 chunk id 判分：切块大小是 128 字，一段完整答案经常横跨相邻两个 chunk，
    「命中了 id」和「上下文里有答案」是两件事。证据片段是逐字从语料里摘的，
    只要它出现在某段检索结果里，就说明模型拿到的上下文足以回答——这个口径跟切块边界无关。
    """
    if not isinstance(k, int) or k <= 0:
        raise ValueError(f"k 必须是正整数，收到 {k!r}")
    snippets = [snippet for snippet in (normalize_text(e) for e in evidences) if snippet]
    if not snippets:
        raise ValueError("evidence 不能为空；no_answer 类样本不应计算检索指标")
    normalized_texts = [normalize_text(text) for text in retrieved_texts[:k]]
    ranks: List[int] = []
    for snippet in snippets:
        rank = 0
        for position, text in enumerate(normalized_texts, start=1):
            if snippet in text:
                rank = position
                break
        ranks.append(rank)
    return ranks


def evidence_recall_at_k(
    retrieved_texts: Sequence[str], evidences: Sequence[str], k: int = DEFAULT_K
) -> float:
    """被覆盖的证据片段占全部片段的比例。"""
    ranks = evidence_ranks(retrieved_texts, evidences, k)
    return sum(1 for rank in ranks if rank) / len(ranks)


def evidence_mrr_at_k(
    retrieved_texts: Sequence[str], evidences: Sequence[str], k: int = DEFAULT_K
) -> float:
    """各证据片段首次被覆盖名次倒数的均值。"""
    ranks = evidence_ranks(retrieved_texts, evidences, k)
    return sum(1.0 / rank for rank in ranks if rank) / len(ranks)


def evidence_ndcg_at_k(
    retrieved_texts: Sequence[str], evidences: Sequence[str], k: int = DEFAULT_K
) -> float:
    """以证据片段为增益单元、按首次覆盖名次折损的 nDCG。

    同一段正文可以一次覆盖多条证据（它们名次相同，增益相加），理想情况是全部证据在第一位，
    所以 IDCG 取「片段数 × 1.0」，保证结果落在 [0, 1]。
    """
    ranks = evidence_ranks(retrieved_texts, evidences, k)
    counts: Dict[int, int] = {}
    for rank in ranks:
        if rank:
            counts[rank] = counts.get(rank, 0) + 1
    dcg = sum(count / math.log2(rank + 1) for rank, count in counts.items())
    return dcg / len(ranks) if ranks else 0.0


def evaluate_evidence_query(
    retrieved_texts: Sequence[str],
    evidences: Sequence[str],
    k: int = DEFAULT_K,
) -> Dict[str, float]:
    """单条样本的证据覆盖指标（主指标）；键名带 evidence_ 前缀，和块级指标区分。"""
    ranks = evidence_ranks(retrieved_texts, evidences, k)
    return {
        f"evidence_recall@{k}": sum(1 for rank in ranks if rank) / len(ranks),
        f"evidence_mrr@{k}": sum(1.0 / rank for rank in ranks if rank) / len(ranks),
        f"evidence_ndcg@{k}": evidence_ndcg_at_k(retrieved_texts, evidences, k),
        "evidence_first_rank": min((rank for rank in ranks if rank), default=0),
    }


def mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


def percentile(values: Iterable[float], p: float) -> float:
    """线性插值分位数（p 取 0~100），用于时延的 p50 / p95。"""
    items = sorted(float(value) for value in values)
    if not items:
        return 0.0
    if not 0 <= p <= 100:
        raise ValueError(f"p 必须在 0~100，收到 {p!r}")
    if len(items) == 1:
        return items[0]
    position = (len(items) - 1) * (p / 100.0)
    lower = int(math.floor(position))
    upper = min(lower + 1, len(items) - 1)
    weight = position - lower
    return items[lower] * (1 - weight) + items[upper] * weight


def aggregate(rows: Sequence[Dict], metric_names: Sequence[str], group_key: str = "type") -> Dict[str, Dict[str, float]]:
    """按类型分组求均值。

    rows 里每条至少含 group_key 与 metric_names 中的字段；返回 {"all": {...}, "<类型>": {...}}，
    每组额外带 "count"。no_answer（没有 gold）的样本不应进 rows。
    """
    groups: Dict[str, List[Dict]] = {"all": list(rows)}
    for row in rows:
        name = str(row.get(group_key, "unknown"))
        groups.setdefault(name, []).append(row)

    summary: Dict[str, Dict[str, float]] = {}
    for name, items in groups.items():
        bucket = {"count": len(items)}
        for metric in metric_names:
            bucket[metric] = mean(item.get(metric, 0.0) for item in items)
        summary[name] = bucket
    return summary