"""指标单测：不进容器、不联网、不加载模型，CI 里直接跑。

    python -m unittest discover -s eval -t . -v
"""

import math
import unittest

from eval.metrics import (
    aggregate,
    evaluate_evidence_query,
    evaluate_query,
    evidence_ndcg_at_k,
    evidence_mrr_at_k,
    evidence_recall_at_k,
    first_hit_rank,
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    percentile,
    recall_at_k,
)


class RecallTest(unittest.TestCase):
    def test_single_gold_at_first_place(self):
        self.assertEqual(recall_at_k(["a", "b", "c"], {"a"}, 3), 1.0)

    def test_single_gold_missed(self):
        self.assertEqual(recall_at_k(["a", "b", "c"], {"z"}, 3), 0.0)

    def test_gold_outside_cutoff(self):
        self.assertEqual(recall_at_k(["a", "b", "c", "d", "e", "g"], {"g"}, 5), 0.0)

    def test_multiple_gold_partial(self):
        self.assertAlmostEqual(recall_at_k(["a", "b", "c"], {"a", "z"}, 3), 0.5)

    def test_multiple_gold_all(self):
        self.assertEqual(recall_at_k(["a", "b", "c"], {"a", "c"}, 3), 1.0)

    def test_empty_ranked(self):
        self.assertEqual(recall_at_k([], {"a"}, 5), 0.0)

    def test_duplicates_count_once(self):
        self.assertEqual(recall_at_k(["a", "a", "a"], {"a"}, 3), 1.0)

    def test_empty_gold_is_error(self):
        with self.assertRaises(ValueError):
            recall_at_k(["a"], set(), 5)

    def test_invalid_k(self):
        for bad_k in (0, -1):
            with self.assertRaises(ValueError):
                recall_at_k(["a"], {"a"}, bad_k)


class MrrTest(unittest.TestCase):
    def test_first_place(self):
        self.assertEqual(mrr_at_k(["a"], {"a"}, 5), 1.0)

    def test_third_place(self):
        self.assertAlmostEqual(mrr_at_k(["x", "y", "a"], {"a"}, 5), 1 / 3)

    def test_missed(self):
        self.assertEqual(mrr_at_k(["x", "y"], {"a"}, 5), 0.0)

    def test_duplicates_do_not_waste_rank(self):
        self.assertAlmostEqual(mrr_at_k(["a", "a", "b"], {"b"}, 5), 0.5)

    def test_only_first_hit_counts(self):
        self.assertAlmostEqual(mrr_at_k(["a", "b"], {"a", "b"}, 5), 1.0)


class NdcgTest(unittest.TestCase):
    def test_perfect(self):
        self.assertAlmostEqual(ndcg_at_k(["a", "b"], {"a", "b"}, 5), 1.0)

    def test_single_hit_at_third(self):
        expected = (1.0 / math.log2(4)) / 1.0
        self.assertAlmostEqual(ndcg_at_k(["x", "y", "a"], {"a"}, 5), expected)

    def test_two_gold_at_rank_two_and_three(self):
        dcg = 1.0 / math.log2(3) + 1.0 / math.log2(4)
        idcg = 1.0 + 1.0 / math.log2(3)
        self.assertAlmostEqual(ndcg_at_k(["x", "a", "b"], {"a", "b"}, 5), dcg / idcg)

    def test_gold_more_than_k_caps_idcg(self):
        self.assertAlmostEqual(ndcg_at_k(["a", "b"], {"a", "b", "c"}, 2), 1.0)

    def test_missed(self):
        self.assertEqual(ndcg_at_k(["x"], {"a"}, 5), 0.0)


class HitTest(unittest.TestCase):
    def test_hit(self):
        self.assertEqual(hit_at_k(["x", "a"], {"a"}, 5), 1.0)

    def test_miss(self):
        self.assertEqual(hit_at_k(["x"], {"a"}, 5), 0.0)


class FirstHitRankTest(unittest.TestCase):
    def test_rank(self):
        self.assertEqual(first_hit_rank(["x", "a", "b"], {"b"}, 5), 3)

    def test_miss(self):
        self.assertEqual(first_hit_rank(["x"], {"a"}, 5), 0)


class EvaluateQueryTest(unittest.TestCase):
    def test_keys_match_k(self):
        result = evaluate_query(["a"], {"a"}, k=3)
        self.assertEqual(set(result), {"recall@3", "mrr@3", "ndcg@3"})


class PercentileTest(unittest.TestCase):
    def test_p50(self):
        self.assertAlmostEqual(percentile([4, 1, 3, 2], 50), 2.5)

    def test_p95(self):
        self.assertAlmostEqual(percentile([1, 2, 3, 4], 95), 3.85)

    def test_single_value(self):
        self.assertEqual(percentile([7], 95), 7.0)

    def test_empty(self):
        self.assertEqual(percentile([], 50), 0.0)

    def test_invalid_p(self):
        with self.assertRaises(ValueError):
            percentile([1, 2], 120)


class AggregateTest(unittest.TestCase):
    def test_group_means(self):
        rows = [
            {"type": "normal", "recall@5": 1.0, "mrr@5": 1.0},
            {"type": "normal", "recall@5": 0.0, "mrr@5": 0.0},
            {"type": "boundary", "recall@5": 1.0, "mrr@5": 1.0},
        ]
        summary = aggregate(rows, ["recall@5", "mrr@5"])
        self.assertEqual(summary["all"]["count"], 3)
        self.assertAlmostEqual(summary["all"]["recall@5"], 2 / 3)
        self.assertEqual(summary["normal"]["count"], 2)
        self.assertAlmostEqual(summary["normal"]["recall@5"], 0.5)
        self.assertEqual(summary["boundary"]["recall@5"], 1.0)

    def test_empty_rows(self):
        summary = aggregate([], ["recall@5"])
        self.assertEqual(summary["all"]["count"], 0)
        self.assertEqual(summary["all"]["recall@5"], 0.0)


class EvidenceCoverageTest(unittest.TestCase):
    """证据覆盖口径：答案片段出现在某段正文里就算命中，与 chunk 边界无关。"""

    def test_single_evidence_in_second_doc(self):
        texts = ["无关内容", "这里提到点积的方差会随维度 d_k 线性增长", "第三段"]
        self.assertEqual(evidence_recall_at_k(texts, ["点积的方差会随维度 d_k 线性增长"], 3), 1.0)
        self.assertAlmostEqual(evidence_mrr_at_k(texts, ["点积的方差会随维度 d_k 线性增长"], 3), 0.5)

    def test_evidence_with_whitespace_and_punctuation(self):
        texts = ["原文：Variance grows  linearly with d_k."]
        self.assertEqual(evidence_recall_at_k(texts, ["Variance grows linearly with d_k"], 1), 1.0)

    def test_evidence_outside_cutoff(self):
        texts = ["a", "b", "目标片段在这里"]
        self.assertEqual(evidence_recall_at_k(texts, ["目标片段在这里"], 2), 0.0)

    def test_multi_evidence_partial(self):
        texts = ["包含片段一的内容", "无关", "包含片段二的内容"]
        self.assertAlmostEqual(evidence_recall_at_k(texts, ["片段一", "片段二"], 2), 0.5)
        self.assertEqual(evidence_recall_at_k(texts, ["片段一", "片段二"], 3), 1.0)

    def test_duplicate_evidence_counted_once(self):
        texts = ["同一段包含片段一"]
        self.assertEqual(evidence_recall_at_k(texts, ["片段一", "片段一"], 1), 1.0)

    def test_ndcg_perfect_when_first_doc_covers_everything(self):
        texts = ["片段一 和 片段二 都在这里"]
        self.assertAlmostEqual(evidence_ndcg_at_k(texts, ["片段一", "片段二"], 1), 1.0)

    def test_ndcg_ordering_matters(self):
        early = evidence_ndcg_at_k(["片段一", "片段二"], ["片段一", "片段二"], 2)
        late = evidence_ndcg_at_k(["无关", "无关", "片段一", "片段二"], ["片段一", "片段二"], 5)
        self.assertGreater(early, late)

    def test_empty_evidence_is_error(self):
        with self.assertRaises(ValueError):
            evidence_recall_at_k(["a"], [], 5)

    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            evidence_recall_at_k(["a"], ["a"], 0)

    def test_evaluate_evidence_query_keys(self):
        result = evaluate_evidence_query(["含片段一"], ["片段一"], 5)
        self.assertEqual(
            set(result),
            {"evidence_recall@5", "evidence_mrr@5", "evidence_ndcg@5", "evidence_first_rank"},
        )
        self.assertEqual(result["evidence_first_rank"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)