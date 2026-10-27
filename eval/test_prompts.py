"""prompt 层单测：拒答判定和上下文拼装都是规则，必须能离线验证。"""

import unittest

from eval.prompts import (
    MAX_CONTEXT_CHARS,
    REFUSAL_MARKER,
    build_user_prompt,
    format_contexts,
    is_refusal,
)


class _Doc:
    def __init__(self, text, source="data/评测基准/语料.md"):
        self.page_content = text
        self.metadata = {"source": source}


class RefusalTest(unittest.TestCase):
    def test_exact_marker(self):
        self.assertTrue(is_refusal(REFUSAL_MARKER))

    def test_marker_with_whitespace(self):
        self.assertTrue(is_refusal("  知识库中未找到相关内容。  \n"))

    def test_marker_with_trailing_caveat(self):
        # 开头就声明资料不足，后面再补内容也按拒答算
        self.assertTrue(is_refusal("知识库中未找到相关内容，但根据常识可能是 42。"))

    def test_answer_with_caveat_at_the_end_is_not_refusal(self):
        # 实测踩过的坑：正常作答后补一句免责声明，不能判成拒答
        answer = (
            "注意力打分时除以根号 d_k 是为了缩放点积结果，防止方差过大 [4]。"
            "知识库中未找到不除时的具体后果描述。"
        )
        self.assertFalse(is_refusal(answer))

    def test_plain_answer(self):
        self.assertFalse(is_refusal("除以根号 d_k 是为了把方差拉回 1。"))

    def test_empty_answer_is_not_refusal(self):
        self.assertFalse(is_refusal(""))
        self.assertFalse(is_refusal("   "))


class ContextTest(unittest.TestCase):
    def test_numbering_and_source_basename(self):
        text = format_contexts([_Doc("第一段", "a/b/唐诗三百首.md"), _Doc("第二段")])
        self.assertIn("[1] 来源：唐诗三百首.md", text)
        self.assertIn("[2] 来源：语料.md", text)
        self.assertLess(text.index("第一段"), text.index("第二段"))

    def test_truncates_beyond_budget(self):
        docs = [_Doc("字" * (MAX_CONTEXT_CHARS // 2 + 10)) for _ in range(5)]
        text = format_contexts(docs)
        self.assertLessEqual(len(text), MAX_CONTEXT_CHARS + 40)
        self.assertIn("[1]", text)
        self.assertNotIn("[5]", text)

    def test_empty_documents(self):
        self.assertEqual(format_contexts([]), "")

    def test_user_prompt_shape(self):
        prompt = build_user_prompt("问题是什么？", [_Doc("资料正文")])
        self.assertIn("【资料】", prompt)
        self.assertIn("资料正文", prompt)
        self.assertIn("【问题】\n问题是什么？", prompt)

    def test_user_prompt_without_documents(self):
        self.assertIn("没有检索到任何资料", build_user_prompt("问题", []))


if __name__ == "__main__":
    unittest.main(verbosity=2)