"""chunk id 单测：验证「跨重建稳定」这件事本身。

    python -m unittest discover -s eval -t . -v
"""

import unittest

from eval.chunks import chunk_id, chunk_ids_of, dedupe_keep_order, normalize_text, source_name


class FakeDocument:
    def __init__(self, content, source=""):
        self.page_content = content
        self.metadata = {"source": source}


class ChunkIdTest(unittest.TestCase):
    def test_same_content_same_id(self):
        self.assertEqual(chunk_id("a.md", "正文"), chunk_id("a.md", "正文"))

    def test_whitespace_does_not_change_id(self):
        self.assertEqual(chunk_id("a.md", "第一行\n第二行"), chunk_id("a.md", "第一行   第二行"))
        self.assertEqual(chunk_id("a.md", "第一行\r\n第二行"), chunk_id("a.md", "第一行\n第二行"))

    def test_content_changes_id(self):
        self.assertNotEqual(chunk_id("a.md", "正文"), chunk_id("a.md", "正文改"))

    def test_source_changes_id(self):
        self.assertNotEqual(chunk_id("a.md", "正文"), chunk_id("b.md", "正文"))

    def test_absolute_or_relative_path_same_id(self):
        self.assertEqual(
            chunk_id("/home/Hemy/data/kb/a.md", "正文"),
            chunk_id("./data/kb/a.md", "正文"),
        )
        self.assertEqual(chunk_id("data\\kb\\a.md", "正文"), chunk_id("data/kb/a.md", "正文"))

    def test_id_is_fixed_length_hex(self):
        value = chunk_id("a.md", "正文")
        self.assertEqual(len(value), 16)
        int(value, 16)

    def test_document_helper(self):
        document = FakeDocument("正文", "/tmp/a.md")
        self.assertEqual(chunk_ids_of([document]), [chunk_id("a.md", "正文")])

    def test_document_without_source(self):
        self.assertEqual(chunk_ids_of([FakeDocument("正文")]), [chunk_id("", "正文")])


class HelperTest(unittest.TestCase):
    def test_normalize_text(self):
        self.assertEqual(normalize_text("  a\n\tb  "), "ab")
        self.assertEqual(normalize_text(None), "")

    def test_source_name(self):
        self.assertEqual(source_name("/a/b/c.md"), "c.md")
        self.assertEqual(source_name("c.md"), "c.md")
        self.assertEqual(source_name(""), "")

    def test_dedupe_keep_order(self):
        self.assertEqual(dedupe_keep_order(["b", "a", "b", "c"]), ["b", "a", "c"])
        self.assertEqual(dedupe_keep_order([]), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)