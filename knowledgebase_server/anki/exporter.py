# -*- coding: utf-8 -*-
"""Anki 卡片包（.apkg）生成模块，底层用 genanki。

为什么要单独一层：
  1. 打包逻辑（卡片字段 → Anki 模型/模板/guid）和"从笔记生成卡片"（要调大模型）
     是两件不同的事，分开后前者纯函数、可单测，不碰网络；
  2. 三种卡片类型的模板/字段差异集中在模型定义里，接口层只挑类型，不用关心 HTML。

支持三种卡片（导出时可选）：
  basic  基础问答卡：正面问题，背面答案
  cloze  填空卡：正文里用 {{c1::关键词}} 挖空，适合定义、公式、术语
  choice 选择题卡：题干 + 选项，翻面给答案与解析

ID 策略（重要）：
  · 模型 ID 用固定常量 —— 多次导出时模板定义不变，用户重复导入不会生成重复模板；
  · 牌组 ID 由牌组名哈希得到 —— 不同知识库导出成不同牌组，同名则落到同一个牌组；
  · 卡片 guid 由内容哈希得到 —— 同一张卡重复导入是"更新"而不是"新增副本"。
"""
import hashlib
import html
import os
import re

import genanki

# 固定模型 ID（同一套模板，重复导入不会产生重复模型）
BASIC_MODEL_ID = 1704537001
CLOZE_MODEL_ID = 1704537002
CHOICE_MODEL_ID = 1704537003

CARD_TYPES = ("basic", "cloze", "choice")
CARD_TYPE_LABELS = {
    "basic": "基础问答卡",
    "cloze": "填空卡",
    "choice": "选择题卡",
}


def _stable_id(seed: str, span: int = 900_000_000, base: int = 1_600_000_000) -> int:
    """把任意字符串映射到一个稳定的正整数 ID（Anki 的 ID 是 int64）。"""
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()
    return base + int(digest[:12], 16) % span


def _md_to_html(text: str) -> str:
    """把笔记里的轻量 Markdown 转成 Anki 能渲染的 HTML。

    先转义再套标签：笔记内容来自用户上传的文件，直接当 HTML 渲染会有注入风险。
    """
    s = html.escape(str(text or ""), quote=False)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = s.replace("\r\n", "\n")
    return s.replace("\n", "<br>")


# ============================ 三种模型 ============================

def _basic_model() -> genanki.Model:
    return genanki.Model(
        BASIC_MODEL_ID,
        "问学知库 - 问答卡",
        fields=[{"name": "Front"}, {"name": "Back"}],
        templates=[{
            "name": "问答",
            "qfmt": "{{Front}}",
            "afmt": "{{FrontSide}}<hr id=answer>{{Back}}",
        }],
    )


def _cloze_model() -> genanki.Model:
    return genanki.Model(
        CLOZE_MODEL_ID,
        "问学知库 - 填空卡",
        model_type=genanki.Model.CLOZE,
        fields=[{"name": "Text"}, {"name": "Back Extra"}],
        templates=[{
            "name": "填空",
            "qfmt": "{{cloze:Text}}",
            "afmt": "{{cloze:Text}}<br>{{Back Extra}}",
        }],
    )


def _choice_model() -> genanki.Model:
    return genanki.Model(
        CHOICE_MODEL_ID,
        "问学知库 - 选择题卡",
        fields=[
            {"name": "Question"},
            {"name": "Options"},
            {"name": "Answer"},
            {"name": "Explanation"},
        ],
        templates=[{
            "name": "选择题",
            "qfmt": "{{Question}}<br><br>{{Options}}",
            "afmt": "{{FrontSide}}<hr id=answer>答案：<b>{{Answer}}</b><br>{{Explanation}}",
        }],
    )


# ============================ 卡片校验/组装 ============================

def normalize_card(card: dict, card_type: str) -> dict | None:
    """校验并规范化单张卡片；字段不全返回 None（调用方计入 skipped）。"""
    card = card or {}

    if card_type == "basic":
        front = str(card.get("front") or card.get("question") or "").strip()
        back = str(card.get("back") or card.get("answer") or "").strip()
        if not front or not back:
            return None
        return {"front": front, "back": back}

    if card_type == "cloze":
        text = str(card.get("text") or "").strip()
        extra = str(card.get("extra") or card.get("back") or "").strip()
        # 至少要有 {{c1::...}} 挖空，否则 Anki 里这张卡是空白
        if not text or not re.search(r"\{\{c\d+::.+?\}\}", text):
            return None
        return {"text": text, "extra": extra}

    if card_type == "choice":
        question = str(card.get("question") or card.get("front") or "").strip()
        options = str(card.get("options") or "").strip()
        answer = str(card.get("answer") or "").strip()
        explanation = str(card.get("explanation") or card.get("back") or "").strip()
        if not question or not answer:
            return None
        return {"question": question, "options": options,
                "answer": answer, "explanation": explanation}

    return None


def _note_for(card: dict, card_type: str, tags: list) -> genanki.Note:
    """按卡片类型组装 genanki.Note，并用内容哈希做 guid。"""
    if card_type == "basic":
        model = _basic_model()
        fields = [_md_to_html(card["front"]), _md_to_html(card["back"])]
        seed = "basic|" + card["front"]
    elif card_type == "cloze":
        model = _cloze_model()
        # 挖空标记要保留原样，只转义其余部分
        text = card["text"]
        parts = re.split(r"(\{\{c\d+::.+?\}\})", text)
        text_html = "".join(
            p if re.fullmatch(r"\{\{c\d+::.+?\}\}", p) else _md_to_html(p)
            for p in parts
        )
        fields = [text_html, _md_to_html(card.get("extra", ""))]
        seed = "cloze|" + card["text"]
    else:
        model = _choice_model()
        fields = [
            _md_to_html(card["question"]),
            _md_to_html(card["options"]),
            _md_to_html(card["answer"]),
            _md_to_html(card.get("explanation", "")),
        ]
        seed = "choice|" + card["question"]

    return genanki.Note(model=model, fields=fields,
                        tags=[t for t in (tags or []) if t],
                        guid=_stable_id(seed))


def build_apkg(cards, card_type: str, deck_name: str, out_path: str,
               tags=None) -> dict:
    """把卡片列表打包成 .apkg，返回 {count, skipped, path, deck}。

    cards 里不合法的卡会被跳过并计数，不会让整次导出失败。
    """
    if card_type not in CARD_TYPES:
        raise ValueError(f"不支持的卡片类型: {card_type}（可选 {', '.join(CARD_TYPES)}）")

    deck_name = (deck_name or "问学知库").strip() or "问学知库"
    deck = genanki.Deck(_stable_id("deck|" + deck_name), deck_name)

    count = 0
    skipped = 0
    for raw in (cards or []):
        norm = normalize_card(raw, card_type)
        if not norm:
            skipped += 1
            continue
        try:
            deck.add_note(_note_for(norm, card_type, list(tags or [])))
            count += 1
        except Exception:
            skipped += 1

    if count == 0:
        raise ValueError("没有可用的卡片（请检查内容是否符合所选卡片类型）")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    genanki.Package(deck).write_to_file(out_path)

    return {"count": count, "skipped": skipped, "path": out_path, "deck": deck_name}


def preview_card(card: dict, card_type: str) -> dict | None:
    """给前端做预览用：返回渲染后的正反面 HTML，不发包。"""
    norm = normalize_card(card, card_type)
    if not norm:
        return None
    if card_type == "basic":
        return {"front": _md_to_html(norm["front"]), "back": _md_to_html(norm["back"])}
    if card_type == "cloze":
        text = norm["text"]
        parts = re.split(r"(\{\{c\d+::.+?\}\})", text)
        front = "".join(
            "[...]" if re.fullmatch(r"\{\{c\d+::.+?\}\}", p) else _md_to_html(p)
            for p in parts
        )
        back = "".join(
            "<b>" + _md_to_html(re.sub(r"^\{\{c\d+::|(\}\})$", "", p)) + "</b>"
            if re.fullmatch(r"\{\{c\d+::.+?\}\}", p) else _md_to_html(p)
            for p in parts
        )
        return {"front": front, "back": back, "extra": _md_to_html(norm["extra"])}
    return {
        "front": _md_to_html(norm["question"]) + "<br><br>" + _md_to_html(norm["options"]),
        "back": "答案：" + _md_to_html(norm["answer"]) + "<br>" + _md_to_html(norm["explanation"]),
    }
