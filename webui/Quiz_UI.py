# -*- coding: utf-8 -*-
"""出题（练习）相关的前端请求函数：出题 / 题目切分 / 错题保存为笔记。
与 Chat_UI / Knowledge_UI / Summary_UI 保持一致：只负责 HTTP 请求逻辑，不持有 UI 组件。
"""
import requests
from configs.setting import BACKEND_URL, AUTH_HEADERS


def generate_quiz(kb_name, num_questions, question_type):
    """根据知识库内容出题（出题模式），只生成题目，保存由用户对完答案后自行勾选"""
    if not kb_name:
        return "❌ 请先选择出题知识库"
    try:
        r = requests.post(
            f"{BACKEND_URL}/knowledge/quiz",
            json={
                "kb_name": kb_name,
                "num_questions": int(num_questions or 5),
                "question_type": question_type or "混合",
            },
            timeout=300,
            headers=AUTH_HEADERS,
        )
        if r.status_code == 200:
            return r.json().get("quiz", "出题结果为空")
        try:
            detail = r.json().get("detail", r.text[:200])
        except Exception:
            detail = r.text[:200]
        return f"❌ 出题失败：{detail}"
    except Exception as e:
        return f"❌ 出题请求异常：{e}"


def _split_quiz_text(text):
    """将出题结果 Markdown 按题号切分为 {题号: 题目文本}（对答案后按题号保存用）"""
    import re as _re
    result = {}
    cur = None
    for line in (text or "").split("\n"):
        m = _re.match(r"^\s*(\d{1,2})\s*[.、．]", line)
        if m:
            cur = int(m.group(1))
            result[cur] = line
        elif cur is not None and line.strip():
            result[cur] += "\n" + line
    return result


def save_quiz_notes(quiz_text, kb_name, save_sel):
    """对完答案后，把勾选（做错/想复习）的题目保存为笔记：自动命名「错题」，同名自动追加，形成错题本"""
    if not quiz_text or not quiz_text.strip():
        return "⚠️ 请先出题"
    if not kb_name:
        return "⚠️ 请先选择出题知识库"
    try:
        indices = [int(x) for x in (save_sel or [])]
    except Exception:
        indices = []
    if not indices:
        return "⚠️ 请勾选要保存的题目（对完答案后勾选错题）"
    by_num = _split_quiz_text(quiz_text)
    picked = [by_num[i] for i in sorted(set(indices)) if i in by_num]
    if not picked:
        return "⚠️ 未匹配到勾选的题号，请确认题目编号"
    content = f"# 「{kb_name}」错题整理\n\n" + "\n\n".join(picked)
    try:
        r = requests.post(
            f"{BACKEND_URL}/knowledge/save_note",
            json={
                "file_name": "错题",
                "content": content,
                "kb_name": kb_name,
            },
            timeout=120,
            headers=AUTH_HEADERS,
        )
        if r.status_code == 200:
            d = r.json()
            msg = f"✅ 已保存 {len(picked)} 题到「{kb_name}」，笔记：{d.get('file_name')}"
            if d.get("appended"):
                msg += "（同名笔记已自动追加）"
            return msg
        try:
            detail = r.json().get("detail", r.text[:200])
        except Exception:
            detail = r.text[:200]
        return f"❌ 保存失败：{detail}"
    except Exception as e:
        return f"❌ 保存请求异常：{e}"
