import os
import requests
import gradio as gr
from configs.setting import BACKEND_URL, AUTH_HEADERS

# 导出时「保存到知识库」下拉选项
AUTO_OPTION = '（自动识别：匹配已有知识库，匹配不到则新建）'
NEW_OPTION = '⚡ 新建知识库（手动指定名称）'


def _message_content_to_text(content):
    """把 Gradio 6 消息 content（text/file 子消息列表或字符串）转为纯文本"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict):
                ctype = c.get('type')
                if ctype == 'text':
                    parts.append(str(c.get('text', '') or ''))
                elif ctype == 'file':
                    f = c.get('file') or {}
                    name = f.get('name') or f.get('path') or ''
                    parts.append(f"[附件: {name}]")
                else:
                    parts.append(str(c))
            else:
                parts.append(str(c))
        return ' '.join(p for p in parts if p)
    return str(content or '')


def _normalize_history(history):
    """
    将 gr.Chatbot 历史转为 [{role, content}]。
    兼容两种格式：
    - Gradio 6+ 扁平消息列表: [{"role": "user"/"assistant", "content": [{"type":"text","text":...}, ...]}, ...]
    - 旧版问答对: [[user, assistant], ...]（含多模态 dict 消息）
    """
    msgs = []
    for turn in history or []:
        if isinstance(turn, dict) and turn.get('role') in ('user', 'assistant'):
            msgs.append({"role": turn['role'], "content": _message_content_to_text(turn.get('content'))})
        elif isinstance(turn, (list, tuple)) and len(turn) == 2:
            user_msg, bot_msg = turn[0], turn[1]
            # 兼容旧的多模态输入：{"text":..., "files":[...]}
            if isinstance(user_msg, dict):
                user_text = user_msg.get('text', '') or ''
                files = user_msg.get('files') or []
                if files:
                    file_names = ', '.join(os.path.basename(f) for f in files)
                    user_text = f"{user_text}\n[附件: {file_names}]"
            else:
                user_text = str(user_msg or '')
            msgs.append({"role": "user", "content": user_text})
            msgs.append({"role": "assistant", "content": str(bot_msg or '')})
    return msgs


def _extract_topic(note):
    """从笔记中提取主题（与后端 match_kb 逻辑一致），用于自动匹配知识库"""
    if not note:
        return ""
    import re
    m = re.search(r"### Q：(.+)", note)
    if m:
        return m.group(1).strip()
    m = re.search(r"模板：(.+)", note)
    if m:
        return m.group(1).strip()
    return note.strip()[:60]


def summarize_conversation(history):
    """
    第 1~4 步：调用后端 /chat/summarize 按 skill 逻辑整理多轮问答为结构化笔记；
    并自动匹配该问题系列是否属于已有知识库，在状态栏提示导出去向。
    """
    msgs = _normalize_history(history)
    if not msgs:
        return (
            gr.update(value="暂无对话记录，请先在「聊天机器人」中完成问答后再整理。"),
            gr.update(visible=False),
            gr.update(value=""),
        )

    try:
        resp = requests.post(BACKEND_URL + '/chat/summarize', json={"history": msgs}, timeout=300, headers=AUTH_HEADERS)
        if resp.status_code == 200:
            note = resp.json().get('note', '')
        else:
            detail = resp.json().get('detail', '') if resp.headers.get('content-type', '').startswith('application/json') else resp.text
            note = f"❌ 总结失败，状态码: {resp.status_code}, {detail}"
    except Exception as e:
        note = f"❌ 请求异常: {e}"

    # 自动匹配已有知识库（第 5 步准备：判断“属于已有知识库则直接保存”）
    hint = ""
    if note and not note.startswith("❌"):
        try:
            topic = _extract_topic(note)
            if topic:
                r = requests.post(BACKEND_URL + '/knowledge/match_kb', json={"topic": topic}, timeout=30, headers=AUTH_HEADERS)
                if r.status_code == 200:
                    matched = r.json().get('kb_name')
                    if matched:
                        hint = f"💡 该笔记与知识库【{matched}】匹配，导出时将直接保存到该知识库。"
                        print(f"[Summary_UI] 自动匹配知识库: {matched}")
                    else:
                        hint = "💡 未匹配到已有知识库，导出时将自动新建知识库保存。"
        except Exception as e:
            print("[Summary_UI] 匹配知识库异常:", e)

    return (
        gr.update(value=note),
        gr.update(visible=False),
        gr.update(value=hint),
    )


def export_note(note, note_name, kb_choice, new_kb_name):
    """
    第 5 步：用户同意导出时，将笔记保存到知识库。
    - 笔记名称由用户设置（留空用默认名「对话问答笔记」），用于分类复习；
    - 知识库去向：自动识别（默认）/ 手动选择已有知识库 / 手动新建知识库，
      便于按科目、题型等分类归档，扩充题库。
    """
    if not note or note.startswith("❌"):
        return gr.update(value="⚠️ 笔记内容为空或未生成成功，请先在「聊天机器人」点击「总结知识点」整理后再导出。")

    file_name = (note_name or "").strip() or "对话问答笔记"
    if not file_name.endswith(".md"):
        file_name += ".md"

    # 知识库去向判定
    kb_name = ""
    if kb_choice and kb_choice != AUTO_OPTION and kb_choice != NEW_OPTION:
        kb_name = kb_choice                      # 手动选择已有知识库（分类归档）
    elif kb_choice == NEW_OPTION and (new_kb_name or "").strip():
        kb_name = new_kb_name.strip()            # 手动新建知识库（指定分类）
    # AUTO_OPTION 或 NEW 未填名称 → kb_name 留空，由后端自动匹配/自动新建

    payload = {
        "file_name": file_name,
        "content": note,
        "kb_name": kb_name,
        "description": "",
        "chunk_size": 128,
        "chunk_overlap": 20,
    }
    try:
        resp = requests.post(BACKEND_URL + '/knowledge/save_note', json=payload, timeout=120, headers=AUTH_HEADERS)
        if resp.status_code == 200:
            res = resp.json()
            if res.get("appended"):
                created = "同名追加"
                msg = (f"✅ 已追加到笔记【{res['file_name']}】（保留原文件，自动标注追加日期）\n"
                       f"文件: {res['file_name']} (路径: ./data/{res['kb_name']}/{res['file_name']})")
            else:
                created = "新建" if res.get("created") else "已有"
                msg = (f"✅ 笔记已保存到知识库【{res['kb_name']}】({created})\n"
                       f"文件: {res['file_name']} (路径: ./data/{res['kb_name']}/{res['file_name']})")
        else:
            detail = resp.json().get('detail', '') if resp.headers.get('content-type', '').startswith('application/json') else resp.text
            msg = f"❌ 保存失败，状态码: {resp.status_code}, {detail}"
    except Exception as e:
        msg = f"❌ 请求异常: {e}"

    return gr.update(value=msg)
