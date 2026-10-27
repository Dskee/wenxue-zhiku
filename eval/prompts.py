"""评测用的固定 prompt。

和线上 `/chat/chatbot` 的 Agent prompt 是两回事：评测要的是「同样的输入换检索结果」，
不能再让 Agent 自己决定调不调工具、查几轮。所以这里给一个固定的、只读资料的单轮 prompt：

- 只用给定资料，禁止模型自带知识（否则 no_answer 类样本测不出拒答）
- 温度 0、不给历史、max_tokens 固定
- 拒答必须逐字输出 REFUSAL_MARKER，方便用规则判定，不用 LLM 当裁判
"""

import re
from typing import Dict, Sequence

REFUSAL_MARKER = "知识库中未找到相关内容。"

SYSTEM_PROMPT = (
    "你是严谨的知识库问答助手，只能依据用户给出的【资料】回答。\n"
    "规则：\n"
    "1. 禁止使用【资料】以外的知识，禁止编造、禁止猜测；\n"
    "2. 先给结论，再给依据，依据后标注资料编号，例如 [1]；\n"
    "3. 如果【资料】不足以回答问题，只输出这一行：" + REFUSAL_MARKER + " 不要解释，不要补充任何其他内容。"
)

MAX_CONTEXT_CHARS = 3000

_REFUSAL_CORES = ("未找到相关内容", "知识库中未找到")


def format_contexts(documents: Sequence) -> str:
    """把检索结果排成编号资料块；超长直接截断，防止 prompt 无上限膨胀。"""
    blocks = []
    used = 0
    for index, document in enumerate(documents, start=1):
        source = (getattr(document, "metadata", None) or {}).get("source", "")
        source = str(source).replace("\\", "/").rsplit("/", 1)[-1]
        text = re.sub(r"\n{3,}", "\n\n", (document.page_content or "").strip())
        block = f"[{index}] 来源：{source}\n{text}"
        if used + len(block) > MAX_CONTEXT_CHARS:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def build_user_prompt(question: str, documents: Sequence) -> str:
    contexts = format_contexts(documents)
    if not contexts:
        contexts = "（没有检索到任何资料）"
    return f"【资料】\n{contexts}\n\n【问题】\n{question}"


def is_refusal(answer: str) -> bool:
    """规则判定是否拒答。

    只看回答的**开头**：真实模型经常先正常作答，末尾补一句「知识库中未找到…的具体描述」
    当免责声明，那种回答不能算拒答（实测踩过：正确答案被判成拒答，误拒率虚高到 100%）。
    空回答不算拒答，单独统计。
    """
    compact = re.sub(r"\s+", "", answer or "").lstrip("：:，,。.")
    return any(compact.startswith(core) for core in _REFUSAL_CORES)


def describe_prompt() -> Dict[str, object]:
    """报告里记录生成层口径，保证可追溯。"""
    return {
        "system_prompt": SYSTEM_PROMPT,
        "refusal_marker": REFUSAL_MARKER,
        "max_context_chars": MAX_CONTEXT_CHARS,
    }