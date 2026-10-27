"""生成层：固定参数调用对话模型（temperature=0，单轮，无历史）。

只评测检索时用 `--no-generate` 跳过；拒答指标必须走这里，因为要靠真实模型输出。
模型 / Key / BaseURL 都从项目 `configs.runtime_config` 读，和线上同一套配置来源。
"""

from typing import List, Tuple

from eval.prompts import SYSTEM_PROMPT, build_user_prompt

GENERATION_TEMPERATURE = 0.0
GENERATION_MAX_TOKENS = 512
GENERATION_TIMEOUT = 120

# Qwen3 系列默认开启思考（enable_thinking），在本项目实测：同一个问题 107.7s vs 关闭后 1.0s，
# 而且 512 的 max_tokens 经常被思考过程吃光、正文只剩半句。评测要看的是「拿到这些资料后
# 答案对不对」，不是思维链长度，所以这里显式关闭；请求参数不影响线上 /chat 的行为。
EXTRA_BODY = {"enable_thinking": False}


def build_chat_model():
    from langchain_openai import ChatOpenAI

    from configs import runtime_config

    return ChatOpenAI(
        model=runtime_config.get("chat_model"),
        api_key=runtime_config.require_api_key("chat"),
        base_url=runtime_config.base_url_for("chat") or None,
        temperature=GENERATION_TEMPERATURE,
        max_tokens=GENERATION_MAX_TOKENS,
        timeout=GENERATION_TIMEOUT,
        max_retries=2,
        extra_body=EXTRA_BODY,
    )


def _flatten_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text", "")))
            else:
                parts.append(str(part))
        return "".join(parts)
    return "" if content is None else str(content)


def generate_answer(model, question: str, documents) -> Tuple[str, float]:
    """返回 (回答文本, 耗时秒)。异常向上抛，由调用方记为该样本失败而不是静默吞掉。"""
    import time

    from langchain_core.messages import HumanMessage, SystemMessage

    messages: List = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_user_prompt(question, documents)),
    ]
    started = time.perf_counter()
    response = model.invoke(messages)
    elapsed = time.perf_counter() - started
    return _flatten_content(getattr(response, "content", "")).strip(), elapsed