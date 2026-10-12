"""自定义流式回调处理器：把 LLM 流式输出的 token 放入异步队列，供前端 SSE 逐字读取。"""
import asyncio
from typing import Any, AsyncIterator
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult


class CustomAsyncIteratorCallbackHandler(AsyncCallbackHandler):
    """
    流式 token 收集器：
    - on_llm_new_token：把每个 token 放入队列（put_nowait，不阻塞 LLM）；
    - 通过 pending 计数跟踪当前正在执行的 LLM 调用数（Agent 多轮 Thought/Action 会多次调用 LLM）；
    - 当所有调用都结束时（pending 归零）放入哨兵 None，aiter() 据此结束。
    """

    def __init__(self) -> None:
        self.queue: asyncio.Queue = asyncio.Queue()
        self._pending: int = 0
        # 记录最后一次工具调用返回（Observation），供 agent 停止/异常时兜底输出搜索结果
        self.last_tool_output: str = ""

    @property
    def always_verbose(self) -> bool:
        return True

    async def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        """捕获工具执行结果，agent 停止时用作兜底答案素材。"""
        if output:
            self.last_tool_output = str(output)[:1500]

    async def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs: Any) -> None:
        self._pending += 1

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if token is not None:
            self.queue.put_nowait(token)

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self._pending -= 1
        if self._pending <= 0:
            self.queue.put_nowait(None)

    async def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        self._pending -= 1
        if self._pending <= 0:
            self.queue.put_nowait(None)

    async def aiter(self) -> AsyncIterator[str]:
        """异步迭代所有 token，遇到哨兵 None 即结束。"""
        while True:
            token = await self.queue.get()
            if token is None:
                break
            yield token
