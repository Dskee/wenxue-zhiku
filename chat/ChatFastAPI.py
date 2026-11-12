import json
import asyncio
import os
from ast import literal_eval
from fastapi import APIRouter, UploadFile, Form, Body, HTTPException
from fastapi.responses import StreamingResponse
from tools.tools_select import tools
from typing import List, AsyncIterable
from langchain_openai import ChatOpenAI
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.agents import AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from configs.prompt import PROMPT_TEMPLATES
from configs.setting import (
    base_url,
    TIME_OUT,
    TEMP_FILE_STORAGE_DIR,
    MAX_UPLOAD_MB,
)
from configs import runtime_config
from utils.callback import CustomAsyncIteratorCallbackHandler
from db_server.data.KnowledgeBase import session
from db_server.data.KnowledgeDB import queryall
from utils.load_docs import get_file_content
from utils.paths import safe_filename

ChatAIapp = APIRouter(prefix='/chat', tags=['chatai'])

# 说明：密钥与 base_url 统一来自 configs.setting（环境变量 / .env），源码中不再出现明文 Key。
# 原文件此处曾再次写死 api_key，重复且会覆盖配置导入，已移除。
aclient = None  # 预留：如需独立 OpenAI 异步客户端，请在首次使用时按需创建


def _strip_invented_observation(text: str) -> str:
    """截掉模型自己"脑补"的 Observation 段。

    ReAct 里 Observation 只能由系统在工具执行后给出。但小模型经常顺着格式往下写，
    把一段假的工具结果也吐出来；这段假结果会被解析器当成正常内容带进下一轮，
    使后续推理全建立在没发生过的检索上。规则：只有在它前面已经出现过 Action 行时，
    才认为这是编造的 Observation，并从该行截断。
    """
    lines = str(text or "").splitlines()
    seen_action = False
    keep = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Action:") or stripped.startswith("Action Input:"):
            seen_action = True
        if seen_action and stripped.startswith("Observation:"):
            break
        keep.append(line)
    return "\n".join(keep).strip()


class LenientReActParser(ReActSingleInputOutputParser):
    """在官方 ReAct 解析器上放宽两处小模型常见的不规范写法。

    1. 自己编造 Observation（见上）；
    2. 只写了 Final Answer、没写 Action —— 官方解析器会因为"Thought 后面找不到 Action"
       直接抛错，5 轮之后整段回答变成 "Agent stopped due to iteration limit"。
    """

    def parse(self, text: str):
        cleaned = _strip_invented_observation(text)
        try:
            return super().parse(cleaned)
        except OutputParserException:
            if "Final Answer:" in cleaned:
                answer = cleaned.split("Final Answer:", 1)[1].strip()
                if answer:
                    return AgentFinish({"output": answer}, cleaned)
            raise


_LENIENT_PARSER = LenientReActParser()

_FINAL_MARK = "Final Answer:"


class AnswerOnlyStream:
    """只把 Final Answer 之后的文字发给前端。

    ReAct 的 Thought / Action / Observation 是给解析器看的中间过程，
    以前它们也被逐 token 推给用户，界面上就是一大段看不懂的英文协议，
    看起来像"卡住了"或"答非所问"。这里做一个简单的状态机：
    标记出现之前的内容全部吞掉，只保留可能与标记前缀重叠的尾巴。
    """

    def __init__(self):
        self._buf = ""
        self._started = False

    def feed(self, token: str) -> str:
        if not token:
            return ""
        if self._started:
            return token
        self._buf += token
        idx = self._buf.find(_FINAL_MARK)
        if idx >= 0:
            self._started = True
            out = self._buf[idx + len(_FINAL_MARK):]
            self._buf = ""
            return out
        # 丢掉确定不是标记开头的部分，只留可能是标记前缀的结尾
        keep = 0
        limit = min(len(_FINAL_MARK) - 1, len(self._buf))
        for n in range(limit, 0, -1):
            if _FINAL_MARK.startswith(self._buf[-n:]):
                keep = n
                break
        self._buf = self._buf[-keep:] if keep else ""
        return ""

    def flush(self) -> str:
        """整段回答没有写 Final Answer 时（例如直接寒暄），把缓存原样交出来。"""
        out = "" if self._started else self._buf
        self._buf = ""
        return out



def _chat_model_extra() -> dict:
    """按服务商决定是否关闭 Qwen3 的"思考"。

    Qwen3 默认先花几百 token 在 reasoning 上（实测一次 510 token 的回复里有 390 是思考），
    ReAct 是文本协议，输出预算被思考吃掉后格式会被截断，表现为
    "Missing 'Action:' after 'Thought:'" 反复重试、最终超时。已知支持的服务商显式关掉；
    本地 vLLM / Ollama 不认这个字段，传了反而报 400，所以不加。
    """
    url = (runtime_config.base_url_for("chat") or base_url or "").lower()
    if "siliconflow" in url or "dashscope" in url or "aliyuncs" in url:
        return {"extra_body": {"enable_thinking": False}}
    return {}



def files_rag(files, uuid):
    """把**本次请求**上传的文件落盘并读取正文，作为本轮问答的知识增强上下文。

    修复：早期实现先写入本轮文件，然后 os.listdir() 遍历整个会话临时目录，
    导致同一会话里历史上传过的文件在之后每一轮提问时都被重复注入 prompt
    （上下文随轮次线性膨胀、又慢又贵）。现在只处理本次传入的 files。
    """
    if not files:
        return ""

    # 定义存储路径；uuid 来自前端，必须清洗，避免 session_id 携带 ../ 写出目录外
    kb_file_storage_path = os.path.join(TEMP_FILE_STORAGE_DIR, safe_filename(uuid or "default"))
    os.makedirs(kb_file_storage_path, exist_ok=True)

    result = []
    limit = MAX_UPLOAD_MB * 1024 * 1024
    for file in files:
        if file is None or not getattr(file, "filename", None):
            continue
        # filename 清洗，防止 ../ 路径穿越覆盖项目文件
        safe_name = safe_filename(file.filename)
        if not safe_name:
            continue
        file_path = os.path.join(kb_file_storage_path, safe_name)
        # 大小限制：边读边写，超过上限立即中止，避免超大文件打满磁盘
        written = 0
        with open(file_path, "wb") as f:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > limit:
                    f.close()
                    os.remove(file_path)
                    raise HTTPException(
                        status_code=413,
                        detail=f"文件 {safe_name} 超过 {MAX_UPLOAD_MB}MB 上传上限",
                    )
                f.write(chunk)
        file_content = get_file_content(file_path)
        result.append(f"document:{file_path}\ncontent:{file_content}\n\n")
    return "".join(result)

@ChatAIapp.post('/chatbot')
async def chatai(
        files: List[UploadFile] = None,
        query: str = Form(..., description="用户输入"),
        sys_prompt: str = Form("you are a helpful assistant.", description="系统提示"),
        history_len: int = Form(-1, description="保留历史消息的数量"),
        history: List[str] = Form([], description="历史对话"),
        temperature: float = Form(0.5, description="LLM采样温度"),
        max_tokens: int = Form(1024, description="LLM最大token数配置"),
        session_id: str = Form(None, description="会话标识"),
        kb_name: str = Form("", description="用户指定的知识库名称，空则自动按相关性选择"),
                 ):
    # 提前校验 Key：流式响应一旦开始返回就无法再改状态码，
    # 所以必须在创建 StreamingResponse 之前检查，否则用户只会看到一段断掉的流。
    runtime_config.require_api_key("chat")

    documents = files_rag(files, session_id)
    # 将字符串形式的 Python 数据结构（如 '{"key": "value"}' 或 '[1, 2, 3]' 或 '42'）安全地转换为实际的 Python 数据类型（例如字典、列表、整数等）
    history = [literal_eval(item) for item in history]
    # 控制历史记录长度，保留指定数量的消息
    histories = ""
    if history_len > 0:
        history = history[-2 * history_len:]
    # 在消息列表中加入历史消息
    for msg in history:
        role = msg.get("role")
        content = msg.get("content")
        histories += f"{role}:{content}\n\n"

    async def agent_chat_iterator() -> AsyncIterable[str]:
        # 使用自己定义的回调处理函数
        callback = CustomAsyncIteratorCallbackHandler()
        callbacks = [callback]

        # 定义聊天模型
        chat_model = ChatOpenAI(
            # 模型 / Key / 地址都在**调用时**从 runtime_config 读取：
            # 在「设置」页改完立刻生效，无需重启服务；本地模型不需要 Key
            model=runtime_config.get("chat_model"),
            api_key=runtime_config.require_api_key("chat"),
            base_url=runtime_config.base_url_for("chat") or base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=True,
            callbacks=callbacks,
            request_timeout=TIME_OUT,  # 兜底：LLM 单次请求超时，防止 API 慢时无限挂起
            **_chat_model_extra(),
        )

        system_prompt = SystemMessagePromptTemplate.from_template(
                    sys_prompt
                )

                # 定义人类提示，用于用户输入
        human_prompt = HumanMessagePromptTemplate.from_template(
                    PROMPT_TEMPLATES["agent"]
                )


        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
        
        # 定义系统提示，用于设置行为规则
       
        # 组装成完整的 ChatPromptTemplate


        # 创建代理：输出解析器用放宽版，容忍模型编造 Observation / 漏写 Action
        agent = create_react_agent(
            chat_model,
            tools,
            chat_prompt,
            output_parser=_LENIENT_PARSER,
            stop_sequence=["\nObservation"],
        )
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,  # 关键！格式解析错误自动兜底
            max_iterations=5,  # 给足轮次：工具调用后模型偶发再输出 Action，需留出重试空间避免提前停止
        )

        # 列出所有的知识库（限制注入数量+截断描述，控制 prompt 长度，避免知识库过多时变慢）
        knowledgebases = queryall(session)
        if not knowledgebases:
            kbs = "暂无知识库"
        elif kb_name:
            # 用户在前端指定了知识库：只注入该库，让 Agent 必须优先检索它
            target = [kb for kb in knowledgebases if kb["name"] == kb_name]
            if target:
                kb = target[0]
                kbs = f"用户指定知识库：kb_name:{kb['name']} - {kb['description'][:40]}"
            else:
                kbs = (f"用户指定的知识库「{kb_name}」不存在，请告知用户。可用知识库：\n"
                       + "\n".join([f"kb_name:{kb['name']} - {kb['description'][:40]}" for kb in knowledgebases[:10]]))
        else:
            # 未指定：按与问题的相关性排序注入 top 10（库名/描述与问题关键词重叠度）
            kbs = _rank_kbs(knowledgebases, query)
        print(f"[DEBUG] 准备启动agent, query={query}")
        # 开启异步任务执行agent
        # asyncio.create_task()用于创建一个新的异步任务。

        task = asyncio.create_task(
            agent_executor.ainvoke(
                 input={"input": query, "history": histories, "knowledgebases": kbs, "documents": documents}

            )
        )
        print(f"[DEBUG] agent task已创建")


        streamed_text = []
        answer_stream = AnswerOnlyStream()
        # 不能直接用 callback.aiter()：它在每轮 LLM 结束时都会收到 None 哨兵，
        # 而 Agent 是按轮调用 LLM 的（Thought/Action → 工具 → 再 Thought/Final Answer），
        # 一旦第一轮就 break，后面真正的回答就再也读不到，只能等任务结束后整段补发。
        # 这里给任务挂一个 done 回调当作"真正的结束哨兵"，每轮之间的 None 跳过不退出。
        # 注意不能用 asyncio.wait_for 包 queue.get() 来轮询：取消可能把已经取出的 token 丢掉。
        task.add_done_callback(lambda _t: callback.queue.put_nowait(None))
        while True:
            token = await callback.queue.get()
            if token is None:
                if task.done() and callback.queue.empty():
                    break
                continue
            # 过滤空 token：硅基流动 Qwen3 思考模型的流式 chunk 常出现 content 为空的中间块
            if not token:
                continue
            # 只转发 Final Answer 正文，中间推理过程不外泄
            visible = answer_stream.feed(token)
            if not visible:
                continue
            streamed_text.append(visible)
            response_data = {
                "answer": visible,
            }
            # 每条 JSON 以 \n 结尾（NDJSON）：前端按行切分后再解析，
            # 避免一个 JSON 对象被 TCP 拆成两个 chunk 时 json.loads 抛异常导致回答中断
            yield (json.dumps(response_data) + "\n").encode('utf-8')

        tail = answer_stream.flush()
        if tail:
            streamed_text.append(tail)
            yield (json.dumps({"answer": tail}) + "\n").encode('utf-8')

        # 等待agent任务执行完毕
        await asyncio.wait_for(task, TIME_OUT)
        # 获取agent执行最终返回结果
        try:
            agent_result = task.result()
            final_answer = agent_result.get("output", "")
        except Exception as e:
            print(f"[DEBUG] agent任务异常: {type(e).__name__}: {e}")
            final_answer = ""
        # AgentExecutor 达到迭代上限时会把这句英文当成最终答案，对用户毫无意义，
        # 统一按"没答出来"处理，改用已检索到的内容兜底
        stopped = final_answer and "Agent stopped" in final_answer
        if stopped:
            final_answer = ""
        emitted = "".join(streamed_text).strip()
        # 兜底输出：若最终答案尚未出现在流式内容中（多轮 agent 的第二轮 LLM 输出偶发
        # 未走流式、或流式 chunk 全空），则补发完整最终答案，避免用户看不到回答且不重复
        if final_answer and final_answer.strip() not in emitted:
            yield (json.dumps({"answer": final_answer}) + "\n").encode('utf-8')
        elif not final_answer and not emitted and callback.last_tool_output:
            # 最坏情况：agent 因迭代/时间限制停止且无最终答案，把最后一次检索结果兜底给用户
            yield (json.dumps({"answer": f"（模型没能给出最终答案，下面是已经检索到的内容，可以先参考）\n{callback.last_tool_output}"}) + "\n").encode('utf-8')
        elif not emitted:
            yield (json.dumps({"answer": "（这次没能生成回答，换个问法或检查一下 API Key / 模型配置再试）"}) + "\n").encode('utf-8')


    # NDJSON：每行一个 JSON 对象，前端按行解析
    return StreamingResponse(agent_chat_iterator(), media_type="application/x-ndjson")
        
        # 流式输出
        
from pydantic import BaseModel, Field

class SummarizeReq(BaseModel):
    """对话问答笔记整理请求体"""
    history: List[dict] = Field(default_factory=list, description="多轮对话历史 [{role, content}]")
    sys_prompt: str = "you are a helpful assistant."


@ChatAIapp.post('/summarize')
async def summarize_conversation(req: SummarizeReq):
    """
    对话问答笔记整理器（对应 TraeCode「对话问答笔记整理器」skill）：
    将用户与模型的多次问答按「回溯分类  提炼最优提问模板  过滤保留  按序整页输出」整理为结构化笔记。
    """
    history = req.history
    if not history:
        return {"note": "暂无对话记录，无法生成笔记。"}

    # 组装历史文本（超长保护：避免超长历史导致 LLM 总结耗时过长/上下文溢出）
    hist_lines = []
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        hist_lines.append(f"{role}: {content}")
    hist_text = "\n\n".join(hist_lines)
    truncated = False
    MAX_HIST_CHARS = 20000
    if len(hist_text) > MAX_HIST_CHARS:
        hist_text = hist_text[:MAX_HIST_CHARS]
        truncated = True

    # 非流式调用模型完成总结，temperature 调低保证准确还原；request_timeout 兜底防卡死
    chat_model = ChatOpenAI(
        model=runtime_config.get("chat_model"),
        api_key=runtime_config.require_api_key("chat"),
        base_url=runtime_config.base_url_for("chat") or base_url,
        temperature=0.2,
        max_tokens=4096,
        streaming=False,
        request_timeout=180,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", PROMPT_TEMPLATES["summary"]),
        ("human", "以下是用户与模型的完整多轮对话记录，请据此整理笔记：\n\n{history}"),
    ])
    chain = prompt | chat_model
    try:
        result = await chain.ainvoke({"history": hist_text})
        note = result.content if hasattr(result, "content") else str(result)
    except Exception as e:
        note = f"❌ 总结失败: {type(e).__name__}: {e}"
    if truncated:
        note += "\n\n> ⚠️ 对话记录过长，本次仅整理了前 20000 字符对应的内容。"
    return {"note": note}


def _rank_kbs(knowledgebases, query):
    """按知识库与问题的相关性排序（jieba 分词后，库名/描述与问题关键词的重叠度），返回 top 10 的注入文本。"""
    try:
        import jieba
        q_words = set(jieba.lcut_for_search(query))
    except Exception:
        q_words = set(query)

    scored = []
    for kb in knowledgebases:
        text = f"{kb['name']} {kb.get('description') or ''}"
        try:
            import jieba
            k_words = set(jieba.lcut_for_search(text))
        except Exception:
            k_words = set(text)
        overlap = len(q_words & k_words)
        # 库名被问题直接提及 加权，优先
        if kb["name"] in query:
            overlap += 5
        scored.append((overlap, kb))
    scored.sort(key=lambda x: -x[0])
    top = scored[:10]
    return "\n".join([f"kb_name:{kb['name']} - {kb['description'][:40]}" for _, kb in top])