import json
import requests
from pydantic import Field
from configs.setting import TIME_OUT, BACKEND_URL, AUTH_HEADERS

# 知识库搜索工具类
class KnowledgeBaseSearch:
    inputs: str = Field(description="包含知识库名称和需要查询问题的字符串")

    @staticmethod
    def _list_kbs():
        """取真实存在的库名。失败返回空列表，交给上层走原样重试。"""
        try:
            resp = requests.get(f"{BACKEND_URL}/knowledge/queryall_knowledge",
                                headers=AUTH_HEADERS, timeout=TIME_OUT)
            names = resp.json() if resp.status_code == 200 else []
        except (requests.RequestException, ValueError):
            return []
        return [n for n in names if isinstance(n, str)] if isinstance(names, list) else []

    def _resolve_kb(self, raw_name):
        """把模型写的库名对齐到真实存在的库。

        模型经常把库里的内容标题（比如"唐诗三百首"）当成库名传进来，
        后端直接 404，它就会改去联网乱试，白白多跑好几轮。
        这里先做大小写/包含匹配，单个库时直接兜底，都不中才把可用库名回给它。
        """
        names = self._list_kbs()
        if not names:
            return raw_name
        if raw_name in names:
            return raw_name
        lowered = raw_name.strip().lower()
        for name in names:
            if name.lower() == lowered:
                return name
        for name in names:
            if lowered and (lowered in name.lower() or name.lower() in lowered):
                return name
        if len(names) == 1:
            return names[0]
        return raw_name

    def run(self, inputs):
        # 只按第一个逗号切分：模型给的查询本身常带逗号（例如"朗诵,背诵"），
        # 用 split(",") 解出三段会直接 ValueError，整轮 Agent 就废了。
        parts = [p.strip() for p in str(inputs or "").split(",", 1)]
        if len(parts) < 2 or not parts[0]:
            return "调用格式错误：Action Input 必须是「知识库名称,查询问题」两段，用英文逗号分隔。"
        kb_name, query = parts[0], parts[1] or parts[0]
        kb_name = self._resolve_kb(kb_name)
        url = f"{BACKEND_URL}/knowledge/search_kb"

        # 请求体数据
        data = {
            "kb_name": kb_name,
            "query": query
        }

        # 将数据转换为JSON格式
        json_data = json.dumps(data)

        # 设置请求头，指定发送的数据格式为JSON，并在配置了 API_TOKEN 时携带鉴权头
        headers = {
            'Content-Type': 'application/json',
            **AUTH_HEADERS,
        }

        # 检索超时/连接失败也要变成一句可读的 Observation，
        # 不能让异常冒到 AgentExecutor 外面
        try:
            response = requests.post(url, data=json_data, headers=headers, timeout=TIME_OUT)
        except requests.RequestException as exc:
            return f"检索知识库「{kb_name}」失败：{exc.__class__.__name__}"
        print(f'=========response.status_code: {response.status_code}===========')

        if response.status_code == 200:
            data = response.json()

            res = "".join([f"source:\n{d['sources']} \ncontext:\n{d['page_contents']}\n" for d in data['results']])
            # 若后端因重排序模型未就绪而降级，一并带上提示，避免"悄悄变差"没人发现
            if data.get("warning"):
                res = f"[提示] {data['warning']}\n\n{res}"
            return res
        if response.status_code == 404:
            names = self._list_kbs()
            hint = "、".join(names) if names else "（读取失败）"
            return (f"知识库「{kb_name}」不存在。当前可用的知识库：{hint}。"
                    f"请用其中的库名重新调用，格式「知识库名,查询问题」。")
        return f"检索知识库「{kb_name}」失败（HTTP {response.status_code}），请换个库或稍后重试。"


knowledge_base_search = KnowledgeBaseSearch()