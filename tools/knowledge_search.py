import json
import requests
from pydantic import Field
from configs.setting import TIME_OUT, BACKEND_URL, AUTH_HEADERS

# 知识库搜索工具类
class KnowledgeBaseSearch:
    inputs: str = Field(description="包含知识库名称和需要查询问题的字符串")

    def run(self, inputs):
        kb_name, query = inputs.split(",")
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

        response = requests.post(url, data=json_data, headers=headers, timeout=TIME_OUT)
        print(f'=========response.status_code: {response.status_code}===========')

        if response.status_code == 200:
            data = response.json()

            res = "".join([f"source:\n{d['sources']} \ncontext:\n{d['page_contents']}\n" for d in data['results']])
            # 若后端因重排序模型未就绪而降级，一并带上提示，避免"悄悄变差"没人发现
            if data.get("warning"):
                res = f"[提示] {data['warning']}\n\n{res}"
            return res
        else:
            return f"无法获取{query}的信息"


knowledge_base_search = KnowledgeBaseSearch()