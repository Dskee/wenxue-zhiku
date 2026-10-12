# 网络搜索API工具类
import requests
from pydantic import Field
from bs4 import BeautifulSoup
from configs.setting import TIME_OUT, serpapi_key

# SerpAPI Key 来自环境变量 / .env（原为写死在源码里的明文 Key，已移除）
SERPAPI_KEY = serpapi_key

# 单次搜索的超时上限（秒）：SerpAPI/必应任一超过即快速失败回退，避免 agent 被长时间挂起
SEARCH_TIMEOUT = 10


class WebSearch:
    query: str = Field(description="需要网上查找的内容")

    def __init__(self, api_key):
        # 初始化函数，用于创建类的实例
        self.api_key = api_key

    def _search_serpapi(self, query):
        """主通道：SerpAPI（baidu 引擎）。返回 (结果字符串, None) 或 (None, 错误提示)。"""
        base_url = "https://serpapi.com/search"
        params = {
            "q": query,
            "api_key": self.api_key,
            "engine": "baidu",
            "rn": 5,         # 检索结果前5个
            "proxy": "http://api.wlai.vip"  # 代理（用了这个好像不会减使用次数）
        }
        response = requests.get(base_url, params=params, timeout=SEARCH_TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            # SerpAPI 对无结果的查询会返回 error 字段（例如 "Baidu hasn't returned any results for this query."）
            if data.get('error'):
                return None, f"网络搜索未返回结果：{data['error']}。请更换关键词重试，或改用其他方式（如直接计算、本地知识库检索）回答该问题。"
            organic_results = data.get('organic_results') or []
            if not organic_results:
                return None, "网络搜索未找到与该问题直接相关的结果。请更换关键词重试，或改用其他方式（如直接计算、本地知识库检索）回答该问题。"
            organic_results = organic_results[0].get('related_news') if organic_results[0].get('related_news') else organic_results
            results = "".join([f"titles:\n{result.get('title', '')} \nlinks:\n{result.get('link', '')}\nsnippets:\n{result.get('snippet', '')}\n" for result in organic_results])
            return results, None
        return None, f"网络搜索服务暂不可用（HTTP {response.status_code}）。"

    def _search_bing(self, query):
        """回退通道：直接抓取必应搜索结果页解析（无需 API key，SerpAPI 额度不足时自动使用）"""
        url = "https://cn.bing.com/search"
        params = {"q": query, "setlang": "zh-hans"}
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
            "Accept-Language": "zh-CN,zh;q=0.9",
        }
        response = requests.get(url, params=params, headers=headers, timeout=SEARCH_TIMEOUT)
        if response.status_code != 200:
            return f"网络搜索服务暂不可用（HTTP {response.status_code}）。请稍后重试。"
        soup = BeautifulSoup(response.text, "html.parser")
        blocks = soup.select("li.b_algo")[:5]
        if not blocks:
            return "网络搜索未找到与该问题直接相关的结果。请更换关键词重试，或改用其他方式（如直接计算、本地知识库检索）回答该问题。"
        results = []
        for b in blocks:
            h2 = b.select_one("h2 a")
            if not h2:
                continue
            title = h2.get_text(strip=True)
            link = h2.get("href", "")
            p = b.select_one("p")
            snippet = p.get_text(strip=True) if p else ""
            results.append(f"titles:\n{title} \nlinks:\n{link}\nsnippets:\n{snippet}\n")
        if not results:
            return "网络搜索未找到与该问题直接相关的结果。请更换关键词重试，或改用其他方式（如直接计算、本地知识库检索）回答该问题。"
        return "".join(results)

    def run(self, query):
        # 主通道：SerpAPI；额度不足/异常时自动回退必应抓取
        # 未配置 SERPAPI_KEY 时直接走回退通道，避免发出必然失败的请求
        err = None
        if self.api_key:
            try:
                result, err = self._search_serpapi(query)
                if result:
                    return result
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
        else:
            err = "未配置 SERPAPI_KEY"
        try:
            return self._search_bing(query)
        except Exception as e:
            detail = f"{type(e).__name__}: {e}"
            return f"网络搜索失败（{detail}；SerpAPI: {err}）。请稍后重试，或改用其他方式（如直接计算、本地知识库检索）回答该问题。"


web_search = WebSearch(SERPAPI_KEY)
