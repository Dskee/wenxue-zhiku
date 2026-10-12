"""天气查询工具：基于 wttr.in 免费接口，无需 API Key。"""
import requests
from pydantic import Field


class WeatherCheck:
    inputs: str = Field(description="城市名称，如：沈阳")

    def run(self, inputs):
        city = str(inputs or "").strip()
        if not city:
            return "请提供城市名称"
        try:
            url = f"https://wttr.in/{city}?format=j1&lang=zh"
            r = requests.get(url, timeout=15)
            if r.status_code != 200:
                return f"无法获取 {city} 的天气信息（接口返回 {r.status_code}）"
            data = r.json()
            cur = data["current_condition"][0]
            # 中文描述可能位于 lang_zh 子节点，缺失时回退 weatherDesc
            desc_zh = ""
            try:
                desc_zh = cur["lang_zh"][0]["value"]
            except (KeyError, IndexError, TypeError):
                try:
                    desc_zh = cur["weatherDesc"][0]["value"]
                except (KeyError, IndexError, TypeError):
                    desc_zh = "未知"
            return (f"{city}当前天气：{desc_zh}，温度 {cur['temp_C']}°C，"
                    f"体感 {cur['FeelsLikeC']}°C，湿度 {cur['humidity']}%，"
                    f"风速 {cur['windspeedKmph']}km/h")
        except Exception as e:
            return f"天气查询失败: {type(e).__name__}: {e}"


weather_check = WeatherCheck()
