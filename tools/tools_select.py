from langchain_core.tools import Tool, StructuredTool
from tools.weather_check import weather_check
from tools.web_search import web_search
from tools.knowledge_search import knowledge_base_search


# 将API工具封装成Langchain的Tool对象
# 注意：工具名统一使用下划线风格（如 web_search），与提示词及 LLM 输出习惯保持一致，避免解析失败
tools = [
    Tool(
        name="weather_check",
        func=weather_check.run,
        description="获取当前地点或指定城市的实时天气信息，包括温度、天气状况等。"
    ),
    Tool(
        name="knowledge_search",
        func=knowledge_base_search.run,
        description="搜索本地知识库，获取与特定主题相关的内容。此工具的输入应是一个由 ',' 分隔的两个字符串，第一个字符串表示您需要查询的知识库，"+
        "第二个字符串表示您需要查询的问题。例如，您想要在 '人工智能' 知识库查询关于 '人工智能学习方法'，输入应为 '人工智能,人工智能学习方法'。"
    ),
    Tool(
        name="web_search",
        func=web_search.run,
        description=("联网搜索工具：当问题需要最新/实时信息时（如今天的天气、最新新闻、实时股价/汇率、体育比分、"
                     "近期事件、人物近况、'XX现在怎么样了'、'XX的最新进展'等），或者本地知识库检索后仍无法回答时，"
                     "**必须调用本工具联网搜索**。输入为需要搜索的关键词（简短、准确）。")
    ),
]
