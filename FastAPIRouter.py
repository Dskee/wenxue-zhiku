"""FastAPI 聚合入口：挂载知识库管理、聊天、总结等全部路由。
启动：uvicorn FastAPIRouter:app --host 0.0.0.0 --port 16055
"""
import secrets

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from chat.ChatFastAPI import ChatAIapp
from knowledgebase_server.loader.KnowledgeFastAPI import knowledgeapp, kb_router
from settings_server.SettingsFastAPI import settingsapp
from configs.setting import API_TOKEN, CORS_ORIGINS
from configs.runtime_config import MissingApiKeyError, ProviderUnavailableError

app = FastAPI(title="问学知库 API", version="1.0.0")


# 统一的「可读错误」处理：
# 以前缺 API Key / Key 失效 / 模型名写错 / 服务连不上，都会变成裸的
# 500 Internal Server Error，使用者完全不知道哪里出了问题。
# 现在统一转成 503 + 中文说明（并附带当前配置，便于自查）。
@app.exception_handler(MissingApiKeyError)
async def handle_missing_api_key(request: Request, exc: MissingApiKeyError):
    return JSONResponse(status_code=503,
                        content={"detail": str(exc), "error": "missing_api_key"})


@app.exception_handler(ProviderUnavailableError)
async def handle_provider_unavailable(request: Request, exc: ProviderUnavailableError):
    return JSONResponse(status_code=503,
                        content={"detail": str(exc), "error": "provider_unavailable"})

# 跨域：只放行配置的来源（默认本机 Gradio 前端）。
# 原实现为 allow_origins=["*"]，任意网站都能跨域调用本后端的删除/写入接口。
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# 可选接口鉴权：仅在设置了 API_TOKEN 时启用（默认关闭，保持本地单机开箱即用）。
# 后端监听 0.0.0.0，若所在网络不可信，请在 .env 中设置 API_TOKEN，
# 并让前端带上 X-API-Token 请求头（configs.setting.AUTH_HEADERS 已统一提供）。
_OPEN_PATHS = {"/", "/docs", "/redoc", "/openapi.json", "/favicon.ico"}


@app.middleware("http")
async def verify_api_token(request: Request, call_next):
    if not API_TOKEN or request.method == "OPTIONS" or request.url.path in _OPEN_PATHS:
        return await call_next(request)
    supplied = request.headers.get("X-API-Token") or ""
    # compare_digest：避免用 == 比较令牌带来的时序侧信道
    if not secrets.compare_digest(supplied, API_TOKEN):
        return JSONResponse(status_code=401, content={"detail": "Unauthorized: 缺少或错误的 X-API-Token"})
    return await call_next(request)


# 知识库管理路由（/knowledge/*）
app.include_router(knowledgeapp)
app.include_router(kb_router)

# 聊天与总结路由（/chat/*）
app.include_router(ChatAIapp)

# 设置路由（/settings/*）：模型与 API Key 的读取 / 保存 / 模型列表
app.include_router(settingsapp)


@app.get("/")
async def root():
    return {"msg": "问学知库后端服务运行中", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=16055)
