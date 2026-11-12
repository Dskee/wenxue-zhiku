# 问学知库 · Web 前端

零依赖的浏览器界面，替代原来的 Gradio 页面。`server.js` 同时做两件事：

1. 托管 `public/` 下的静态文件（HTML/CSS/JS）；
2. 把 `/api/**` 原样反代到 FastAPI 后端。

因为页面和接口同源，**不需要改动后端的 CORS 配置**，也不需要额外开端口。

## 运行

```bash
# 1. 先起后端（默认 16055）
python FastAPIRouter.py

# 2. 再起前端
cd web
node server.js
```

打开 <http://127.0.0.1:7860> 即可。

## 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `PORT` | `7860` | 前端监听端口 |
| `HOST` | `0.0.0.0` | 监听地址 |
| `API_BASE` | `http://127.0.0.1:16055` | FastAPI 地址 |
| `API_TOKEN` | 空 | 若后端设置了 `API_TOKEN`，填同样的值，代理会带上 `X-API-Token` |

示例（后端跑在另一台机器上）：

```bash
API_BASE=http://192.168.1.10:16055 node server.js
```

## 目录

```
web/
├── server.js          静态托管 + API 反向代理
├── package.json       只声明元信息，没有任何 npm 依赖
└── public/
    ├── index.html     单页结构（问答 / 知识库 / 文件 / 整理 / 出题 / Anki / 设置）
    ├── style.css      主题与布局
    └── app.js         所有交互逻辑，原生 JS，无需构建
```

## 说明

- 聊天是**流式**的：后端返回 NDJSON，代理逐块转发，页面一边收一边渲染。
- 文件上传走 multipart，代理是流式转发，不落临时文件。
- 后端没起来时，页面底部会显示「后端未连接」，并给出中文提示，而不是白屏。
- 访问 `/healthz` 可以快速确认前端与后端的连通性。
