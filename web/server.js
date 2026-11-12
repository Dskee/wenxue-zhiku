#!/usr/bin/env node
/*
 * 问学知库 · Web 前端服务
 *
 * 只做两件事：
 *   1) 把 public/ 下的静态文件发出去；
 *   2) 把 /api/** 反向代理到 FastAPI 后端（默认 http://127.0.0.1:16055）。
 *
 * 为什么用代理而不是让浏览器直连 16055：浏览器里前端在 7860，后端在 16055，
 * 是两个源，要么后端放开 CORS，要么前端代理。代理更简单，也不用为了前端
 * 去放松后端的安全设置。
 *
 * 不依赖任何 npm 包，`node server.js` 直接跑。因为代理用的是流式转发，
 * 聊天的逐 token 输出和文件上传都是边收边转，不会被缓冲成一整块。
 */
'use strict';

const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const { URL } = require('node:url');

const PORT = Number(process.env.PORT || 7860);
const HOST = process.env.HOST || '0.0.0.0';
const API_BASE = (process.env.API_BASE || 'http://127.0.0.1:16055').replace(/\/+$/, '');
const API_TOKEN = process.env.API_TOKEN || '';
const PUBLIC_DIR = path.join(__dirname, 'public');
const API_PREFIX = '/api';

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.ico': 'image/x-icon',
  '.woff2': 'font/woff2',
  '.apkg': 'application/octet-stream',
};

function send(res, status, body, headers) {
  const payload = Buffer.isBuffer(body) ? body : Buffer.from(String(body), 'utf-8');
  res.writeHead(status, Object.assign({ 'Content-Type': 'text/plain; charset=utf-8' }, headers || {}));
  res.end(payload);
}

function apiTarget() {
  const u = new URL(API_BASE);
  return { protocol: u.protocol, hostname: u.hostname, port: u.port || '80' };
}

function proxy(req, res, url) {
  // /api/knowledge/status -> /knowledge/status
  const upstreamPath = (url.pathname.slice(API_PREFIX.length) || '/') + url.search;
  const target = apiTarget();

  const headers = Object.assign({}, req.headers);
  headers.host = target.hostname + (target.port === '80' ? '' : ':' + target.port);
  delete headers.connection;
  if (API_TOKEN) headers['x-api-token'] = API_TOKEN;

  const upstream = http.request(
    {
      protocol: target.protocol,
      hostname: target.hostname,
      port: target.port,
      method: req.method,
      path: upstreamPath,
      headers,
    },
    (up) => {
      const out = Object.assign({}, up.headers);
      // 让中间层（如果有反向代理）不要缓冲，否则流式输出会整段憋到最后
      out['x-accel-buffering'] = 'no';
      res.writeHead(up.statusCode || 502, out);
      up.pipe(res);
    }
  );

  upstream.on('error', (err) => {
    if (res.headersSent) {
      res.destroy();
      return;
    }
    send(
      res,
      502,
      JSON.stringify({
        detail:
          '连不上后端服务（' + API_BASE + '）：' + err.message +
          '\n请先启动 FastAPI：uvicorn FastAPIRouter:app --host 0.0.0.0 --port 16055',
      }),
      { 'Content-Type': 'application/json; charset=utf-8' }
    );
  });

  // 客户端提前断开时，把上游请求也取消，别让后端白跑
  req.on('aborted', () => upstream.destroy());
  req.pipe(upstream);
}

function serveStatic(req, res, url) {
  let rel;
  try {
    rel = decodeURIComponent(url.pathname);
  } catch {
    return send(res, 400, 'bad path');
  }
  if (rel === '/' || rel === '') rel = '/index.html';

  const filePath = path.join(PUBLIC_DIR, path.normalize(rel));
  const resolved = path.resolve(filePath);
  // 路径穿越防护：解析后必须仍在 public/ 里
  if (resolved !== PUBLIC_DIR && !resolved.startsWith(PUBLIC_DIR + path.sep)) {
    return send(res, 403, 'forbidden');
  }

  fs.readFile(resolved, (err, data) => {
    if (err) {
      // 不认识的路由一律交回单页应用（方便以后加前端路由）
      return fs.readFile(path.join(PUBLIC_DIR, 'index.html'), (e2, html) => {
        if (e2) return send(res, 404, 'not found');
        res.writeHead(200, { 'Content-Type': MIME['.html'], 'Cache-Control': 'no-cache' });
        res.end(html);
      });
    }
    res.writeHead(200, {
      'Content-Type': MIME[path.extname(resolved).toLowerCase()] || 'application/octet-stream',
      'Cache-Control': 'no-cache',
    });
    res.end(req.method === 'HEAD' ? undefined : data);
  });
}

const server = http.createServer((req, res) => {
  const url = new URL(req.url, 'http://' + (req.headers.host || 'localhost'));

  if (url.pathname === '/healthz') {
    return send(res, 200, JSON.stringify({ ok: true, api: API_BASE }), {
      'Content-Type': 'application/json; charset=utf-8',
    });
  }
  if (url.pathname === API_PREFIX || url.pathname.startsWith(API_PREFIX + '/')) {
    return proxy(req, res, url);
  }
  return serveStatic(req, res, url);
});

server.listen(PORT, HOST, () => {
  console.log('问学知库前端已启动： http://' + (HOST === '0.0.0.0' ? '127.0.0.1' : HOST) + ':' + PORT);
  console.log('后端 API 代理目标：' + API_BASE + (API_TOKEN ? '（已附带 X-API-Token）' : ''));
});

process.on('SIGINT', () => {
  console.log('\n已停止');
  process.exit(0);
});
