import json

import gradio as gr
import requests

from configs.setting import AUTH_HEADERS, BACKEND_URL


def chatai_to_backend(prompt, history, sys_prompt, history_len, temperature, max_tokens, stream, session_id, kb_name=""):
  files = [('files', (open(file_path, 'rb'))) for file_path in prompt['files']]
  if prompt['files'] != []:
    query = f"{prompt['text']}\n" + ''.join(prompt['files'][0])
  else:
    query = f"{prompt['text']}\n"
  payload = {
    'query': query,
    'history': [str(h) for h in history],
    'sys_prompt': sys_prompt,
    'history_len': history_len,
    'temperature': temperature,
    'max_tokens': max_tokens,
    'session_id': session_id,  # 将唯一ID传递给后端
    'kb_name': kb_name or ""   # 指定知识库；留空则后端按相关性自动选择
  }

  try:
    response = requests.post(f"{BACKEND_URL}/chat/chatbot", files=files, data=payload, stream=True, timeout=360, headers=AUTH_HEADERS)
    if response.status_code != 200:
      yield '请求失败，请检查后台服务器是否正常运行。'
      return

    # 后端返回 NDJSON：每行一个 {"answer": ...}。
    # 修复：早期实现直接对每个网络 chunk 调 json.loads，一旦一个 JSON 对象被 TCP
    # 拆成两个 chunk（token 多、网络抖动时很常见）就会抛异常、回答中断。
    # 现在先按 \n 切行缓存，只有拿到完整一行才解析，天然容忍任意拆包方式。
    chunks = ""
    buffer = ""
    # 注意：不要用 iter_content(decode_unicode=True) —— 它依赖响应的 charset，
    # 而 application/x-ndjson 不带 charset，requests 会回退成 bytes，直接与 str 拼接会 TypeError。
    # 这里统一按字节读取并显式 UTF-8 解码，两种响应类型都稳。
    for chunk in response.iter_content(chunk_size=None):
      if not chunk:
        continue
      if isinstance(chunk, (bytes, bytearray)):
        chunk = chunk.decode('utf-8', errors='ignore')
      buffer += chunk
      while "\n" in buffer:
        line, buffer = buffer.split("\n", 1)
        line = line.strip()
        if not line:
          continue
        try:
          data = json.loads(line)
        except json.JSONDecodeError:
          # 单行不是合法 JSON（后端异常输出等）：跳过该行，不要中断整段回答
          continue
        chunks += data.get('answer', '')
        if stream:
          yield chunks

    # 处理结尾可能没有换行的残留内容
    tail = buffer.strip()
    if tail:
      try:
        data = json.loads(tail)
        chunks += data.get('answer', '')
      except json.JSONDecodeError:
        pass
    if not stream:
      # 非流式：累积完毕后一次性给出
      yield chunks

  except Exception as e:
    yield f'请求异常: {str(e)}'
