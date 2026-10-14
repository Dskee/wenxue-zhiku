"""设置页的前端请求逻辑：模型来源、模型与 API Key 的读取 / 保存 / 拉取列表 / 连接测试。
与 Chat_UI / Knowledge_UI 等保持一致：只负责 HTTP 请求逻辑，不持有 UI 组件。
"""
import requests
import gradio as gr

from configs.setting import BACKEND_URL, AUTH_HEADERS

# 来源预设缓存（进程内复用，避免每次切换都请求）
_PROFILE_CACHE: dict = {}


def _url(path: str = "") -> str:
    return f"{BACKEND_URL}/settings{path}"


def _detail(resp):
    try:
        return resp.json().get("detail", resp.text[:200])
    except Exception:
        return resp.text[:200]


def _get_json(path: str, timeout: int = 20):
    r = requests.get(_url(path), timeout=timeout, headers=AUTH_HEADERS)
    if r.status_code != 200:
        raise RuntimeError(_detail(r))
    return r.json()


def _post_json(path: str, payload: dict, timeout: int = 60):
    r = requests.post(_url(path), json=payload, timeout=timeout, headers=AUTH_HEADERS)
    return r.status_code, (r.json() if r.content else {})


def _profiles() -> dict:
    global _PROFILE_CACHE
    if not _PROFILE_CACHE:
        try:
            _PROFILE_CACHE = _get_json("/profiles")
        except Exception:
            _PROFILE_CACHE = {"default": "siliconflow", "profiles": {}}
    return _PROFILE_CACHE


def profile_choices():
    """给下拉框用的 (显示名, 实际值) 列表。"""
    data = _profiles()
    items = [(v.get("label", k), k) for k, v in (data.get("profiles") or {}).items()]
    return items


def _summarize(s: dict) -> str:
    src = s.get("profile_label") or s.get("profile") or "(未设置)"
    chat = s.get("chat_model") or "(未设置)"
    emb = s.get("embedding_model") or "(未设置)"
    if s.get("has_key"):
        key = f"已配置 {s.get('api_key_masked', '')}"
    elif s.get("needs_key"):
        key = "**未配置**（保存后即可调用）"
    else:
        key = "不需要（本地模型）"
    emb_src = ""
    if s.get("embedding_uses_own_config"):
        emb_src = "（向量化用独立地址）"
    return (f"**来源**　{src}　|　**对话模型** `{chat}`　|　**向量模型** `{emb}`{emb_src}　|　**API Key**：{key}")


def load_settings():
    """页面加载时读取后端设置。

    **不回填真实 API Key**（接口本身也只返回打码值），密码框永远留空，留空即"不修改"。
    返回 6 项：来源 / 对话模型 / 向量模型 / 地址 / 向量化独立地址 / 状态。
    """
    blank = gr.update()
    try:
        s = _get_json("")
    except Exception as e:
        return blank, blank, blank, blank, blank, f"❌ 读取设置失败：{e}"

    return (
        gr.update(value=s.get("profile")),
        gr.update(value=s.get("chat_model") or ""),
        gr.update(value=s.get("embedding_model") or ""),
        gr.update(value=s.get("base_url") or ""),
        gr.update(value=s.get("embedding_base_url") or ""),
        _summarize(s),
    )


def on_profile_change(profile_key):
    """切换「模型来源」时：自动填好该来源的默认接口地址，并显示使用提示。"""
    p = (_profiles().get("profiles") or {}).get(profile_key) or {}
    hint = p.get("hint") or ""
    if hint:
        hint = f"💡 **{p.get('label', profile_key)}**：{hint}"
    return gr.update(value=p.get("base_url") or ""), hint


def _apply_models(d, kind, current):
    models = list(d.get("chat_models") if kind == "chat" else d.get("embedding_models") or [])
    value = (current or "").strip() or (models[0] if models else None)
    if d.get("source") == "provider":
        msg = f"✅ 已从 `{d.get('base_url', '')}` 拉取到 {d.get('total', len(models))} 个模型"
        if kind == "chat":
            msg += f"（其中对话模型 {len(models)} 个）"
    else:
        msg = f"⚠️ 未能获取真实列表，已用内置候选：{d.get('error') or '未知原因'}"
    return gr.update(choices=models, value=value), msg


def fetch_chat_models(current_chat_model):
    """从【对话】来源拉取模型列表。"""
    try:
        d = _get_json("/models?kind=chat", timeout=90)
    except Exception as e:
        return gr.update(), f"❌ 拉取对话模型列表失败：{e}"
    return _apply_models(d, "chat", current_chat_model)


def fetch_embedding_models(current_embedding_model):
    """从【向量化】来源拉取模型列表（可能配了独立地址）。"""
    try:
        d = _get_json("/models?kind=embedding", timeout=90)
    except Exception as e:
        return gr.update(), f"❌ 拉取向量模型列表失败：{e}"
    return _apply_models(d, "embedding", current_embedding_model)


def save_settings(profile, chat_model, api_key, base_url, embedding_model,
                  use_sep_embedding, emb_base_url, emb_api_key):
    """保存设置。**留空的项表示不修改**（尤其是两个密码框）。

    取消「向量化使用独立配置」勾选时，会显式清空独立的地址与 Key，回到"跟随对话"。
    """
    payload = {
        "profile": (profile or "").strip() or None,
        "chat_model": (chat_model or "").strip() or None,
        "api_key": (api_key or "").strip() or None,
        "base_url": (base_url or "").strip() or None,
        "embedding_model": (embedding_model or "").strip() or None,
        "embedding_base_url": (emb_base_url or "").strip() or None,
        "embedding_api_key": (emb_api_key or "").strip() or None,
    }
    if not use_sep_embedding:
        payload["clear"] = ["embedding_base_url", "embedding_api_key"]

    try:
        code, d = _post_json("", payload, timeout=40)
    except Exception as e:
        return f"❌ 请求异常：{type(e).__name__}: {e}"
    if code != 200:
        return f"❌ 保存失败：{d.get('detail') or d}"
    return f"✅ {d.get('msg')}\n\n{_summarize(d.get('settings') or {})}"


def save_api_key(api_key):
    """侧边栏专用：只保存 API Key，其余设置不动。"""
    if not (api_key or "").strip():
        return "⚠️ 请先填写 API Key（留空不会修改任何设置）"
    return save_settings(None, None, api_key, None, None, True, None, None)


def test_connection():
    """真实调用一次对话模型与向量化模型，验证当前设置是否可用。"""
    lines = []
    for kind, title in (("chat", "对话模型"), ("embedding", "向量化模型")):
        try:
            _, d = _post_json("/test", {"kind": kind}, timeout=120)
        except Exception as e:
            lines.append(f"❌ **{title}**：请求异常 {type(e).__name__}: {e}")
            continue
        if d.get("ok"):
            lines.append(f"✅ **{title}**：{d.get('detail')}")
        else:
            lines.append(f"❌ **{title}**：{d.get('detail')}")
            if d.get("hint"):
                lines.append(f"　　↳ {d['hint']}")
    return "\n\n".join(lines)
