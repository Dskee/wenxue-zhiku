import json
import requests
import gradio as gr
import os
from configs.setting import BACKEND_URL, AUTH_HEADERS

def add(name, description):
    url = BACKEND_URL + '/knowledge/add_knowledge'
    data = {
        "name": name,
        "description": description
    }
    try:
        response = requests.post(url, json=data, headers=AUTH_HEADERS)
        print("====create knowledge response:", response)
        print("原始响应内容:", response.text)
        if response.status_code == 200:
            return gr.update(value="知识库创建成功！", visible=True), gr.update(value=""), gr.update(value="")
        else:
            return gr.update(value=f"知识库创建失败: {response.json().get('detail','')}", visible=True), gr.update(value=""), gr.update(value="")
    except Exception as e:
        return gr.update(value=f"请求失败: {e}", visible=True), gr.update(value=""), gr.update(value="")


def delete(kb_name):
    """
    后端：name:str = Body(...)，不允许传 {"name":"xxx"}对象
    必须直接把字符串裸放body，不能包字典！
    """
    print(f"[DEBUG]待删除知识库名称: {kb_name}")
    url = BACKEND_URL + '/knowledge/del_knowledge'
    headers = {"Content-Type": "application/json"}
    try:

        response = requests.delete(url, params={"name": kb_name}, headers=AUTH_HEADERS)
        print("====delete knowledge response:", response)
        print("====delete knowledge response.json:", response.json())
        if response.status_code == 200:
            return gr.update(value="知识库删除成功！", visible=True), gr.update(value="")
        else:
            return gr.update(value=f"删除失败: {response.json().get('detail','')}", visible=True), gr.update(value="")
    except Exception as e:
        return gr.update(value=f"请求失败: {e}", visible=True), gr.update(value="")


def query_list():
    url = BACKEND_URL + '/knowledge/queryall_knowledge'
    response = requests.get(url, headers=AUTH_HEADERS)
    print("====list knowledge response:", response)
    if response.status_code == 200:
        knowledge_name_list = response.json()
        print("知识库name列表：", knowledge_name_list)
        if len(knowledge_name_list) > 0:
            return gr.update(choices=knowledge_name_list, value=knowledge_name_list[0])
        else:
            return gr.update(choices=[], value=None)
    else:
        return gr.update(choices=[], value=None)


selected_files = []  # 用于存储已选择的文件列表
def update(file):
    global selected_files
    if file:
        # 累积 + 去重：支持一次多选，也支持分次选择累积；相同文件不重复添加
        for f in file:
            if f not in selected_files:
                selected_files.append(f)
    return gr.update(value=selected_files, visible=True), gr.update(value=[])


def upload_docs(kb_name_upload, file_uploaded, chunk_size, chunk_overlap):
    upload_docs_url = BACKEND_URL + '/knowledge/upload_docs'
    files = [('files', (open(file_path, 'rb'))) for file_path in file_uploaded if os.path.getsize(file_path) > 0 ] # 过滤掉空文件
    if not files:
        return gr.update(value=f'上传文件内容为空', visible=True), gr.update(value=[], visible=False)

    data = {
        'kb_name': kb_name_upload,
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap
    }
    global selected_files
    selected_files = []  # 清空已选择的文件列表
    try:
        responses = requests.post(upload_docs_url, files=files, data=data, headers=AUTH_HEADERS)

        if responses.status_code == 200:
            return gr.update(value='文件上传成功', visible=True), gr.update(value=[], visible=False)
        else:
            return gr.update(value=f"文件上传失败,状态码：{responses.status_code}, 错误信息：{responses.json().get('detail', '')}", visible=True), gr.update(value=[], visible=False)

    except Exception as e:
        return gr.update(value=f'请求失败: {e}', visible=True), gr.update(value=[], visible=False)


def delete_selected_file(kb_drop, file_drop):
    print(f"[DEBUG] kb_drop={repr(kb_drop)}, file_drop={repr(file_drop)}")
    delete_file_url = BACKEND_URL + '/knowledge/delete_file'

    if kb_drop is None or file_drop is None:
        print("[DEBUG] 触发空值拦截")
        return (
            gr.update(value='⚠️ 请先选择知识库和待删除文件', visible=True),
            gr.update(choices=[], value=None)
        )

    data = {
        "kb_name": kb_drop,
        "file_name": file_drop
    }
    try:
        resp = requests.post(delete_file_url, json=data, timeout=5, headers=AUTH_HEADERS)
        print(f"[DEBUG] delete resp:{resp.status_code}, {resp.text}")
        if resp.status_code == 200:
            return (
                gr.update(value="✅文件删除成功", visible=True),
                gr.update(choices=[], value=None)
            )
        else:
            detail = resp.json().get('detail', '')
            return (
                gr.update(value=f"❌文件删除失败,状态码:{resp.status_code}, {detail}", visible=True),
                gr.update(value=None)
            )
    except Exception as e:
        print(f"[DEBUG] 请求异常 {e}")
        return (
            gr.update(value=f"❌请求失败：{e}", visible=True),
            gr.update(value=None)
        )


def get_kb_file_list(kb_name):
    list_file_url = BACKEND_URL + '/knowledge/list_files'
    if not kb_name:
        return gr.update(choices=[], value=None)

    data = {
        "kb_name": kb_name
    }
    try:
        resp = requests.post(list_file_url, params={"kb_name": kb_name}, headers=AUTH_HEADERS)
        print(f"[get_kb_file_list] http状态码: {resp.status_code}")
        if resp.status_code == 200:
            res_data = resp.json()
            file_list = res_data.get("files", [])
            return gr.update(choices=file_list, value=None)
        else:
            return gr.update(choices=[], value=None)
    except Exception as e:
        print("获取文件列表异常：", e)
        return gr.update(choices=[], value=None)


def read_file_content(kb_name, file_name):
    """查看指定知识库内文件（笔记）的正文内容，用于复习梳理知识点"""
    if not kb_name or not file_name:
        return gr.update(value="⚠️ 请先选择知识库和文件", visible=True)
    try:
        resp = requests.post(BACKEND_URL + '/knowledge/read_file',
                             json={"kb_name": kb_name, "file_name": file_name}, timeout=30, headers=AUTH_HEADERS)
        if resp.status_code == 200:
            content = resp.json().get("content", "")
            return gr.update(value=content, visible=True)
        else:
            detail = resp.json().get('detail', '') if resp.headers.get('content-type', '').startswith('application/json') else resp.text
            return gr.update(value=f"❌ 读取失败: {detail}", visible=True)
    except Exception as e:
        return gr.update(value=f"❌ 请求异常: {e}", visible=True)


def update_file_content(kb_name, file_name, content):
    """修改指定知识库内文件（笔记）的正文内容，并重建知识库向量索引"""
    if not kb_name or not file_name:
        return gr.update(value="⚠️ 请先选择知识库和文件", visible=True)
    if not content or not content.strip():
        return gr.update(value="⚠️ 笔记内容为空，无法保存", visible=True)
    try:
        resp = requests.post(BACKEND_URL + '/knowledge/update_file',
                             json={"kb_name": kb_name, "file_name": file_name, "content": content}, timeout=120, headers=AUTH_HEADERS)
        if resp.status_code == 200:
            res = resp.json()
            rebuilt = res.get("rebuilt_chunks", 0)
            msg = f"✅ 笔记已保存修改（重建向量索引 {rebuilt} 块）\n文件: {res['file_name']} (路径: ./data/{res['kb_name']}/{res['file_name']})"
            return gr.update(value=msg, visible=True)
        else:
            detail = resp.json().get('detail', '') if resp.headers.get('content-type', '').startswith('application/json') else resp.text
            return gr.update(value=f"❌ 保存修改失败: {detail}", visible=True)
    except Exception as e:
        return gr.update(value=f"❌ 请求异常: {e}", visible=True)
