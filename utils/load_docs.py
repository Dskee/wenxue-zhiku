"""文件内容读取工具：按扩展名分派加载器，返回纯文本内容（用于对话中上传文件的知识增强）。"""
import os


def get_file_content(file_path: str) -> str:
    """读取本地文件内容，返回字符串。支持 .txt/.md/.csv/.pdf，其他扩展名按文本尝试解码。"""
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader
        docs = PyPDFLoader(file_path).load()
        return "\n".join(d.page_content for d in docs)

    if ext == ".csv":
        from langchain_community.document_loaders import CSVLoader
        docs = CSVLoader(file_path).load()
        return "\n".join(d.page_content for d in docs)

    # .txt / .md 及其他文本类：按编码依次尝试
    for enc in ("utf-8", "gbk", "gb18030"):
        try:
            with open(file_path, "r", encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, LookupError):
            continue
    with open(file_path, "rb") as f:
        return f.read().decode("utf-8", errors="ignore")
