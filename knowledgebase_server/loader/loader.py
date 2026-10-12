import os
from fastapi import HTTPException
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader, CSVLoader


class data_loader:
    def get_loader(self, file_path):
        # 根据文件格式选择适当的loader
        _, file_extension = os.path.splitext(file_path)
        if file_extension == ".txt":
            loader = TextLoader(file_path, encoding='utf-8')
        elif file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_extension == ".md":
            # 使用 TextLoader 而非 UnstructuredMarkdownLoader：
            # 后者首次使用需联网下载依赖（离线/内网环境会报 Name or service not known）
            loader = TextLoader(file_path, encoding='utf-8')
        elif file_extension == ".csv":
            loader = CSVLoader(file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        return loader
