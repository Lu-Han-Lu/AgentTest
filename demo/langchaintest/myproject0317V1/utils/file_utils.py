# demo/langchaintest/myproject0317V1/utils/file_utils.py
# utils/file_utils.py
import json
import os
from typing import Dict, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
    TextLoader,
    WebBaseLoader
)
import glob

try:
    from langchain_community.document_loaders import UnstructuredWordDocumentLoader

    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    print("警告：未安装 unstructured 库，.doc 文件将无法加载。运行：pip install unstructured")


class FileRegistry:
    """管理已处理文件的注册表"""

    def __init__(self, registry_path: str = "./file_registry.json"):
        self.registry_path = registry_path
        self.data = self._load()

    def _load(self) -> Dict:
        """加载注册表文件"""
        if os.path.exists(self.registry_path):
            with open(self.registry_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save(self):
        """保存注册表到文件"""
        with open(self.registry_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def get_file_info(self, file_path: str) -> Optional[Dict]:
        """获取文件的记录，包含 mtime 和 ids"""
        return self.data.get(file_path)

    def update_file(self, file_path: str, mtime: float, ids: List[str]):
        """更新或添加文件记录"""
        self.data[file_path] = {"mtime": mtime, "ids": ids}
        self._save()

    def remove_file(self, file_path: str):
        """移除文件记录"""
        if file_path in self.data:
            del self.data[file_path]
            self._save()

    def needs_update(self, file_path: str) -> bool:
        """检查文件是否需要处理（不存在或修改时间变化）"""
        if not os.path.exists(file_path):
            return False
        current_mtime = os.path.getmtime(file_path)
        info = self.get_file_info(file_path)
        return info is None or info["mtime"] != current_mtime


def detect_file_type(file_path: str) -> str:
    """检测文件类型"""
    ext = os.path.splitext(file_path)[1].lower()
    ext_map = {
        '.pdf': 'pdf',
        '.docx': 'docx', '.doc': 'doc',
        '.xlsx': 'xlsx', '.xls': 'xls',
        '.pptx': 'pptx', '.ppt': 'ppt',
        '.html': 'html', '.htm': 'html',
        '.md': 'markdown',
        '.txt': 'text',
        '.csv': 'csv',
        '.json': 'json',
        '.xml': 'xml',
    }
    return ext_map.get(ext, 'unknown')


def load_single_document(file_path: str, **kwargs) -> List[Document]:
    """
    根据文件类型加载单个文档，返回Document列表
    """
    file_type = detect_file_type(file_path)
    print(f"检测到文件类型: {file_type}, 路径: {file_path}")

    try:
        if file_type == 'pdf':
            loader = PyPDFLoader(file_path)
            return loader.load()

        elif file_type == 'docx':
            loader = Docx2txtLoader(file_path)
            return loader.load()

        elif file_type == 'doc':  # .doc 是二进制格式，需要特殊处理
            if UNSTRUCTURED_AVAILABLE:
                print(f"使用 UnstructuredWordDocumentLoader 加载：{file_path}")
                loader = UnstructuredWordDocumentLoader(file_path)
                return loader.load()
            else:
                print(f"跳过 .doc 文件（需要安装 unstructured 库）: {file_path}")
                return []

        elif file_type == 'xlsx':
            loader = UnstructuredExcelLoader(file_path, mode="elements")
            return loader.load()

        elif file_type == 'pptx':
            loader = UnstructuredPowerPointLoader(file_path)
            return loader.load()

        elif file_type == 'html':
            loader = UnstructuredHTMLLoader(file_path)
            return loader.load()

        elif file_type == 'markdown':
            loader = UnstructuredMarkdownLoader(file_path)
            return loader.load()

        elif file_type == 'csv':
            loader = CSVLoader(file_path)
            return loader.load()

        elif file_type == 'json':
            from langchain_community.document_loaders.json_loader import JSONLoader
            loader = JSONLoader(file_path, jq_schema='.', text_content=False)
            return loader.load()

        elif file_type == 'text':
            loader = TextLoader(file_path, encoding='utf-8')
            return loader.load()

        else:
            # 未知格式尝试用TextLoader
            print(f"未知格式，尝试作为文本加载: {file_path}")
            loader = TextLoader(file_path, encoding='utf-8', autodetect_encoding=True)
            return loader.load()

    except Exception as e:
        print(f"加载文件失败 {file_path}: {e}")
        return []


def load_documents_from_paths(
        paths: List[str],
        recursive: bool = True,
        glob_pattern: str = "**/*.*"
) -> List[Document]:
    """
    从多个文件路径或目录加载文档
    - paths: 可以是文件路径列表，也可以是目录路径列表
    - recursive: 是否递归子目录
    - glob_pattern: 文件匹配模式
    """
    all_docs = []

    for path in paths:
        if os.path.isfile(path):
            # 单个文件
            docs = load_single_document(path)
            all_docs.extend(docs)
            print(f"已加载 {len(docs)} 个文档块 from {path}")

        elif os.path.isdir(path):
            # 目录：遍历所有文件
            print(f"扫描目录: {path}")
            file_list = glob.glob(os.path.join(path, glob_pattern), recursive=recursive)
            for file_path in file_list:
                docs = load_single_document(file_path)
                all_docs.extend(docs)
        else:
            print(f"路径不存在: {path}")

    return all_docs


def load_web_page(url: str) -> List[Document]:
    """
    从网页URL加载内容
    """
    loader = WebBaseLoader(url)
    docs = loader.load()
    # 添加来源URL到元数据
    for doc in docs:
        doc.metadata["source"] = url
        doc.metadata["source_type"] = "web"
    return docs
