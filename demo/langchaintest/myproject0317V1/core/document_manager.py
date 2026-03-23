# demo/langchaintest/myproject0317V1/core/document_manager.py
"""文档管理器：统一文档加载、存储、检索基础能力，供增强检索器复用"""
import os
from typing import List, Dict, Any, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from demo.langchaintest.myproject0317V1.config import VECTOR_STORE_PATH, FILE_REGISTRY_PATH
from demo.langchaintest.myproject0317V1.utils.embedding_utils import create_embeddings
from demo.langchaintest.myproject0317V1.utils.file_utils import FileRegistry, load_documents_from_paths


class DocumentManager:
    """文档管理器：封装文档加载、存储、检索的核心能力"""

    def __init__(
            self,
            vector_store_path: str = VECTOR_STORE_PATH,
            registry_path: str = FILE_REGISTRY_PATH,
            embedding_model_name: str = None
    ):
        # 路径配置
        self.vector_store_path = vector_store_path
        self.registry_path = registry_path

        # 初始化组件
        self.embeddings = create_embeddings(embedding_model_name)
        self.file_registry = FileRegistry(registry_path)
        self.vectorstore = self._init_vectorstore()

        # 缓存：已加载的文档（避免重复加载）
        self.loaded_docs_cache: Dict[str, List[Document]] = {}

    def _init_vectorstore(self) -> Chroma:
        """初始化向量库"""
        if os.path.exists(self.vector_store_path):
            # 加载已有向量库
            return Chroma(
                persist_directory=self.vector_store_path,
                embedding_function=self.embeddings
            )
        else:
            # 创建新向量库
            os.makedirs(self.vector_store_path, exist_ok=True)
            return Chroma(
                persist_directory=self.vector_store_path,
                embedding_function=self.embeddings
            )

    # ========== 文档加载 ==========
    def load_documents(self, paths: List[str], recursive: bool = True) -> List[Document]:
        """加载文档（复用file_utils）"""
        docs = load_documents_from_paths(paths, recursive=recursive)
        # 更新缓存
        for path in paths:
            self.loaded_docs_cache[path] = docs
        return docs

    # ========== 文档存储 ==========
    def add_documents(self, docs: List[Document], update_registry: bool = True) -> List[str]:
        """添加文档到向量库"""
        if not docs:
            return []

        # 添加到向量库
        doc_ids = self.vectorstore.add_documents(docs)

        # 更新文件注册表（记录文档ID和修改时间）
        if update_registry:
            for doc, doc_id in zip(docs, doc_ids):
                source = doc.metadata.get("source", "unknown")
                mtime = os.path.getmtime(source) if os.path.exists(source) else 0
                self.file_registry.update_file(source, mtime, [doc_id])

        # Deleted:# 持久化向量库
        # Deleted:self.vectorstore.persist()
        # 新版 ChromaDB 自动持久化，无需手动调用 persist()

        return doc_ids

    def update_vectorstore(self, docs: List[Document]) -> List[str]:
        """更新向量库（增量更新，避免重复）"""
        # 过滤已存在的文档（通过source和内容哈希）
        new_docs = []
        for doc in docs:
            source = doc.metadata.get("source")
            if source and not self.file_registry.needs_update(source):
                continue  # 文件未修改，跳过
            new_docs.append(doc)

        # 添加新文档
        return self.add_documents(new_docs)

    def delete_documents(self, doc_ids: List[str]) -> bool:
        """删除指定ID的文档"""
        try:
            self.vectorstore.delete(ids=doc_ids)
            self.vectorstore.persist()
            # 更新注册表
            for doc_id in doc_ids:
                for file_path, file_info in self.file_registry.data.items():
                    if doc_id in file_info.get("ids", []):
                        file_info["ids"].remove(doc_id)
                        self.file_registry.update_file(file_path, file_info["mtime"], file_info["ids"])
            return True
        except Exception as e:
            print(f"删除文档失败：{e}")
            return False

    # ========== 基础检索 ==========
    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        """基础相似性检索（适配统一检索接口）"""
        return self.vectorstore.similarity_search(query, k=k)

    def get_retriever(self, k: int = 3) -> VectorStoreRetriever:
        """获取基础检索器（供增强检索器复用）"""
        return self.vectorstore.as_retriever(search_kwargs={"k": k})

    # ========== 辅助方法 ==========
    def get_all_documents(self) -> Dict[str, Any]:
        """获取向量库中所有文档信息"""
        return self.vectorstore.get()

    def clear_vectorstore(self) -> bool:
        """清空向量库"""
        try:
            # 删除向量库文件
            for root, dirs, files in os.walk(self.vector_store_path):
                for file in files:
                    os.remove(os.path.join(root, file))
            # 重置向量库
            self.vectorstore = self._init_vectorstore()
            # 清空注册表
            self.file_registry.data = {}
            self.file_registry._save()
            # 清空缓存
            self.loaded_docs_cache = {}
            return True
        except Exception as e:
            print(f"清空向量库失败：{e}")
            return False

    def get_document_count(self) -> int:
        """获取向量库中文档数量"""
        return len(self.get_all_documents()["ids"])

    def get_document_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """获取指定文档的元数据"""
        all_docs = self.get_all_documents()
        idx = all_docs["ids"].index(doc_id) if doc_id in all_docs["ids"] else -1
        if idx >= 0:
            return all_docs["metadatas"][idx]
        return None

    def split_documents(self, docs: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
        """
        切分文档为小片段（适配向量存储）
        :param docs: 原始文档列表
        :param chunk_size: 每个片段的最大字符数
        :param chunk_overlap: 片段间重叠字符数
        :return: 切分后的文档片段列表
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        # 初始化文本切分器（语义感知的递归切分）
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],  # 适配中文的分隔符
            length_function=len
        )

        # 切分文档
        split_docs = text_splitter.split_documents(docs)

        # 补充元数据（记录原始文档信息）
        for i, doc in enumerate(split_docs):
            doc.metadata["chunk_id"] = i
            doc.metadata["chunk_size"] = chunk_size
            doc.metadata["chunk_overlap"] = chunk_overlap

        return split_docs