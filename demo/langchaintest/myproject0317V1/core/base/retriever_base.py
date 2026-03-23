# demo/langchaintest/myproject0317V1/core/base/retriever_base.py
"""统一检索接口：供RAG直接使用，也供Agent作为工具调用"""
from abc import ABC, abstractmethod
from typing import List

from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_core.documents import Document
# 注释掉 CohereRerank 导入（避免依赖）
# from langchain_classic.retrievers.document_compressors import CohereRerank
from langchain_community.retrievers import BM25Retriever

from demo.langchaintest.myproject0317V1.core.document_manager import DocumentManager


class BaseRetriever(ABC):
    """检索器基类：定义统一接口"""

    @abstractmethod
    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        """核心检索方法"""
        pass

    @abstractmethod
    def get_retriever(self):
        """获取LangChain Retriever对象"""
        pass


class EnhancedRetriever(BaseRetriever):
    """增强检索器：混合检索（语义+关键词）+ 重排序（可选）+ 上下文压缩"""

    def __init__(self, doc_manager: DocumentManager, use_rerank: bool = False):  # 默认禁用重排序
        self.doc_manager = doc_manager
        self.use_rerank = use_rerank  # 改为默认False
        self.base_retriever = doc_manager.get_retriever()
        self.enhanced_retriever = self._build_enhanced_retriever()

    def _build_enhanced_retriever(self):
        """构建增强检索器"""
        # 1. 构建BM25关键词检索器（补充语义检索的不足）
        try:
            # 容错：如果向量库为空，直接返回基础检索器
            all_docs = self.doc_manager.get_all_documents()
            if not all_docs["documents"]:
                return self.base_retriever

            bm25_retriever = BM25Retriever.from_texts(all_docs["documents"])
            bm25_retriever.k = 3
        except Exception as e:
            print(f"BM25检索器初始化失败，仅使用基础语义检索：{e}")
            return self.base_retriever

        # 2. 混合检索（语义+关键词）
        ensemble_retriever = EnsembleRetriever(
            retrievers=[self.base_retriever, bm25_retriever],
            weights=[0.7, 0.3]  # 语义检索权重更高
        )

        # 3. 重排序（可选，默认禁用）
        if self.use_rerank:
            try:
                # 增加容错：允许传入API Key或从环境变量读取
                import os
                cohere_api_key = os.environ.get("COHERE_API_KEY")
                if not cohere_api_key:
                    raise ValueError("COHERE_API_KEY 未配置")
                from langchain_classic.retrievers.document_compressors import CohereRerank
                compressor = CohereRerank(top_n=3, cohere_api_key=cohere_api_key)
                compression_retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=ensemble_retriever
                )
                return compression_retriever
            except Exception as e:
                print(f"重排序功能初始化失败，使用混合检索：{e}")
                return ensemble_retriever
        else:
            return ensemble_retriever

    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        """统一检索接口"""
        return self.enhanced_retriever.invoke(query)[:k]

    def get_retriever(self):
        """返回LangChain Retriever对象"""
        return self.enhanced_retriever