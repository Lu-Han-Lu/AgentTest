# demo/langchaintest/myproject0317V1/core/rag/retriever_ext.py
"""检索器扩展：实现多粒度检索、路由检索等高级功能"""
from typing import List

from langchain_classic.retrievers import MultiQueryRetriever
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from demo.langchaintest.myproject0317V1.core.base.llm_base import LLMManager
from demo.langchaintest.myproject0317V1.core.base.retriever_base import BaseRetriever, EnhancedRetriever


class MultiGranularityRetriever(BaseRetriever):
    """多粒度检索器：粗粒度（文档级）+ 细粒度（段落级）"""

    def __init__(self, doc_manager, coarse_k: int = 2, fine_k: int = 3):
        self.doc_manager = doc_manager
        self.coarse_k = coarse_k  # 粗粒度检索文档数
        self.fine_k = fine_k  # 细粒度检索段落数
        self.base_retriever = EnhancedRetriever(doc_manager)

    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        """多粒度检索"""
        # 1. 粗粒度检索：获取相关文档
        coarse_docs = self.base_retriever.retrieve(query, self.coarse_k)
        doc_ids = [doc.metadata.get("source") for doc in coarse_docs]

        # 2. 细粒度检索：在相关文档内检索段落
        fine_docs = []
        for doc_id in doc_ids:
            # 过滤当前文档的段落
            fine_retriever = self.doc_manager.get_retriever()
            fine_results = fine_retriever.invoke(query)
            fine_docs.extend([d for d in fine_results if d.metadata.get("source") == doc_id])

        # 3. 去重并返回
        unique_docs = []
        seen_content = set()
        for doc in fine_docs[:k]:
            content = doc.page_content[:100]  # 取前100字符去重
            if content not in seen_content:
                seen_content.add(content)
                unique_docs.append(doc)
        return unique_docs

    def get_retriever(self):
        """适配LangChain Retriever接口"""

        def _retrieve(query: str):
            return self.retrieve(query)

        return _retrieve


class RoutingRetriever(BaseRetriever):
    """路由检索器：根据问题类型选择不同检索策略"""

    def __init__(self, doc_manager, llm_manager: LLMManager = None):
        self.doc_manager = doc_manager
        self.llm_manager = llm_manager or LLMManager()
        self.enhanced_retriever = EnhancedRetriever(doc_manager)
        self.multi_gran_retriever = MultiGranularityRetriever(doc_manager)

        # 路由提示词
        self.routing_prompt = PromptTemplate(
            template="""请判断用户问题的类型，仅返回以下关键词之一：
1. general：通用问题（无需精准检索）
2. technical：技术问题（需要多粒度精准检索）
3. conversational：闲聊问题（无需检索）

用户问题：{query}
判断结果：""",
            input_variables=["query"]
        )

    def _route_query(self, query: str) -> str:
        """路由判断问题类型"""
        prompt = self.routing_prompt.format(query=query)
        result = self.llm_manager.invoke(prompt).strip().lower()
        # 兜底处理
        if result not in ["general", "technical", "conversational"]:
            return "general"
        return result

    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        """根据路由结果选择检索策略"""
        route_type = self._route_query(query)

        if route_type == "technical":
            return self.multi_gran_retriever.retrieve(query, k)
        elif route_type == "general":
            return self.enhanced_retriever.retrieve(query, k)
        else:  # conversational
            return []

    def get_retriever(self):
        """适配LangChain Retriever接口"""

        def _retrieve(query: str):
            return self.retrieve(query)

        return _retrieve


class MultiQueryEnhancedRetriever(BaseRetriever):
    """多查询检索器：生成多个查询词提升召回率"""

    def __init__(self, doc_manager, llm_manager: LLMManager = None):
        self.doc_manager = doc_manager
        self.llm_manager = llm_manager or LLMManager()
        self.base_retriever = self.doc_manager.get_retriever()

        # 多查询检索器
        self.multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=self.base_retriever,
            llm=self.llm_manager.get_llm(),
            prompt=PromptTemplate(
                template="""你是一个查询改写助手，请为原查询生成3个不同的改写版本，提升检索召回率。
原查询：{question}
改写要求：
1. 保持语义不变
2. 用词/句式不同
3. 每行一个改写结果

改写结果：""",
                input_variables=["question"]
            )
        )

    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        """多查询检索"""
        docs = self.multi_query_retriever.invoke(query)
        # 去重
        unique_docs = []
        seen_content = set()
        for doc in docs[:k]:
            content = doc.page_content[:100]
            if content not in seen_content:
                seen_content.add(content)
                unique_docs.append(doc)
        return unique_docs

    def get_retriever(self):
        return self.multi_query_retriever
