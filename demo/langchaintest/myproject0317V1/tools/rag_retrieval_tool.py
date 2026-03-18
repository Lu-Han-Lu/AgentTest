"""将RAG检索封装为Agent可调用的工具"""
from demo.langchaintest.myproject0317V1.core.base.retriever_base import EnhancedRetriever
from demo.langchaintest.myproject0317V1.core.base.tool_base import BaseAgentTool


class RAGRetrievalTool(BaseAgentTool):
    """RAG检索工具：供Agent调用"""

    def __init__(self, retriever: EnhancedRetriever):
        self.retriever = retriever
        self._name = "rag_retrieval"
        self._description = """
        用于检索本地文档中的信息，适用于：
        1. 技术文档问答
        2. 本地知识库查询
        3. 非实时性的专业问题
        输入：用户的查询问题
        输出：相关的文档内容片段
        """

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def run(self, query: str, **kwargs) -> str:
        """执行RAG检索"""
        docs = self.retriever.retrieve(query)
        if not docs:
            return "未检索到相关文档内容"

        # 格式化检索结果
        formatted_docs = "\n\n".join([
            f"【文档片段 {i + 1}】：{doc.page_content}\n【来源】：{doc.metadata.get('source', '未知')}"
            for i, doc in enumerate(docs)
        ])

        return f"本地文档检索结果：\n{formatted_docs}"