# demo/langchaintest/myproject0317V1/core/base/memory_manager.py
"""统一记忆管理：支持会话记忆+长期记忆，供RAG和Agent复用"""
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime
import json
import os

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from demo.langchaintest.myproject0317V1.utils.embedding_utils import create_embeddings


class MemoryManager:
    """统一记忆管理器：
    - 会话记忆：短期、小窗口、高优先级
    - 长期记忆：持久化、向量检索、低优先级
    """
    def __init__(
        self,
        long_term_memory_path: str = "./long_term_memory",
        session_memory_max_rounds: int = 5,
        embedding_model_name: str = None,
        embedding_device: str = None
    ):
        # 会话记忆（内存）：{user_id: [(q, a, timestamp), ...]}
        self.session_memories: Dict[str, List[Tuple[str, str, float]]] = {}
        self.session_max_rounds = session_memory_max_rounds

        # 长期记忆（持久化向量库）
        self.long_term_path = long_term_memory_path
        self.embeddings = create_embeddings(embedding_model_name, embedding_device)
        self.long_term_vectorstore = self._init_long_term_memory()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

    def _init_long_term_memory(self) -> Chroma:
        """初始化长期记忆向量库"""
        if os.path.exists(self.long_term_path) and os.listdir(self.long_term_path):
            return Chroma(
                persist_directory=self.long_term_path,
                embedding_function=self.embeddings
            )
        else:
            return Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.long_term_path
            )

    # ========== 会话记忆操作（RAG/Agent 复用） ==========
    def get_session_memory(self, user_id: str = "default") -> List[Tuple[str, str]]:
        """获取会话记忆（仅返回q/a，去掉timestamp）"""
        if user_id not in self.session_memories:
            self.session_memories[user_id] = []
        # 仅返回(q,a)元组，适配原有接口
        return [(q, a) for q, a, _ in self.session_memories[user_id]]

    def add_session_memory(self, user_id: str, question: str, answer: str):
        """添加会话记忆"""
        if user_id not in self.session_memories:
            self.session_memories[user_id] = []
        # 记录时间戳，用于排序/清理
        self.session_memories[user_id].append((question, answer, datetime.now().timestamp()))
        # 限制最大轮数
        if len(self.session_memories[user_id]) > self.session_max_rounds:
            self.session_memories[user_id] = self.session_memories[user_id][-self.session_max_rounds:]

    def clear_session_memory(self, user_id: str = "default"):
        """清空会话记忆"""
        self.session_memories[user_id] = []

    # ========== 长期记忆操作（RAG/Agent 复用） ==========
    def add_long_term_memory(self, user_id: str, content: str, memory_type: str = "chat"):
        """添加长期记忆（提炼后的关键信息）"""
        # 容错：空内容不存储
        if not content or not isinstance(content, str):
            return

        # 切分为小片段
        chunks = self.text_splitter.split_text(content)

        # 创建Document对象（关键修复：替换字典）
        docs = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk,
                metadata={
                    "user_id": user_id,
                    "type": memory_type,
                    "timestamp": datetime.now().timestamp(),
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            )
            docs.append(doc)

        # 添加到向量库（新版Chroma自动持久化，移除persist()）
        self.long_term_vectorstore.add_documents(docs)

    def retrieve_long_term_memory(self, user_id: str, query: str, k: int = 3) -> List[str]:
        """检索长期记忆"""
        # 过滤当前用户的记忆，按时间衰减排序
        results = self.long_term_vectorstore.similarity_search(
            query=query,
            k=k,
            filter={"user_id": user_id}
        )
        # 提取记忆内容
        return [doc.page_content for doc in results]

    def clear_long_term_memory(self, user_id: str):
        """清空指定用户的长期记忆"""
        # 查询用户所有记忆ID并删除
        all_docs = self.long_term_vectorstore.get(where={"user_id": user_id})
        if all_docs["ids"]:
            self.long_term_vectorstore.delete(ids=all_docs["ids"])
            # 移除persist()调用，新版Chroma自动持久化

    # ========== 统一记忆转换（RAG/Agent 复用） ==========
    def convert_memory_to_messages(self, user_id: str, include_long_term: bool = False, query: str = "") -> List[BaseMessage]:
        """
        统一将记忆转换为LangChain消息格式
        :param user_id: 用户ID
        :param include_long_term: 是否包含长期记忆
        :param query: 当前查询（用于检索长期记忆）
        :return: 消息列表
        """
        messages = []

        # 1. 添加会话记忆
        session_memory = self.get_session_memory(user_id)
        for q, a in session_memory:
            messages.append(HumanMessage(content=q))
            messages.append(AIMessage(content=a))

        # 2. 添加长期记忆（如果开启）
        if include_long_term and query:
            long_term_memory = self.retrieve_long_term_memory(user_id, query)
            if long_term_memory:
                memory_text = "\n\n【历史长期记忆】：\n" + "\n".join(long_term_memory)
                messages.append(AIMessage(content=memory_text))

        return messages