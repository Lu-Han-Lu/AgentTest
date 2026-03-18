# demo/langchaintest/myproject0317V1/core/rag/rag_chat.py
"""RAG对话：复用通用记忆管理和检索接口"""
from typing import Dict, Any
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from demo.langchaintest.myproject0317V1.core.base.llm_base import create_llm
from demo.langchaintest.myproject0317V1.core.base.memory_manager import MemoryManager
from demo.langchaintest.myproject0317V1.core.base.retriever_base import EnhancedRetriever
from demo.langchaintest.myproject0317V1.utils.llm_utils import prepare_inputs, clean_answer, display_result
from demo.langchaintest.myproject0317V1.utils.prompt_utils import create_rag_prompt_with_memory


class RAGChat:
    """RAG对话：复用通用基础层"""

    def __init__(
            self,
            retriever: EnhancedRetriever,
            memory_manager: MemoryManager = None,
            llm: ChatOpenAI = None,
            use_long_term_memory: bool = True
    ):
        self.retriever = retriever
        self.memory_manager = memory_manager or MemoryManager()
        self.llm = llm or create_llm()
        self.use_long_term_memory = use_long_term_memory
        self.rag_chain = self._build_rag_chain()

    def _build_rag_chain(self):
        """构建增强RAG链（复用通用检索和记忆）"""
        prompt = create_rag_prompt_with_memory()

        def retrieve_and_prepare(inputs):
            """统一检索和输入准备（复用通用层）"""
            question = inputs['question']
            user_id = inputs.get('user_id', 'default')

            # 1. 获取记忆（会话+长期）
            history_messages = self.memory_manager.convert_memory_to_messages(
                user_id=user_id,
                include_long_term=self.use_long_term_memory,
                query=question
            )

            # 2. 检索文档
            docs = self.retriever.retrieve(question)

            # 3. 准备输入
            prepared = prepare_inputs({"question": question, "docs": docs})
            prepared['history'] = history_messages

            return prepared

        return (
                RunnableLambda(retrieve_and_prepare)
                | RunnableParallel(
            answer=(lambda x: x) | prompt | self.llm | StrOutputParser(),
            sources=(lambda x: x["sources"]),
            retrieved_docs=(lambda x: x["retrieved_docs"]),
            history=(lambda x: x["history"])
        )
        )

    def chat(self, user_input: str, user_id: str = "default", display: bool = True) -> Dict[str, Any]:
        """单轮对话（复用通用记忆管理）"""
        # 1. 调用RAG链
        result = self.rag_chain.invoke({
            "question": user_input,
            "user_id": user_id
        })

        # 2. 清理答案
        answer = clean_answer(result["answer"])

        # 3. 显示结果
        if display:
            display_result(user_input, result)

        # 4. 更新记忆（会话+长期）
        self.memory_manager.add_session_memory(user_id, user_input, answer)

        # 5. 提炼并添加长期记忆（可选）
        if self.use_long_term_memory:
            # 简单提炼：取答案的核心内容（可替换为LLM提炼）
            long_term_content = f"用户问：{user_input}\n助手答：{answer}"
            self.memory_manager.add_long_term_memory(user_id, long_term_content)

        return {
            "answer": answer,
            "sources": result["sources"],
            "retrieved_docs": result["retrieved_docs"]
        }

    # 复用通用记忆管理的方法（透传）
    def clear_session_memory(self, user_id: str = "default"):
        self.memory_manager.clear_session_memory(user_id)

    def clear_long_term_memory(self, user_id: str = "default"):
        self.memory_manager.clear_long_term_memory(user_id)

    def start_chat_loop(self):
        """启动交互式对话（逻辑不变，仅底层调用通用层）"""
        current_user = "default"
        print("\n开始增强版RAG多轮对话（支持长期记忆/混合检索）")
        print("输入 'user:<用户名>' 切换用户，输入 'exit' 退出")

        while True:
            user_input = input(f"\n[{current_user}] 用户: ").strip()

            if user_input.lower() in ['exit', 'quit']:
                break

            if user_input.startswith("user:"):
                current_user = user_input[5:].strip() or "default"
                print(f"已切换到用户: {current_user}")
                continue

            if not user_input:
                continue

            self.chat(user_input, current_user)