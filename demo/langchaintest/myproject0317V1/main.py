# demo/langchaintest/myproject0317V1/main.py
"""统一入口：演示RAG和Agent的底层打通"""
import os

from demo.langchaintest.myproject0317V1.config import USER_AGENT, VECTOR_STORE_PATH, LONG_TERM_MEMORY_PATH, \
    DOCUMENT_PATH

os.environ['USER_AGENT'] = USER_AGENT

from demo.langchaintest.myproject0317V1.core.agent.tool_agent import ToolAgent
from demo.langchaintest.myproject0317V1.core.base.memory_manager import MemoryManager
from demo.langchaintest.myproject0317V1.core.base.retriever_base import EnhancedRetriever
from demo.langchaintest.myproject0317V1.core.base.tool_base import ToolRegistry
from demo.langchaintest.myproject0317V1.core.document_manager import DocumentManager
from demo.langchaintest.myproject0317V1.core.rag.rag_chat import RAGChat
from demo.langchaintest.myproject0317V1.tools.calculate_tool import calculate_tool
from demo.langchaintest.myproject0317V1.tools.city_info_tool import city_info_tool
from demo.langchaintest.myproject0317V1.tools.datetime_tool import datetime_tool
from demo.langchaintest.myproject0317V1.tools.rag_retrieval_tool import RAGRetrievalTool
from demo.langchaintest.myproject0317V1.tools.text_process_tool import text_process_tool
from demo.langchaintest.myproject0317V1.tools.todo_tool import todo_tool
from demo.langchaintest.myproject0317V1.tools.weather_tool import weather_tool
from demo.langchaintest.myproject0317V1.tools.web_search_tool import web_search_tool

if __name__ == "__main__":
    # 1. 初始化通用组件（RAG和Agent复用）
    doc_manager = DocumentManager(VECTOR_STORE_PATH)
    memory_manager = MemoryManager(LONG_TERM_MEMORY_PATH)
    tool_registry = ToolRegistry()

    # 2. 加载文档（仅需执行一次）
    source_paths = [DOCUMENT_PATH]
    documents = doc_manager.load_documents(source_paths)
    chunks = doc_manager.split_documents(documents)
    doc_manager.update_vectorstore(chunks)

    # 3. 初始化增强检索器（RAG和Agent复用）
    enhanced_retriever = EnhancedRetriever(doc_manager)

    # 4. 注册工具（包含RAG检索工具）
    tool_registry.register_tool(web_search_tool)  # 联网搜索工具
    tool_registry.register_tool(RAGRetrievalTool(enhanced_retriever))  # RAG检索工具
    tool_registry.register_tool(weather_tool)  # 天气工具
    tool_registry.register_tool(calculate_tool)  # 计算工具
    tool_registry.register_tool(datetime_tool)  # 日期工具
    tool_registry.register_tool(text_process_tool)  # 文本处理工具
    tool_registry.register_tool(city_info_tool)  # 城市信息工具
    tool_registry.register_tool(todo_tool)  # 待办工具

    # 5. 初始化RAG和Agent（复用所有通用组件）
    rag_chat = RAGChat(enhanced_retriever, memory_manager)
    tool_agent = ToolAgent(tool_registry, memory_manager)

    # 6. 选择运行模式
    mode = input("请选择运行模式（1-RAG对话，2-Agent对话）：").strip()
    if mode == "1":
        rag_chat.start_chat_loop()
    elif mode == "2":
        tool_agent.start_chat_loop()
    else:
        print("无效模式")
