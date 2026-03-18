# demo/langchaintest/myproject0317V1/utils/prompt_utils.py
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def create_rag_prompt() -> ChatPromptTemplate:
    """创建基础RAG提示模板"""
    return ChatPromptTemplate.from_messages([
        ("system", """你是一个技术文档问答助手。请严格根据提供的参考资料回答用户问题。
如果参考资料中没有相关信息，请说"根据已有资料，我无法回答这个问题"。
不要编造信息。"""),
        ("user", """参考资料：
{context}

用户问题：{question}

请基于参考资料回答："""),
    ])


def create_rag_prompt_with_memory() -> ChatPromptTemplate:
    """创建带记忆的RAG提示模板"""
    return ChatPromptTemplate.from_messages([
        ("system", """你是一个智能助手，可以处理多种类型的请求，请根据用户问题的性质选择恰当的回应方式：

1. **技术文档问答**  
   当用户问题与提供的参考资料相关时，请严格依据参考资料回答。如果参考资料中找不到相关信息，请回复“根据已有资料，我无法回答这个问题”，切勿编造信息。

2. **日常对话**  
   如果用户进行问候、感谢、闲聊或提出与资料无关的一般性问题，请以友好、自然的方式直接回应，无需参考参考资料。

3. **工具调用（开发中）**  
   未来你将能够调用外部工具来辅助回答，但目前该功能尚未启用，请勿尝试调用或假装调用工具。

请始终遵循上述原则。如果问题模糊不清，可以礼貌地请求用户澄清。"""),
        MessagesPlaceholder(variable_name="history"),
        ("user", """当前用户问题：{question}

参考资料：
{context}

请回答："""),
    ])

def create_agent_prompt() -> ChatPromptTemplate:
    """
    创建支持工具调用的 Agent 提示模板
    """
    today = datetime.today().strftime("%Y-%m-%d")

    return ChatPromptTemplate.from_messages([
        ("system", f"""你是一个智能助手，可以处理多种类型的请求。今天的日期是 {today}。

你有以下能力：
1. **技术文档问答**：当用户问题与提供的参考资料相关时，请严格依据参考资料回答。
2. **联网搜索**：当用户需要实时信息、最新事件，或本地文档无法回答时，使用 web_search 工具搜索互联网。
3. **日常对话**：对于问候、感谢等闲聊，直接友好回应。

使用工具时，请遵循：
- 如果问题需要最新信息，优先使用 web_search
- 如果问题涉及本地文档，结合文档内容回答
- 可以在一次回答中同时使用文档知识和搜索结果

请根据用户问题的性质选择合适的回应方式。"""),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),  # 工具调用暂存区
    ])