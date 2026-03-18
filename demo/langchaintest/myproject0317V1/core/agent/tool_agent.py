# demo/langchaintest/myproject0317V1/core/agent/tool_agent.py
"""Agent对话：复用通用记忆/工具/检索接口，集成RAG工具"""
from typing import List, Dict, Any, Optional

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent

from demo.langchaintest.myproject0317V1.core.base.llm_base import create_llm
from demo.langchaintest.myproject0317V1.core.base.memory_manager import MemoryManager
from demo.langchaintest.myproject0317V1.core.base.tool_base import ToolRegistry
from demo.langchaintest.myproject0317V1.utils.llm_utils import clean_answer
from demo.langchaintest.myproject0317V1.utils.prompt_utils import create_agent_prompt


class ToolAgent:
    """增强版Agent：复用通用基础层，集成RAG工具"""

    def __init__(
            self,
            tool_registry: ToolRegistry,
            memory_manager: MemoryManager = None,
            llm: ChatOpenAI = None,
            max_iterations: int = 3,
            use_long_term_memory: bool = True
    ):
        self.tool_registry = tool_registry
        self.memory_manager = memory_manager or MemoryManager()
        self.llm = llm or create_llm()
        self.max_iterations = max_iterations
        self.use_long_term_memory = use_long_term_memory
        self.prompt = create_agent_prompt()

    def create_agent_executor(self, user_id: str, query: str) -> AgentExecutor:
        """创建Agent执行器（注入记忆）"""
        # 1. 获取所有工具
        tools = self.tool_registry.get_all_tools()

        # 2. 创建Agent
        agent = create_openai_tools_agent(self.llm, tools, self.prompt)

        # 3. 创建执行器（增强错误处理）
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=self._handle_parsing_errors,
            max_iterations=self.max_iterations,
            return_intermediate_steps=True  # 返回中间步骤，便于调试
        )

    def _handle_parsing_errors(self, error: Exception) -> str:
        """增强错误处理（复用通用逻辑）"""
        print(f"Agent执行错误：{error}")
        return f"处理你的请求时出现小问题，请重试：{str(error)[:100]}"

    def run(self, user_input: str, user_id: str = "default") -> Dict[str, Any]:
        """运行Agent（复用通用记忆管理）"""
        # 1. 构建消息（包含会话+长期记忆）
        messages = self.memory_manager.convert_memory_to_messages(
            user_id=user_id,
            include_long_term=self.use_long_term_memory,
            query=user_input
        )
        messages.append(HumanMessage(content=user_input))

        # 2. 创建Agent执行器
        agent_executor = self.create_agent_executor(user_id, user_input)

        # 3. 运行Agent
        response = agent_executor.invoke({
            "messages": messages,
            "agent_scratchpad": [],
            "user_id": user_id
        })

        # 4. 清理答案
        answer = clean_answer(response["output"])

        # 5. 更新记忆
        self.memory_manager.add_session_memory(user_id, user_input, answer)

        # 6. 提炼长期记忆
        if self.use_long_term_memory:
            # 提取工具调用结果中的关键信息
            intermediate_steps = response.get("intermediate_steps", [])
            tool_info = "\n".join([
                f"使用工具 {step[0].tool}：{step[0].tool_input} → {step[1]}"
                for step in intermediate_steps
            ]) if intermediate_steps else ""
            long_term_content = f"用户问：{user_input}\n助手答：{answer}\n工具调用：{tool_info}"
            self.memory_manager.add_long_term_memory(user_id, long_term_content)

        return {
            "answer": answer,
            "intermediate_steps": response.get("intermediate_steps", []),
            "raw_response": response
        }

    # 复用通用记忆管理方法
    def clear_session_memory(self, user_id: str = "default"):
        self.memory_manager.clear_session_memory(user_id)

    def clear_long_term_memory(self, user_id: str = "default"):
        self.memory_manager.clear_long_term_memory(user_id)

    def start_chat_loop(self):
        """启动Agent对话（集成RAG工具）"""
        current_user = "default"
        print("\n开始增强版Agent多轮对话（支持RAG检索/长期记忆/多工具）")
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

            # 运行Agent
            result = self.run(user_input, current_user)

            # 输出结果
            print(f"助手: {result['answer']}")
            if result.get("intermediate_steps"):
                print(f"[工具调用] 共执行 {len(result['intermediate_steps'])} 步工具调用")