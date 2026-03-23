# demo/langchaintest/myproject0317V1/core/agent/agent_reflector.py
"""Agent反思模块：实现Self-RAG/自我修正，对标业界标杆"""
from typing import Dict, Any, List
from langchain_core.prompts import PromptTemplate

from demo.langchaintest.myproject0317V1.core.base.llm_base import LLMManager


class AgentReflector:
    """Agent反思器：检查回答质量并自我修正"""

    def __init__(self, llm_manager: LLMManager = None):
        self.llm_manager = llm_manager or LLMManager()
        # 反思提示词
        self.reflect_prompt = PromptTemplate(
            template="""你是一个AI助手的质量审核员，请完成以下任务：
1. 检查回答是否符合用户问题要求
2. 检查回答是否准确、完整、无编造信息
3. 检查工具调用是否必要且有效
4. 给出修正建议（如果需要）

用户问题：{query}
助手回答：{answer}
工具调用记录：{tool_calls}
检索到的文档：{context}

请按照以下格式输出：
【是否需要修正】：是/否
【修正原因】：简要说明问题
【修正后的回答】：如果需要修正，请给出修正后的回答；否则保持原回答

注意：仅在确实需要修正时才修改，不要无故改写。""",
            input_variables=["query", "answer", "tool_calls", "context"]
        )

    def reflect_and_correct(
            self,
            query: str,
            answer: str,
            tool_calls: List[Any] = None,
            context: List[str] = None
    ) -> Dict[str, Any]:
        """执行反思并修正回答"""
        # 格式化输入
        tool_calls_str = "\n".join([
            f"工具：{tc[0].tool}，输入：{tc[0].tool_input}，输出：{tc[1]}"
            for tc in tool_calls or []
        ]) or "无"

        context_str = "\n".join(context or []) or "无"

        # 构建反思提示
        prompt = self.reflect_prompt.format(
            query=query,
            answer=answer,
            tool_calls=tool_calls_str,
            context=context_str
        )

        # 调用LLM进行反思
        reflection = self.llm_manager.invoke(prompt)

        # 解析反思结果
        result = self._parse_reflection(reflection, answer)
        return {
            "original_answer": answer,
            "corrected_answer": result["corrected"],
            "need_correction": result["need_correction"],
            "reason": result["reason"],
            "reflection_raw": reflection
        }

    def _parse_reflection(self, reflection: str, original_answer: str) -> Dict[str, Any]:
        """解析反思结果"""
        lines = reflection.split("\n")
        need_correction = False
        reason = ""
        corrected_answer = original_answer

        for line in lines:
            line = line.strip()
            if "【是否需要修正】" in line:
                need_correction = "是" in line
            elif "【修正原因】" in line:
                reason = line.split("：", 1)[1] if "：" in line else ""
            elif "【修正后的回答】" in line:
                if need_correction and len(line.split("：", 1)) > 1:
                    corrected_answer = line.split("：", 1)[1]

        return {
            "need_correction": need_correction,
            "reason": reason,
            "corrected": corrected_answer
        }


# 集成反思能力的Agent扩展
class ReflectiveToolAgent:
    """带反思能力的ToolAgent扩展"""

    def __init__(self, tool_agent, reflector: AgentReflector = None):
        self.tool_agent = tool_agent
        self.reflector = reflector or AgentReflector()

    def run_with_reflection(
            self,
            user_input: str,
            user_id: str = "default",
            context_docs: List[str] = None
    ) -> Dict[str, Any]:
        """运行Agent并执行反思修正"""
        # 1. 执行原始Agent调用
        agent_result = self.tool_agent.run(user_input, user_id)

        # 2. 执行反思修正
        reflect_result = self.reflector.reflect_and_correct(
            query=user_input,
            answer=agent_result["answer"],
            tool_calls=agent_result["intermediate_steps"],
            context=[doc.page_content for doc in context_docs] if context_docs else None
        )

        # 3. 更新记忆（使用修正后的回答）
        self.tool_agent.memory_manager.add_session_memory(
            user_id,
            user_input,
            reflect_result["corrected_answer"]
        )

        # 4. 合并结果
        return {
            **agent_result,
            "reflect_result": reflect_result,
            "final_answer": reflect_result["corrected_answer"]
        }





























