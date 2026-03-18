"""文本处理工具：摘要、关键词提取、中英文翻译"""
import re

from demo.langchaintest.myproject0317V1.core.base.llm_base import LLMManager
from demo.langchaintest.myproject0317V1.core.base.tool_base import BaseAgentTool


class TextProcessTool(BaseAgentTool):
    def __init__(self):
        self._name = "text_process"
        self._description = """
        用于文本处理，支持：
        1. 文本摘要（生成100字内的摘要）
        2. 关键词提取（提取3-5个关键词）
        3. 中英文互译（如：翻译为英文、翻译成中文）
        输入格式示例：
        - 摘要：请总结这段文字：[文本内容]
        - 关键词：提取关键词：[文本内容]
        - 翻译：翻译为英文：你好世界
        """
        self.llm_manager = LLMManager()

    def _process_text(self, query: str) -> str:
        """处理文本请求"""
        # 提取核心文本（去掉指令部分）
        content = ""
        if "总结" in query or "摘要" in query:
            content = re.sub(r'总结|摘要|请总结|请生成摘要|：', '', query).strip()
            prompt = f"请总结以下文本，控制在100字以内：\n{content}"
            return f"文本摘要：\n{self.llm_manager.invoke(prompt)}"

        elif "关键词" in query or "提取" in query:
            content = re.sub(r'提取|关键词|：', '', query).strip()
            prompt = f"请提取以下文本的3-5个核心关键词：\n{content}"
            return f"核心关键词：\n{self.llm_manager.invoke(prompt)}"

        elif "翻译" in query:
            content = re.sub(r'翻译|翻译成|为|：', '', query).strip()
            if "英文" in query:
                prompt = f"请将以下文本翻译成英文：\n{content}"
                return f"英文翻译：\n{self.llm_manager.invoke(prompt)}"
            else:
                prompt = f"请将以下文本翻译成中文：\n{content}"
                return f"中文翻译：\n{self.llm_manager.invoke(prompt)}"

        else:
            return "请明确文本处理类型：摘要、关键词提取、中英文翻译"

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def run(self, query: str, **kwargs) -> str:
        if not query:
            return "请输入需要处理的文本内容"
        return self._process_text(query)


# 实例化（可直接注册）
text_process_tool = TextProcessTool()