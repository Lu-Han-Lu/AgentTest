# demo/langchaintest/myproject0317V1/core/base/tool_base.py
"""统一工具接口：供Agent复用，也可扩展RAG的工具能力"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from langchain_core.tools import BaseTool, Tool
from langchain_core.callbacks import CallbackManagerForToolRun


class BaseAgentTool(ABC):
    """Agent工具基类：定义统一接口"""
    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述"""
        pass

    @abstractmethod
    def run(self, query: str, **kwargs) -> str:
        """工具执行方法"""
        pass

    def to_langchain_tool(self) -> BaseTool:
        """转换为LangChain Tool对象（统一适配Agent）"""
        def _run(
            query: str,
            run_manager: CallbackManagerForToolRun = None,
            **kwargs
        ) -> str:
            return self.run(query, **kwargs)

        return Tool(
            name=self.name,
            description=self.description,
            func=_run
        )


# 工具注册中心（供Agent复用）
class ToolRegistry:
    """工具注册中心：统一管理所有工具"""
    def __init__(self):
        self.tools: Dict[str, BaseAgentTool] = {}

    def register_tool(self, tool: BaseAgentTool):
        """注册工具"""
        self.tools[tool.name] = tool

    def unregister_tool(self, tool_name: str):
        """注销工具"""
        if tool_name in self.tools:
            del self.tools[tool_name]

    def get_tool(self, tool_name: str) -> BaseAgentTool:
        """获取工具"""
        return self.tools.get(tool_name)

    def get_all_tools(self) -> List[BaseTool]:
        """获取所有LangChain格式的工具"""
        return [tool.to_langchain_tool() for tool in self.tools.values()]

    def get_tool_names(self) -> List[str]:
        """获取所有工具名称"""
        return list(self.tools.keys())