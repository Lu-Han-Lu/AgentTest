"""联网搜索工具：适配BaseAgentTool接口"""
import os
from langchain_tavily import TavilySearch

from demo.langchaintest.myproject0317V1.core.base.tool_base import BaseAgentTool


class WebSearchTool(BaseAgentTool):
    """联网搜索工具：基于Tavily"""

    def __init__(self, max_results: int = 3, search_depth: str = "basic"):
        self.max_results = max_results
        self.search_depth = search_depth
        self._name = "web_search"
        self._description = """
        用于获取实时/最新的互联网信息，适用于：
        1. 实时事件、新闻、天气、股价等时效性问题
        2. 最新技术动态、版本更新、行业资讯
        3. 本地文档中没有的信息
        输入：用户的查询问题（简洁明了）
        输出：相关的搜索结果（标题+链接+摘要）
        """
        # 初始化Tavily工具
        self._validate_api_key()
        self.tavily_tool = TavilySearch(
            max_results=self.max_results,
            search_depth=self.search_depth,
            include_raw_content=False,
            include_images=False
        )

    def _validate_api_key(self):
        """验证API密钥"""
        if not os.environ.get("TAVILY_API_KEY"):
            raise EnvironmentError("请设置 TAVILY_API_KEY 环境变量（从https://app.tavily.com获取）")

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def run(self, query: str, **kwargs) -> str:
        """执行联网搜索"""
        try:
            # 调用Tavily搜索
            results = self.tavily_tool.invoke({"query": query})

            # 格式化结果
            formatted_results = []
            for i, result in enumerate(results):
                formatted_results.append(
                    f"【搜索结果 {i + 1}】\n"
                    f"标题：{result.get('title', '无')}\n"
                    f"链接：{result.get('url', '无')}\n"
                    f"摘要：{result.get('content', '无')}\n"
                )

            if not formatted_results:
                return "未检索到相关的互联网信息"

            return "\n".join(formatted_results)

        except Exception as e:
            return f"联网搜索失败：{str(e)[:100]}"


# 便捷实例化
web_search_tool = WebSearchTool(max_results=3, search_depth="basic")
