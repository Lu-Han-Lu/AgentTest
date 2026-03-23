"""天气查询工具（Mock 版）：适配BaseAgentTool接口，无第三方API依赖"""
import re

from demo.langchaintest.myproject0317V1.core.base.tool_base import BaseAgentTool


class WeatherTool(BaseAgentTool):
    """天气查询工具（Mock版）：支持查询指定城市今天/明天的天气"""

    def __init__(self):
        # 工具基本信息（Agent 根据 description 判断是否调用）
        self._name = "weather_query"
        self._description = """
        用于查询指定城市的天气信息，适用于：
        1. 查询国内主流城市（北京/上海/深圳/广州）的今天/明天天气
        2. 输入格式示例："北京今天的天气"、"上海明天天气"、"深圳天气"
        输出：该城市指定日期的天气情况
        """
        # Mock 天气数据
        self.weather_data = {
            "北京": {"今天": "晴，15-25°C", "明天": "多云，16-26°C"},
            "上海": {"今天": "多云，18-28°C", "明天": "小雨，19-27°C"},
            "深圳": {"今天": "阵雨，22-30°C", "明天": "阴，23-31°C"},
            "广州": {"今天": "雷阵雨，23-32°C", "明天": "多云，24-33°C"},
        }

    def _parse_query(self, query: str) -> tuple[str, str]:
        """解析用户查询，提取城市和日期"""
        # 清理无关字符
        query = query.strip().replace("天气", "").replace("查询", "").replace("的", "").replace("？", "").replace("吗",
                                                                                                                 "")

        # 提取日期（今天/明天）
        date_pattern = r"(今天|明天)"
        date_match = re.search(date_pattern, query)
        date = date_match.group(1) if date_match else "今天"

        # 提取城市（北京/上海/深圳/广州）
        city_pattern = r"(北京|上海|深圳|广州)"
        city_match = re.search(city_pattern, query)
        city = city_match.group(1) if city_match else ""

        return city, date

    def get_weather(self, city: str, date: str = "今天") -> str:
        """核心逻辑：查询指定城市的天气情况（复用你提供的 Mock 函数）"""
        if city not in self.weather_data:
            return f"抱歉，找不到{city}的天气信息"

        weather = self.weather_data[city].get(date, "未知")
        return f"{city}{date}天气: {weather}"

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def run(self, query: str, **kwargs) -> str:
        """执行天气查询（Agent 调用的核心方法）"""
        # 解析查询中的城市和日期
        city, date = self._parse_query(query)

        # 容错：未识别到城市
        if not city:
            return "请明确指定要查询的城市（支持：北京/上海/深圳/广州），例如：北京今天的天气"

        # 调用 Mock 方法获取天气
        return self.get_weather(city, date)


# 便捷实例化
weather_tool = WeatherTool()