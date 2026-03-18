"""日期时间工具：查询当前时间、计算日期差、节假日等"""
import re
from datetime import datetime, timedelta

from demo.langchaintest.myproject0317V1.core.base.tool_base import BaseAgentTool


class DatetimeTool(BaseAgentTool):
    def __init__(self):
        self._name = "datetime_tool"
        self._description = """
        用于日期时间相关查询，支持：
        1. 查询当前时间/今天日期
        2. 查询明天/昨天日期
        3. 计算两个日期的差值（如：2026-03-17和2026-04-01差几天）
        输入格式示例：当前时间、明天日期、2026-03-17到2026-04-01差几天
        """
        # Mock 节假日数据（2026年）
        self.holidays = {
            "2026-01-01": "元旦",
            "2026-02-17": "春节",
            "2026-04-04": "清明节",
            "2026-05-01": "劳动节"
        }

    def _parse_datetime_query(self, query: str) -> str:
        """解析并处理日期查询"""
        now = datetime.now()

        # 查询当前时间/日期
        if "当前时间" in query:
            return f"当前时间：{now.strftime('%Y-%m-%d %H:%M:%S')}"
        elif "今天日期" in query:
            return f"今天日期：{now.strftime('%Y-%m-%d')}（{['一', '二', '三', '四', '五', '六', '日'][now.weekday()]}）"

        # 查询明天/昨天
        elif "明天日期" in query:
            tomorrow = now + timedelta(days=1)
            return f"明天日期：{tomorrow.strftime('%Y-%m-%d')}（{['一', '二', '三', '四', '五', '六', '日'][tomorrow.weekday()]}）"
        elif "昨天日期" in query:
            yesterday = now - timedelta(days=1)
            return f"昨天日期：{yesterday.strftime('%Y-%m-%d')}（{['一', '二', '三', '四', '五', '六', '日'][yesterday.weekday()]}）"

        # 计算日期差
        date_pattern = r'(\d{4}-\d{2}-\d{2})'
        dates = re.findall(date_pattern, query)
        if len(dates) == 2:
            try:
                date1 = datetime.strptime(dates[0], "%Y-%m-%d")
                date2 = datetime.strptime(dates[1], "%Y-%m-%d")
                diff = abs((date2 - date1).days)
                return f"{dates[0]} 和 {dates[1]} 相差 {diff} 天"
            except:
                return "日期格式错误，请使用：YYYY-MM-DD"

        # 查询节假日
        elif "节假日" in query or "节日" in query:
            date = re.search(date_pattern, query)
            if date:
                holiday = self.holidays.get(date.group(), "非节假日")
                return f"{date.group()}：{holiday}"
            else:
                return "支持查询指定日期的节假日，示例：2026-01-01是什么节日"

        else:
            return "支持查询：当前时间、今天/明天/昨天日期、日期差、指定日期节假日"

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def run(self, query: str, **kwargs) -> str:
        return self._parse_datetime_query(query)


# 实例化（可直接注册）
datetime_tool = DatetimeTool()